#!/usr/bin/env python3
"""
Boltz-2 Integration for Protein-Molecule Interaction Prediction
===============================================================

This module integrates Boltz-2 structural predictions with the
ProteinMoleculeModel to enable two complementary workflows:

**Mode 1: Direct Inference (BoltzP2MPredictor)**
    Uses the Boltz-2 CLI to generate structure predictions and extract
    confidence metrics (affinity, pLDDT, ipTM) for a given protein–ligand
    pair. Results are cached by MD5 hash to avoid redundant computation.

    Usage::

        predictor = BoltzP2MPredictor(cache_dir="data/boltz_cache")
        result = predictor.predict(
            protein_seq="MKTAYIAKQ...",
            smiles="CC(=O)Oc1ccccc1C(=O)O",
        )
        # result keys: affinity_pred_value, affinity_probability_binary,
        #              ligand_iptm, complex_plddt, confidence_score,
        #              structure_path

        # Or from the CLI:
        python model_boltz_p2m.py \\
            --predict \\
            --protein "MKTAYIAKQ..." \\
            --smiles "CC(=O)Oc1ccccc1C(=O)O" \\
            --use-msa-server

**Mode 2: Hybrid Training (HybridP2MModel + BoltzEnhancedDataset)**
    Combines pre-cached Boltz-2 structural features with a trained
    ProteinMoleculeModel via a small MLP fusion network. The cache is
    populated ahead of time by ``BoltzP2MPredictor.predict_batch()``.

    Usage::

        # Pre-populate cache (run once, can take hours):
        predictor = BoltzP2MPredictor(cache_dir="data/boltz_cache")
        predictor.predict_batch(protein_seqs, smiles_list)

        # Then train the hybrid model:
        python model_boltz_p2m.py \\
            --data-dir data/protein_molecule \\
            --boltz-cache data/boltz_cache \\
            --base-checkpoint models/p2m_model.pt \\
            --epochs 30
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from model.model_p2m import (
        ProteinMoleculeModel,
        encode_smiles,
        SMILES_PAD_IDX,
    )
except ImportError:
    from model_p2m import (
        ProteinMoleculeModel,
        encode_smiles,
        SMILES_PAD_IDX,
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _md5(text: str) -> str:
    """Return the MD5 hex digest of a UTF-8 encoded string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _boltz_key(protein_seq: str, smiles: str) -> str:
    """Canonical, filesystem-safe cache key for a (protein, ligand) pair."""
    return _md5(f"{protein_seq}||{smiles}")


# ---------------------------------------------------------------------------
# BoltzP2MPredictor
# ---------------------------------------------------------------------------


class BoltzP2MPredictor:
    """
    Wraps the Boltz-2 CLI (``boltz predict``) to run protein–small-molecule
    structure and affinity predictions.

    Each (protein_seq, smiles) pair is identified by an MD5 hash, and results
    are persisted in ``cache_dir`` so that re-running the same pair is free.
    The .cif structure file is also copied into the cache directory.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for result JSON files and .cif structures.
    use_msa_server : bool
        Whether to pass ``--use_msa_server`` to Boltz-2 (requires internet,
        but produces higher-quality predictions).
    boltz_bin : str
        Name or full path of the Boltz-2 executable (default ``"boltz"``).
    """

    def __init__(
        self,
        cache_dir: str = "data/boltz_cache",
        use_msa_server: bool = True,
        boltz_bin: str = "boltz",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_msa_server = use_msa_server
        self.boltz_bin = boltz_bin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, protein_seq: str, smiles: str) -> Dict:
        """
        Run Boltz-2 for a single (protein, ligand) pair.

        Returns a dict with keys:
            ``affinity_pred_value``, ``affinity_probability_binary``,
            ``ligand_iptm``, ``complex_plddt``, ``confidence_score``,
            ``structure_path``
        """
        key = _boltz_key(protein_seq, smiles)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        result = self._run_boltz(protein_seq, smiles, key)
        self._save_cache(key, result)
        return result

    def predict_batch(
        self,
        protein_seqs: List[str],
        smiles_list: List[str],
    ) -> List[Dict]:
        """
        Run Boltz-2 for a list of (protein, ligand) pairs sequentially.

        Pairs whose results are already cached are returned immediately
        without calling Boltz-2.
        """
        if len(protein_seqs) != len(smiles_list):
            raise ValueError("protein_seqs and smiles_list must have the same length")

        results: List[Dict] = []
        for protein_seq, smiles in tqdm(
            zip(protein_seqs, smiles_list),
            total=len(protein_seqs),
            desc="Boltz-2 batch predictions",
        ):
            results.append(self.predict(protein_seq, smiles))
        return results

    def to_model_output(self, result: Dict) -> Dict[str, torch.Tensor]:
        """
        Convert a Boltz-2 result dict to the same output format as
        ``ProteinMoleculeModel.forward()`` (batch dimension = 1).

        The ``confidence_score`` (0–1) is mapped to a logit so that
        ``sigmoid(logit) ≈ confidence_score``.

        Returns
        -------
        dict with keys:
            ``interaction_logits`` : Tensor [1]
            ``affinity``           : Tensor [1]
        """
        confidence = float(result.get("confidence_score", 0.5))
        # Clamp to avoid log(0); map probability → logit
        confidence = max(min(confidence, 1.0 - 1e-7), 1e-7)
        logit = float(np.log(confidence / (1.0 - confidence)))
        affinity = float(result.get("affinity_pred_value", 0.0))

        return {
            "interaction_logits": torch.tensor([logit], dtype=torch.float32),
            "affinity": torch.tensor([affinity], dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_yaml(self, protein_seq: str, smiles: str) -> str:
        """Compose a Boltz-2 YAML input for one protein–ligand complex."""
        return (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {protein_seq}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: {smiles}\n"
        )

    def _run_boltz(self, protein_seq: str, smiles: str, key: str) -> Dict:
        """
        Write a YAML input, invoke ``boltz predict``, parse the output, and
        copy the .cif structure file to the persistent cache directory.

        Raises ``RuntimeError`` if Boltz-2 exits with a non-zero code.
        """
        with tempfile.TemporaryDirectory(prefix="boltz_p2m_") as raw_tmp:
            tmpdir = Path(raw_tmp)

            # Write input YAML
            input_file = tmpdir / f"{key}.yaml"
            input_file.write_text(self._build_yaml(protein_seq, smiles))

            output_dir = tmpdir / "output"
            output_dir.mkdir()

            # Assemble CLI command
            cmd: List[str] = [
                self.boltz_bin,
                "predict",
                str(input_file),
                "--out_dir", str(output_dir),
                "--model", "boltz2",
            ]
            if self.use_msa_server:
                cmd.append("--use_msa_server")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Boltz-2 exited with code {proc.returncode}.\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )

            return self._parse_output(output_dir, key)

    def _parse_output(self, output_dir: Path, key: str) -> Dict:
        """
        Walk the Boltz-2 output directory and extract prediction metrics.

        Expected layout (Boltz-2 ≥ 1.0)::

            output_dir/predictions/<stem>/
                confidence_model_0.json
                affinity_model_0.json
                <stem>_model_0.cif
        """
        # Locate the per-prediction subdirectory
        pred_root = output_dir / "predictions"
        pred_dirs = list(pred_root.glob("*")) if pred_root.exists() else []
        pred_dir: Path = pred_dirs[0] if pred_dirs else output_dir

        # Confidence JSON
        confidence_data: Dict = {}
        conf_json = pred_dir / "confidence_model_0.json"
        if conf_json.exists():
            with open(conf_json) as fh:
                confidence_data = json.load(fh)

        # Affinity JSON
        affinity_data: Dict = {}
        aff_json = pred_dir / "affinity_model_0.json"
        if aff_json.exists():
            with open(aff_json) as fh:
                affinity_data = json.load(fh)

        # Copy .cif structure into persistent cache
        structure_path: Optional[str] = None
        cif_files = list(pred_dir.glob("*.cif"))
        if cif_files:
            dest = self.cache_dir / f"{key}.cif"
            shutil.copy2(cif_files[0], dest)
            structure_path = str(dest)

        return {
            "affinity_pred_value": float(
                affinity_data.get("affinity_pred_value", 0.0)
            ),
            "affinity_probability_binary": float(
                affinity_data.get("affinity_probability_binary", 0.5)
            ),
            "ligand_iptm": float(confidence_data.get("ligand_iptm", 0.0)),
            "complex_plddt": float(confidence_data.get("complex_plddt", 0.0)),
            "confidence_score": float(confidence_data.get("confidence_score", 0.5)),
            "structure_path": structure_path,
        }

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[Dict]:
        path = self._cache_path(key)
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
        return None

    def _save_cache(self, key: str, result: Dict) -> None:
        with open(self._cache_path(key), "w") as fh:
            json.dump(result, fh, indent=2)


# ---------------------------------------------------------------------------
# BoltzEnhancedDataset
# ---------------------------------------------------------------------------


class BoltzEnhancedDataset(Dataset):
    """
    Augments the standard protein-molecule TSV dataset with pre-cached
    Boltz-2 structural features.

    TSV format (tab-separated, with header row)::

        index  protein_seq  smiles  label  affinity_value

    The ``boltz_cache_dir`` must contain JSON files named ``{md5_key}.json``
    as written by ``BoltzP2MPredictor``. Missing entries silently fall back
    to a zero-vector ``[0.0, 0.5, 0.0, 0.0]``.

    Boltz feature vector (dim 4, in order):
        [0] ``affinity_pred_value``
        [1] ``affinity_probability_binary``
        [2] ``ligand_iptm``
        [3] ``complex_plddt``

    Each ``__getitem__`` returns a dict with keys:
        ``protein_seq``, ``smiles``, ``label``, ``affinity``,
        ``boltz_features``
    """

    BOLTZ_FEATURE_DIM: int = 4
    _FALLBACK_FEATURES: List[float] = [0.0, 0.5, 0.0, 0.0]

    def __init__(
        self,
        data_file: Path,
        boltz_cache_dir: str = "data/boltz_cache",
        max_protein_len: int = 1000,
        max_smiles_len: int = 150,
        cache_only: bool = False,
    ) -> None:
        self.boltz_cache_dir = Path(boltz_cache_dir)
        self.max_protein_len = max_protein_len
        self.max_smiles_len = max_smiles_len
        self.cache_only = cache_only

        # Load TSV samples — detect column positions from header so the
        # dataset works with both the old schema (no interaction_type column)
        # and the new schema (interaction_type between affinity and source).
        self.samples: List[Tuple[str, str, float, float, int]] = []
        with open(data_file, "r") as fh:
            raw_header = fh.readline().strip().split("\t")
            cols = {c.lower(): i for i, c in enumerate(raw_header)}

            idx_protein = cols.get("protein_seq", 1)
            idx_smiles  = cols.get("smiles", 2)
            idx_label   = cols.get("label", 3)
            idx_affinity = cols.get("affinity", 4)
            idx_itype   = cols.get("interaction_type", None)

            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) <= max(idx_protein, idx_smiles, idx_label):
                    continue
                try:
                    protein_seq    = parts[idx_protein]
                    smiles         = parts[idx_smiles]
                    label          = float(parts[idx_label])
                    affinity_value = float(parts[idx_affinity]) if idx_affinity < len(parts) else 0.0
                    if idx_itype is not None and idx_itype < len(parts):
                        try:
                            interaction_type = int(parts[idx_itype])
                        except ValueError:
                            interaction_type = 3
                    else:
                        interaction_type = 3  # fallback: OTHER (column absent in old schema)
                except (ValueError, IndexError):
                    continue
                self.samples.append((protein_seq, smiles, label, affinity_value, interaction_type))

        # Report coverage statistics at construction time
        total = len(self.samples)
        cached_set = {
            (p, s)
            for p, s, *_ in self.samples
            if (self.boltz_cache_dir / f"{_boltz_key(p, s)}.json").exists()
        }
        cached = len(cached_set)
        pct = 100.0 * cached / max(total, 1)
        print(
            f"[BoltzEnhancedDataset] {total:,} samples loaded | "
            f"Boltz-2 cache coverage: {cached:,}/{total:,} ({pct:.1f}%)"
        )

        # In cache_only mode (Phase 2), drop samples without Boltz features
        if cache_only:
            self.samples = [
                s for s in self.samples
                if (s[0], s[1]) in cached_set
            ]
            print(
                f"[BoltzEnhancedDataset] cache_only=True → "
                f"kept {len(self.samples):,} samples with Boltz features"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        protein_seq, smiles, label, affinity_value, interaction_type = self.samples[idx]

        # Load Boltz-2 features or fall back to safe neutral defaults
        key = _boltz_key(protein_seq, smiles)
        cache_path = self.boltz_cache_dir / f"{key}.json"
        if cache_path.exists():
            with open(cache_path) as fh:
                bd = json.load(fh)
            boltz_feats: List[float] = [
                float(bd.get("affinity_pred_value", 0.0)),
                float(bd.get("affinity_probability_binary", 0.5)),
                float(bd.get("ligand_iptm", 0.0)),
                float(bd.get("complex_plddt", 0.0)),
            ]
        else:
            boltz_feats = list(self._FALLBACK_FEATURES)

        return {
            "protein_seq": protein_seq[: self.max_protein_len],
            "smiles": encode_smiles(smiles, self.max_smiles_len),
            "label": torch.tensor(label, dtype=torch.float32),
            "affinity": torch.tensor(affinity_value, dtype=torch.float32),
            "boltz_features": torch.tensor(boltz_feats, dtype=torch.float32),
            "interaction_type": torch.tensor(interaction_type, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# HybridP2MModel
# ---------------------------------------------------------------------------


class HybridP2MModel(nn.Module):
    """
    Fuses a pre-trained ``ProteinMoleculeModel`` with Boltz-2 structural
    features via a small MLP encoder and learned prediction heads.

    Architecture::

        base_model(protein_seqs, smiles)
            → base_logit  [B]          (interaction_logits)
            → base_affin  [B]          (affinity)

        boltz_encoder(boltz_features [B, 4])
            → boltz_enc   [B, H//4]

        combined = cat([boltz_enc, base_logit[:, None], base_affin[:, None]])
            → [B, H//4 + 2]

        interaction_head(combined) → final_logit  [B]
        affinity_head(combined)    → final_affin  [B]

    Parameters
    ----------
    base_model : ProteinMoleculeModel
        A (possibly pre-trained) base model. Weights can optionally be frozen.
    hidden_dim : int
        Must match the ``hidden_dim`` of ``base_model`` (default 512).
    dropout : float
        Dropout probability for the Boltz encoder and affinity head.
    freeze_base : bool
        If ``True``, gradients are disabled for all base model parameters.
    """

    def __init__(
        self,
        base_model: "ProteinMoleculeModel",
        hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        boltz_proj_dim: int = hidden_dim // 4  # 128 when hidden_dim=512

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Boltz feature encoder: 4 → boltz_proj_dim
        self.boltz_encoder = nn.Sequential(
            nn.Linear(BoltzEnhancedDataset.BOLTZ_FEATURE_DIM, boltz_proj_dim),
            nn.LayerNorm(boltz_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(boltz_proj_dim, boltz_proj_dim),
        )

        # Combined dim: boltz_proj_dim + 2 scalars (base_logit + base_affinity)
        combined_dim: int = boltz_proj_dim + 2

        # New binary interaction head
        self.interaction_head = nn.Linear(combined_dim, 1)

        # New affinity regression head
        self.affinity_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        protein_sequences: List[str],
        molecule_sequences: torch.Tensor,
        boltz_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that combines base model outputs with Boltz-2 features.

        Parameters
        ----------
        protein_sequences : List[str]
            Batch of amino acid sequences (length B).
        molecule_sequences : Tensor [B, mol_len]
            Encoded SMILES tensors.
        boltz_features : Tensor [B, 4]
            Pre-cached Boltz-2 feature vectors.
        return_attention : bool
            Forwarded to the base model; adds attention tensors to output.

        Returns
        -------
        dict with keys:
            ``interaction_logits`` : Tensor [B]
            ``affinity``           : Tensor [B]
            ``interaction_type``   : Tensor [B, 4]  (passed through from base)
            ``protein_attention``  : Tensor (optional)
            ``molecule_attention`` : Tensor (optional)
        """
        # 1. Run base model to obtain scalar predictions
        base_out = self.base_model(
            protein_sequences,
            molecule_sequences,
            return_attention=return_attention,
        )
        base_logit: torch.Tensor = base_out["interaction_logits"]   # [B]
        base_affin: torch.Tensor = base_out["affinity"]             # [B]

        # 2. Encode Boltz-2 structural features
        boltz_enc = self.boltz_encoder(boltz_features)              # [B, H//4]

        # 3. Concatenate Boltz encoding with base scalar predictions
        combined = torch.cat(
            [
                boltz_enc,
                base_logit.unsqueeze(-1),   # [B, 1]
                base_affin.unsqueeze(-1),   # [B, 1]
            ],
            dim=-1,
        )  # [B, H//4 + 2]

        # 4. New prediction heads
        output: Dict[str, torch.Tensor] = {
            "interaction_logits": self.interaction_head(combined).squeeze(-1),
            "affinity": self.affinity_head(combined).squeeze(-1),
            "interaction_type": base_out["interaction_type"],
        }

        # Pass through optional attention tensors from base model
        if return_attention:
            for attn_key in ("protein_attention", "molecule_attention"):
                if attn_key in base_out:
                    output[attn_key] = base_out[attn_key]

        return output


# ---------------------------------------------------------------------------
# DataLoader collation
# ---------------------------------------------------------------------------


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of ``BoltzEnhancedDataset`` samples into a batch dict."""
    return {
        "protein_seq": [b["protein_seq"] for b in batch],
        "smiles": torch.stack([b["smiles"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "affinity": torch.stack([b["affinity"] for b in batch]),
        "boltz_features": torch.stack([b["boltz_features"] for b in batch]),
        "interaction_type": torch.stack([b["interaction_type"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_hybrid_epoch(
    model: HybridP2MModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run one training epoch for the ``HybridP2MModel``.

    Loss = BCE(interaction_logits, labels)
           + 0.1 * MSE(predicted_affinity, true_affinity)  [positive pairs only]

    Returns
    -------
    dict with keys ``loss`` and ``accuracy``.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        labels = batch["label"].to(device)
        smiles = batch["smiles"].to(device)
        affinity = batch["affinity"].to(device)
        boltz_feats = batch["boltz_features"].to(device)

        optimizer.zero_grad()

        output = model(batch["protein_seq"], smiles, boltz_feats)

        # Primary objective: binary cross-entropy for interaction prediction
        bce_loss = F.binary_cross_entropy_with_logits(
            output["interaction_logits"], labels
        )

        # Auxiliary objective: affinity regression on positive samples
        mask = labels > 0.5
        affinity_loss = torch.tensor(0.0, device=device)
        if mask.sum() > 0:
            affinity_loss = F.mse_loss(
                output["affinity"][mask], affinity[mask]
            )

        loss = bce_loss + 0.1 * affinity_loss

        # Interaction type CE loss (positive samples only)
        pos_mask = labels > 0.5
        if pos_mask.sum() > 0:
            itype_true = batch["interaction_type"].to(device)[pos_mask]
            itype_logits = output["interaction_type"][pos_mask]
            itype_loss = F.cross_entropy(itype_logits, itype_true)
            loss = loss + 0.3 * itype_loss

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(output["interaction_logits"]) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Periodically flush the MPS allocator to prevent fragmentation buildup
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            if batch_idx % 100 == 0:
                torch.mps.empty_cache()

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
    }


def evaluate_hybrid(
    model: HybridP2MModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the hybrid model on a dataloader.

    Returns
    -------
    dict with keys: ``loss``, ``accuracy``, ``auc``, ``f1``,
    ``affinity_rmse``.
    """
    model.eval()
    all_labels: List[float] = []
    all_probs: List[float] = []
    all_affinities_true: List[float] = []
    all_affinities_pred: List[float] = []
    total_loss = 0.0
    correct = 0
    total = 0
    itype_correct = 0
    itype_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch["label"].to(device)
            smiles = batch["smiles"].to(device)
            affinity = batch["affinity"].to(device)
            boltz_feats = batch["boltz_features"].to(device)

            output = model(batch["protein_seq"], smiles, boltz_feats)

            loss = F.binary_cross_entropy_with_logits(
                output["interaction_logits"], labels
            )
            total_loss += loss.item()

            probs = torch.sigmoid(output["interaction_logits"])
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            # Collect affinity data for positive samples only
            mask = labels > 0.5
            if mask.sum() > 0:
                all_affinities_true.extend(affinity[mask].cpu().tolist())
                all_affinities_pred.extend(
                    output["affinity"][mask].cpu().tolist()
                )

            # Interaction type accuracy (positive samples only)
            pos_mask = labels > 0.5
            if pos_mask.sum() > 0:
                itype_true = batch["interaction_type"].to(device)[pos_mask]
                itype_pred = output["interaction_type"][pos_mask].argmax(dim=-1)
                itype_correct += (itype_pred == itype_true).sum().item()
                itype_total += pos_mask.sum().item()

    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)

    # AUC — gracefully handle single-class batches
    auc = (
        float(roc_auc_score(labels_arr, probs_arr))
        if len(np.unique(labels_arr)) > 1
        else 0.5
    )

    preds_arr = (probs_arr > 0.5).astype(float)
    _, _, f1, _ = precision_recall_fscore_support(
        labels_arr, preds_arr, average="binary", zero_division=0
    )

    affinity_rmse = (
        float(
            np.sqrt(mean_squared_error(all_affinities_true, all_affinities_pred))
        )
        if all_affinities_true
        else 0.0
    )

    return {
        "loss": total_loss / max(len(dataloader), 1),
        "accuracy": correct / max(total, 1),
        "auc": auc,
        "f1": float(f1),
        "affinity_rmse": affinity_rmse,
        "interaction_type_accuracy": itype_correct / max(itype_total, 1),
    }


def _save_hybrid_checkpoint(
    path: Path,
    model: "HybridP2MModel",
    optimizer,
    scheduler,
    epoch: int,
    best_auc: float,
    patience_counter: int,
    history: Dict,
) -> None:
    """Save a full resumable hybrid checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
            "patience_counter": patience_counter,
            "hidden_dim": model.hidden_dim,
            "history": history,
        },
        path,
    )


def train_hybrid_model(
    model: HybridP2MModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: Path,
    patience: int = 10,
    resume_path: Path = None,
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping for the ``HybridP2MModel``.

    Uses AdamW with a ReduceLROnPlateau scheduler (monitors validation AUC).
    Two checkpoints are written:
      - ``save_path``         — best model by validation AUC
      - ``save_path.latest``  — most recent epoch (always, for reliable resume)

    Parameters
    ----------
    model : HybridP2MModel
    train_loader, val_loader : DataLoader
    epochs : int
    lr : float
    device : torch.device
    save_path : Path
        Where to write the best model checkpoint (.pt).
    patience : int
        Number of epochs without improvement before early stopping.
    resume_path : Path, optional
        Checkpoint to resume from (restores model, optimizer, scheduler,
        history, and epoch counter).

    Returns
    -------
    Training history dict with lists: ``train_loss``, ``val_loss``,
    ``val_auc``, ``val_f1``, ``val_affinity_rmse``.
    """
    # Only optimise parameters that require gradients (respects freeze_base)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc = 0.0
    patience_counter = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_f1": [],
        "val_affinity_rmse": [],
    }
    start_epoch = 0

    # Restore from checkpoint if requested
    if resume_path and Path(resume_path).exists():
        print(f"\n  Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_auc        = ckpt.get("best_auc", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        history         = ckpt.get("history", history)
        start_epoch     = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch + 1} | best AUC so far: {best_auc:.4f}")
    elif resume_path:
        print(f"  ⚠  Resume checkpoint not found at {resume_path} — starting fresh")

    latest_path = Path(save_path).with_suffix(".latest.pt")

    print("\n" + "=" * 70)
    print("HYBRID P2M MODEL — TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_m = train_hybrid_epoch(model, train_loader, optimizer, device)
        val_m = evaluate_hybrid(model, val_loader, device)

        scheduler.step(val_m["auc"])

        print(
            f"  Train → Loss: {train_m['loss']:.4f} | "
            f"Acc: {train_m['accuracy']:.4f}"
        )
        print(
            f"  Val   → Loss: {val_m['loss']:.4f} | "
            f"Acc: {val_m['accuracy']:.4f} | "
            f"AUC: {val_m['auc']:.4f} | "
            f"F1: {val_m['f1']:.4f}"
        )
        if val_m["affinity_rmse"] > 0:
            print(f"  Affinity RMSE: {val_m['affinity_rmse']:.4f}")

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_auc"].append(val_m["auc"])
        history["val_f1"].append(val_m["f1"])
        history["val_affinity_rmse"].append(val_m["affinity_rmse"])

        # Always save latest checkpoint so training can be resumed at any point
        _save_hybrid_checkpoint(
            latest_path, model, optimizer, scheduler,
            epoch, best_auc, patience_counter, history,
        )

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            patience_counter = 0
            _save_hybrid_checkpoint(
                Path(save_path), model, optimizer, scheduler,
                epoch, best_auc, patience_counter, history,
            )
            print(f"  ✓ Saved best model → {save_path} (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

    print("\n" + "=" * 70)
    print(f"Training complete.  Best validation AUC: {best_auc:.4f}")
    print("=" * 70)
    return history


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boltz-2 Hybrid Protein–Molecule Interaction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Shared arguments ---
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing protein_molecule_interactions.tsv (training mode)",
    )
    parser.add_argument(
        "--boltz-cache",
        type=str,
        default="data/boltz_cache",
        help="Directory for Boltz-2 result JSON / .cif cache",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="boltz_hybrid_p2m.pt",
        help="Path to save/load the hybrid model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a specific device (cpu / cuda / mps). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="Path to a pre-trained ProteinMoleculeModel .pt to use as the base model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a hybrid checkpoint to resume from. "
             "Use 'latest' to auto-load {checkpoint}.latest.pt.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only train on samples that have pre-computed Boltz-2 features "
             "(skips all zero-feature pairs). Off by default.",
    )
    parser.add_argument(
        "--esm-cache",
        type=str,
        default=None,
        help="Path to pre-computed ESM-2 embedding cache (.pt) from precompute_esm.py. "
             "Skips loading ESM-2 entirely and removes the main training bottleneck.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience (epochs without AUC improvement)",
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze base model weights and only train the fusion layers",
    )

    # --- Direct inference mode ---
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run Boltz-2 directly on a single protein–ligand pair and exit",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Amino acid sequence for --predict mode",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default=None,
        help="Ligand SMILES string for --predict mode",
    )
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help="Pass --use_msa_server to the Boltz-2 CLI (requires internet)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[boltz_p2m] Device: {device}")

    # ==================================================================
    # Mode 1: Direct Boltz-2 inference
    # ==================================================================
    if args.predict:
        if not args.protein or not args.smiles:
            parser.error("--predict requires both --protein and --smiles")

        predictor = BoltzP2MPredictor(
            cache_dir=args.boltz_cache,
            use_msa_server=args.use_msa_server,
        )

        truncated = (
            args.protein[:60] + "..." if len(args.protein) > 60 else args.protein
        )
        print(f"\nRunning Boltz-2 prediction …")
        print(f"  Protein : {truncated}")
        print(f"  SMILES  : {args.smiles}")

        result = predictor.predict(args.protein, args.smiles)

        print("\n--- Boltz-2 Results ---")
        for key, val in result.items():
            print(f"  {key}: {val}")

        tensor_out = predictor.to_model_output(result)
        prob = torch.sigmoid(tensor_out["interaction_logits"]).item()
        print(f"\n  Interaction probability : {prob:.4f}")
        print(f"  Predicted affinity      : {tensor_out['affinity'].item():.4f}")
        return

    # ==================================================================
    # Mode 2: Hybrid model training
    # ==================================================================
    if not args.data_dir:
        parser.error("--data-dir is required for training mode (or use --predict)")

    data_dir   = Path(args.data_dir)
    train_file = data_dir / "p2m_train.tsv"
    val_file   = data_dir / "p2m_val.tsv"

    if not train_file.exists():
        print(f"❌  Training file not found: {train_file}")
        print("    Run:  python -m prepare.prepare_all --data-dir data --output-dir data/prepared")
        sys.exit(1)
    if not val_file.exists():
        print(f"❌  Validation file not found: {val_file}")
        print("    Run:  python -m prepare.prepare_all --data-dir data --output-dir data/prepared")
        sys.exit(1)

    # Build datasets from the pre-split files produced by prepare_p2m_data.py.
    # Always train on all samples — zero vectors are used as fallback when
    # Boltz-2 features are absent. Use --cache-only to restrict to cached pairs.
    train_dataset = BoltzEnhancedDataset(
        data_file=train_file,
        boltz_cache_dir=args.boltz_cache,
        cache_only=args.cache_only,
    )
    val_dataset = BoltzEnhancedDataset(
        data_file=val_file,
        boltz_cache_dir=args.boltz_cache,
        cache_only=args.cache_only,
    )

    if len(train_dataset) == 0:
        print("❌  Training dataset is empty. Aborting.")
        sys.exit(1)

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Build base ProteinMoleculeModel
    base_model = ProteinMoleculeModel(
        protein_dim=512,
        molecule_dim=512,
        hidden_dim=512,
        device=device,
        esm_cache_path=args.esm_cache,
    ).to(device)

    if args.base_checkpoint:
        base_ckpt_path = Path(args.base_checkpoint)
        if base_ckpt_path.exists():
            ckpt = torch.load(base_ckpt_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = base_model.load_state_dict(state, strict=False)
            print(
                f"✓  Loaded base weights from {base_ckpt_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            print(
                f"⚠  Base checkpoint not found: {base_ckpt_path}. "
                "Using random initialisation."
            )

    # Build hybrid model
    model = HybridP2MModel(
        base_model=base_model,
        hidden_dim=512,
        freeze_base=args.freeze_base,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"HybridP2MModel: {total_params:,} total params | "
        f"{trainable_params:,} trainable"
    )

    # Run training
    save_path = Path(args.checkpoint)
    # Resolve --resume path
    resume_path = None
    if args.resume:
        save_path = Path(args.checkpoint)
        if args.resume == "latest":
            resume_path = save_path.with_suffix(".latest.pt")
        else:
            resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"  ⚠  Resume file not found: {resume_path}")

    history = train_hybrid_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=Path(args.checkpoint),
        patience=args.patience,
        resume_path=resume_path,
    )

    # Persist training history alongside the checkpoint
    hist_path = save_path.with_suffix(".history.json")
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\nTraining history saved to {hist_path}")
    print("Done!")


if __name__ == "__main__":
    main()
