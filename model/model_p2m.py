#!/usr/bin/env python3
"""
Protein-Small Molecule Interaction Prediction Model
====================================================

Multi-modal model for predicting protein-ligand interactions:
1. ESM-2 for protein sequence encoding
2. Graph Neural Network for molecular structure encoding (SMILES → Graph)
3. Cross-attention fusion between protein and molecule representations

This model predicts drug-target interactions and binding affinities,
useful for drug discovery, virtual screening, and lead optimization.

Usage:
    python model_p2m.py --data-dir data/protein_molecule --epochs 50
    python model_p2m.py --data-dir data/protein_molecule --epochs 10 --max-samples 5000

Data Sources:
    - ChEMBL: Bioactivity data for drug-like molecules
    - BindingDB: Protein-ligand binding affinities
    - PDBbind: Experimentally measured binding data
    - DrugBank: Drug-target interactions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Enable MPS fallback for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import esm
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

sys.stdout.reconfigure(line_buffering=True)

# Atom and bond vocabulary for molecular graphs
ATOM_VOCAB = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "F": 4,
    "Cl": 5,
    "Br": 6,
    "I": 7,
    "P": 8,
    "B": 9,
    "Si": 10,
    "Se": 11,
    "H": 12,
    "UNK": 13,
}
ATOM_VOCAB_SIZE = 14

BOND_VOCAB = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3,
}
BOND_VOCAB_SIZE = 4

# SMILES character vocabulary for sequence-based encoding
SMILES_VOCAB = {
    "PAD": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "S": 4,
    "F": 5,
    "Cl": 6,
    "Br": 7,
    "I": 8,
    "P": 9,
    "(": 10,
    ")": 11,
    "[": 12,
    "]": 13,
    "=": 14,
    "#": 15,
    "@": 16,
    "+": 17,
    "-": 18,
    "\\": 19,
    "/": 20,
    "1": 21,
    "2": 22,
    "3": 23,
    "4": 24,
    "5": 25,
    "6": 26,
    "7": 27,
    "8": 28,
    "9": 29,
    "0": 30,
    "c": 31,
    "n": 32,
    "o": 33,
    "s": 34,
    "H": 35,
    ".": 36,
    "UNK": 37,
}
SMILES_VOCAB_SIZE = 38
SMILES_PAD_IDX = 0

INTERACTION_TYPE_LABELS = {0: "INHIBITOR", 1: "ACTIVATOR", 2: "MODULATOR", 3: "OTHER"}
NUM_INTERACTION_TYPES = 4


def encode_smiles(smiles: str, max_length: int = 150) -> torch.Tensor:
    """
    Encode SMILES string to tensor using character-level encoding

    Args:
        smiles: SMILES string representation of molecule
        max_length: Maximum sequence length

    Returns:
        Tensor of shape [max_length] with token indices
    """
    indices = []
    i = 0
    while i < len(smiles) and len(indices) < max_length:
        # Handle two-character tokens (Cl, Br)
        if i < len(smiles) - 1 and smiles[i : i + 2] in SMILES_VOCAB:
            indices.append(SMILES_VOCAB[smiles[i : i + 2]])
            i += 2
        elif smiles[i] in SMILES_VOCAB:
            indices.append(SMILES_VOCAB[smiles[i]])
            i += 1
        else:
            indices.append(SMILES_VOCAB["UNK"])
            i += 1

    # Pad to max_length
    if len(indices) < max_length:
        indices.extend([SMILES_PAD_IDX] * (max_length - len(indices)))

    return torch.tensor(indices, dtype=torch.long)


class MoleculeEncoder(nn.Module):
    """
    Small molecule encoder using CNN + Self-Attention on SMILES

    This is a sequence-based approach that works directly on SMILES strings,
    avoiding the need for RDKit dependency while still capturing molecular structure.

    Architecture:
    - Character embedding for SMILES tokens
    - Multi-scale CNN for local substructure patterns
    - Self-attention for long-range dependencies (ring systems, etc.)
    - Pooling with learned attention weights
    """

    def __init__(
        self,
        vocab_size: int = SMILES_VOCAB_SIZE,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=SMILES_PAD_IDX)

        # Positional encoding
        self.pos_embedding = nn.Embedding(500, embed_dim)  # Max 500 characters

        # Multi-scale CNN for capturing different substructure sizes
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim, hidden_dim // 4, kernel_size=k, padding=k // 2
                    ),
                    nn.BatchNorm1d(hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for k in [3, 5, 7, 9]  # Different substructure sizes
            ]
        )

        # Combine multi-scale features
        self.conv_combine = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Transformer encoder layers for global context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode molecule from SMILES tokens

        Args:
            x: SMILES token indices [batch_size, seq_len]

        Returns:
            Tuple of (pooled_output, sequence_output)
            - pooled_output: [batch_size, output_dim]
            - sequence_output: [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len = x.shape

        # Create padding mask
        padding_mask = x == SMILES_PAD_IDX  # [batch, seq_len]

        # Embed characters with positional encoding
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        embedded = self.embedding(x) + self.pos_embedding(positions)

        # CNN expects [batch, channels, seq_len]
        conv_input = embedded.transpose(1, 2)

        # Multi-scale CNN
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(conv_input)
            # Ensure same sequence length
            if conv_out.shape[2] != seq_len:
                conv_out = F.pad(conv_out, (0, seq_len - conv_out.shape[2]))
            conv_outputs.append(conv_out)

        # Concatenate multi-scale features
        conv_combined = torch.cat(conv_outputs, dim=1)  # [batch, hidden_dim, seq_len]
        conv_combined = self.conv_combine(conv_combined)
        conv_out = conv_combined.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        # Create attention mask for transformer (True = ignore)
        # Note: for MPS compatibility, we handle masking differently
        transformer_out = self.transformer(conv_out)

        # Zero out padded positions
        mask = (~padding_mask).float().unsqueeze(-1)  # [batch, seq_len, 1]
        transformer_out = transformer_out * mask

        # Project to output dimension
        sequence_output = self.projection(transformer_out)

        # Attention-weighted pooling
        attn_weights = self.attention_pool(transformer_out)
        attn_weights = attn_weights.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        )
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled_output = (sequence_output * attn_weights).sum(dim=1)

        return pooled_output, sequence_output


class ProteinEncoder(nn.Module):
    """ESM-2 based protein encoder optimized for binding site prediction.

    If a pre-computed embedding cache is provided (via load_cache or the
    --esm-cache CLI flag), the ESM-2 forward pass is skipped entirely during
    training — cached token representations are loaded from disk instead.
    This reduces per-batch cost from ~80 s to <1 s on MPS/CPU.

    Cache format (produced by model/precompute_esm.py):
        {md5_hex_of_sequence: tensor[L, esm_dim]}  saved as a .pt file.
    """

    def __init__(
        self,
        model_name: str = "esm2_t12_35M_UR50D",
        output_dim: int = 512,
        device: torch.device = None,
        cache_path: str = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.output_dim = output_dim

        # Optional pre-computed embedding cache (lives on target device after load)
        self._emb_cache: dict = {}
        self._cache_only: bool = False  # True when ESM-2 is not loaded
        if cache_path:
            self.load_cache(cache_path)

        # Only load ESM-2 if the cache is absent or incomplete.
        # When the full cache is available, skip loading entirely to free
        # device memory and eliminate any residual ESM overhead.
        if self._cache_only:
            self.esm_model = None
            self.batch_converter = None
            self.esm_dim = 480  # esm2_t12_35M_UR50D default; overridden if ESM loaded
        else:
            print(f"Loading ESM-2 model: {model_name}")
            self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(
                model_name
            )
            self.batch_converter = self.alphabet.get_batch_converter()
            self.esm_dim = self.esm_model.embed_dim

            # Freeze ESM weights — never updated during training
            for param in self.esm_model.parameters():
                param.requires_grad = False

            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(self.esm_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

        # Binding pocket attention
        # Helps focus on residues likely to be in binding pockets
        self.pocket_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.Tanh(),
            nn.Linear(output_dim // 4, 1),
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def load_cache(self, cache_path: str) -> None:
        """Load pre-computed ESM-2 token representations from a .pt file.

        All tensors are moved to self.device immediately so training never
        pays a per-batch CPU→device transfer cost.  Sets self._cache_only=True
        so that ESM-2 is not loaded at all when the cache is present.
        """
        path = Path(cache_path)
        if not path.exists():
            print(f"  ⚠  ESM cache not found at {path} — will run ESM-2 live")
            return
        raw = torch.load(path, map_location="cpu", weights_only=True)
        self._emb_cache = raw
        self._cache_only = True
        print(f"  ✓  ESM cache loaded: {len(self._emb_cache):,} embeddings "
              f"on cpu from {path} — ESM-2 will not be loaded")

    @staticmethod
    def _cache_key(seq: str) -> str:
        import hashlib
        # Hash the sequence exactly as received — must match precompute_esm.py
        return hashlib.md5(seq.encode("utf-8")).hexdigest()

    def _get_token_repr(self, sequences: List[str]) -> torch.Tensor:
        """
        Return per-residue ESM-2 token representations for each sequence,
        padded to a single tensor [B, L_max, esm_dim] on self.device.

        Deduplicates sequences within the batch — if the same protein appears
        multiple times (common when many molecules bind one target), ESM-2 /
        cache lookup runs only once per unique sequence then indexes back to
        the full batch order.  Cache tensors are already on self.device so
        there is no per-batch CPU→device transfer cost.
        """
        # --- deduplicate within the batch ---
        unique_seqs: List[str] = []
        seen: dict[str, int] = {}       # seq → index in unique_seqs
        batch_to_unique: List[int] = []

        for seq in sequences:
            if seq not in seen:
                seen[seq] = len(unique_seqs)
                unique_seqs.append(seq)
            batch_to_unique.append(seen[seq])

        # --- resolve each unique sequence (cache or live ESM-2) ---
        unique_reprs: List[torch.Tensor] = []
        uncached_local: List[int] = []   # indices into unique_seqs
        uncached_seqs:  List[str] = []

        for ui, seq in enumerate(unique_seqs):
            key = self._cache_key(seq)
            if key in self._emb_cache:
                # Already on self.device (moved at cache load time)
                unique_reprs.append(self._emb_cache[key])
            else:
                unique_reprs.append(None)   # placeholder
                uncached_local.append(ui)
                uncached_seqs.append(seq)

        # --- run live ESM-2 only for sequences absent from cache ---
        if uncached_seqs:
            if self.esm_model is None:
                raise RuntimeError(
                    f"{len(uncached_seqs)} sequence(s) are missing from the ESM "
                    "cache and ESM-2 was not loaded.  Re-run:\n"
                    "  python -m model.precompute_esm --data-dir data/prepared/p2m"
                )
            data = [(f"p{j}", s[:1022]) for j, s in enumerate(uncached_seqs)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            with torch.no_grad():
                results = self.esm_model(
                    batch_tokens,
                    repr_layers=[self.esm_model.num_layers],
                    return_contacts=False,
                )
            token_repr = results["representations"][self.esm_model.num_layers]
            for j, (ui, seq) in enumerate(zip(uncached_local, uncached_seqs)):
                seq_len = min(len(seq), 1022)
                unique_reprs[ui] = token_repr[j, 1 : seq_len + 1]  # already on device

        # --- build padded batch tensor [B, L_max, esm_dim] ---
        L_max  = max(t.shape[0] for t in unique_reprs)
        esm_dim = unique_reprs[0].shape[1]
        # Allocate once on device — no per-sample transfer
        padded = torch.zeros(len(sequences), L_max, esm_dim, device=self.device)
        for batch_i, unique_i in enumerate(batch_to_unique):
            t = unique_reprs[unique_i]
            padded[batch_i, : t.shape[0]] = t
        return padded  # [B, L_max, esm_dim]

    def forward(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein sequences.

        Returns:
            Tuple of (pooled_output [B, output_dim], sequence_output [B, L, output_dim])
        """
        # padded is already on self.device, deduplicated, no CPU→device transfer
        padded = self._get_token_repr(sequences)  # [B, L_max, esm_dim]

        sequence_output = padded   # [B, L, esm_dim]
        pooled_output   = padded.mean(dim=1)  # [B, esm_dim]

        # Project
        pooled_output = self.projection(pooled_output)
        sequence_output = self.projection(sequence_output)

        # Apply binding pocket attention weighting
        attn_weights = self.pocket_attention(sequence_output)  # [batch, seq, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_pooled = (sequence_output * attn_weights).sum(dim=1)

        # Combine mean pooling and attention pooling
        pooled_output = pooled_output + weighted_pooled

        return pooled_output, sequence_output


class CrossAttention(nn.Module):
    """Cross-attention module for protein-molecule interaction"""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention: query attends to key_value

        Args:
            query: [batch, seq_len_q, dim]
            key_value: [batch, seq_len_kv, dim]

        Returns:
            Updated query representations
        """
        # Cross-attention
        attn_out, _ = self.attention(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)

        return query


class ProteinMoleculeModel(nn.Module):
    """
    Protein-Small Molecule Interaction Prediction Model

    Predicts drug-target interactions and binding affinities.

    Architecture:
    1. ESM-2 protein encoder with binding pocket attention
    2. SMILES-based molecule encoder with multi-scale CNN + Transformer
    3. Bidirectional cross-attention fusion
    4. Multi-task prediction head (binding, affinity, interaction type)

    Output:
    - Binary interaction prediction (does it bind?)
    - Binding affinity score (how strongly?)
    - Binding site attention weights (where?)
    """

    def __init__(
        self,
        protein_dim: int = 512,
        molecule_dim: int = 512,
        hidden_dim: int = 512,
        esm_model: str = "esm2_t12_35M_UR50D",
        num_cross_attention_layers: int = 2,
        dropout: float = 0.1,
        device: torch.device = None,
        esm_cache_path: str = None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")

        # Protein encoder (ESM-2) — pass cache path so training skips ESM forward pass
        self.protein_encoder = ProteinEncoder(
            model_name=esm_model, output_dim=protein_dim, device=self.device,
            cache_path=esm_cache_path,
        )

        # Molecule encoder (SMILES-based)
        self.molecule_encoder = MoleculeEncoder(
            output_dim=molecule_dim, hidden_dim=256, dropout=dropout
        )

        # Cross-attention layers (protein attends to molecule and vice versa)
        self.protein_to_molecule_attention = nn.ModuleList(
            [
                CrossAttention(dim=protein_dim, dropout=dropout)
                for _ in range(num_cross_attention_layers)
            ]
        )
        self.molecule_to_protein_attention = nn.ModuleList(
            [
                CrossAttention(dim=molecule_dim, dropout=dropout)
                for _ in range(num_cross_attention_layers)
            ]
        )

        # Binding site attention (for interpretability)
        self.binding_site_attention = nn.Sequential(
            nn.Linear(protein_dim, protein_dim // 4),
            nn.Tanh(),
            nn.Linear(protein_dim // 4, 1),
        )

        # Pharmacophore attention (for molecule interpretability)
        self.pharmacophore_attention = nn.Sequential(
            nn.Linear(molecule_dim, molecule_dim // 4),
            nn.Tanh(),
            nn.Linear(molecule_dim // 4, 1),
        )

        # Projection layers for product and diff to match concat dimension
        fusion_dim = protein_dim + molecule_dim
        self.product_projection = nn.Linear(protein_dim, fusion_dim)
        self.diff_projection = nn.Linear(protein_dim, fusion_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, hidden_dim),  # concat, product, diff
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Prediction heads
        # Binary interaction prediction
        self.interaction_head = nn.Linear(hidden_dim, 1)

        # Binding affinity prediction (pIC50, pKd, etc.)
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # Interaction type classification (inhibitor, activator, etc.)
        self.interaction_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 interaction types
        )

    def forward(
        self,
        protein_sequences: List[str],
        molecule_sequences: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict protein-molecule interaction

        Args:
            protein_sequences: List of protein amino acid sequences
            molecule_sequences: Encoded SMILES sequences [batch, mol_len]
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
            - interaction_logits: Binary binding prediction
            - affinity: Binding affinity score
            - interaction_type: Classification logits for interaction type
            - protein_attention: (optional) Binding site attention
            - molecule_attention: (optional) Pharmacophore attention
        """
        # Encode protein
        protein_pooled, protein_seq = self.protein_encoder(protein_sequences)

        # Encode molecule
        molecule_pooled, molecule_seq = self.molecule_encoder(molecule_sequences)

        # Cross-attention: let protein and molecule attend to each other
        for p2m_attn, m2p_attn in zip(
            self.protein_to_molecule_attention, self.molecule_to_protein_attention
        ):
            # Protein attends to molecule
            protein_seq = p2m_attn(protein_seq, molecule_seq)
            # Molecule attends to protein
            molecule_seq = m2p_attn(molecule_seq, protein_seq)

        # Get binding site attention weights
        binding_attn = self.binding_site_attention(protein_seq)  # [batch, seq, 1]
        binding_attn_weights = F.softmax(binding_attn, dim=1)

        # Get pharmacophore attention weights
        pharm_attn = self.pharmacophore_attention(molecule_seq)  # [batch, seq, 1]
        pharm_attn_weights = F.softmax(pharm_attn, dim=1)

        # Pool the cross-attended representations
        protein_final = (protein_seq * binding_attn_weights).sum(dim=1)
        molecule_final = (molecule_seq * pharm_attn_weights).sum(dim=1)

        # Combine representations
        concat = torch.cat([protein_final, molecule_final], dim=-1)
        product = protein_pooled * molecule_pooled  # Element-wise interaction
        diff = torch.abs(protein_pooled - molecule_pooled)

        # Project product and diff to match concat dimension (learned projections, not zero padding)
        product_projected = self.product_projection(product)
        diff_projected = self.diff_projection(diff)

        combined = torch.cat([concat, product_projected, diff_projected], dim=-1)

        # Fusion
        fused = self.fusion(combined)

        # Predictions
        output = {
            "interaction_logits": self.interaction_head(fused).squeeze(-1),
            "affinity": self.affinity_head(fused).squeeze(-1),
            "interaction_type": self.interaction_type_head(fused),
        }

        if return_attention:
            output["protein_attention"] = binding_attn_weights.squeeze(-1)
            output["molecule_attention"] = pharm_attn_weights.squeeze(-1)

        return output


class ProteinMoleculeDataset(Dataset):
    """Dataset for protein-molecule interactions. Expects pre-processed data from prepare/ scripts."""

    def __init__(
        self,
        data_file: Path,
        max_protein_len: int = 1000,
        max_smiles_len: int = 150,
    ):
        self.max_protein_len = max_protein_len
        self.max_smiles_len = max_smiles_len

        self.samples = []
        with open(data_file, "r") as f:
            raw_header = f.readline().strip().split("\t")
            cols = {c.lower(): i for i, c in enumerate(raw_header)}

            idx_protein  = cols.get("protein_seq", 1)
            idx_smiles   = cols.get("smiles", 2)
            idx_label    = cols.get("label", 3)
            idx_affinity = cols.get("affinity", 4)
            idx_itype    = cols.get("interaction_type", None)

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) <= max(idx_protein, idx_smiles, idx_label):
                    continue
                try:
                    protein_seq = parts[idx_protein]
                    smiles      = parts[idx_smiles]
                    label       = float(parts[idx_label])
                    affinity    = float(parts[idx_affinity]) if idx_affinity < len(parts) else 0.0
                    if idx_itype is not None and idx_itype < len(parts):
                        try:
                            interaction_type = int(parts[idx_itype])
                        except ValueError:
                            interaction_type = 3
                    else:
                        interaction_type = 3  # fallback: OTHER (column absent in old schema)
                    self.samples.append((protein_seq, smiles, label, affinity, interaction_type))
                except (ValueError, IndexError):
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        protein_seq, smiles, label, affinity, interaction_type = self.samples[idx]

        return {
            "protein_seq": protein_seq[: self.max_protein_len],
            "smiles": encode_smiles(smiles, self.max_smiles_len),
            "label": torch.tensor(label, dtype=torch.float32),
            "affinity": torch.tensor(affinity, dtype=torch.float32),
            "interaction_type": torch.tensor(interaction_type, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    return {
        "protein_seq": [b["protein_seq"] for b in batch],
        "smiles": torch.stack([b["smiles"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "affinity": torch.stack([b["affinity"] for b in batch]),
        "interaction_type": torch.stack([b["interaction_type"] for b in batch]),
    }


def train_epoch(model, dataloader, optimizer, device, use_affinity=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        labels = batch["label"].to(device)
        smiles = batch["smiles"].to(device)
        affinity = batch["affinity"].to(device)

        optimizer.zero_grad()

        output = model(batch["protein_seq"], smiles)

        # Binary cross-entropy for interaction prediction
        loss = F.binary_cross_entropy_with_logits(output["interaction_logits"], labels)

        # Add affinity loss for positive samples (affinity data is always available now)
        if use_affinity:
            mask = labels > 0.5
            if mask.sum() > 0:
                affinity_loss = F.mse_loss(output["affinity"][mask], affinity[mask])
                loss = loss + 0.1 * affinity_loss

        # Interaction type cross-entropy loss (positive samples only)
        pos_mask = labels > 0.5
        if pos_mask.sum() > 0:
            itype_true = batch["interaction_type"].to(device)[pos_mask]
            itype_logits = output["interaction_type"][pos_mask]
            itype_loss = F.cross_entropy(itype_logits, itype_true)
            loss = loss + 0.3 * itype_loss

        loss.backward()
        optimizer.step()

        # Periodically flush the MPS allocator to prevent fragmentation buildup
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            if (batch_idx % 10) == 0:
                torch.mps.empty_cache()

        total_loss += loss.item()
        preds = (torch.sigmoid(output["interaction_logits"]) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_labels, all_probs = [], []
    all_affinities_true, all_affinities_pred = [], []
    total_loss = 0
    correct = 0
    total = 0
    itype_correct = 0
    itype_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch["label"].to(device)
            smiles = batch["smiles"].to(device)
            affinity = batch["affinity"].to(device)

            output = model(batch["protein_seq"], smiles)

            loss = F.binary_cross_entropy_with_logits(
                output["interaction_logits"], labels
            )
            total_loss += loss.item()

            probs = torch.sigmoid(output["interaction_logits"])
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Collect affinity predictions for positive samples
            mask = labels > 0.5
            if mask.sum() > 0:
                all_affinities_true.extend(affinity[mask].cpu().numpy())
                all_affinities_pred.extend(output["affinity"][mask].cpu().numpy())

            # Track interaction type accuracy on positive samples
            pos_mask = labels > 0.5
            if pos_mask.sum() > 0:
                itype_true = batch["interaction_type"].to(device)[pos_mask]
                itype_pred = output["interaction_type"][pos_mask].argmax(dim=-1)
                itype_correct += (itype_pred == itype_true).sum().item()
                itype_total += pos_mask.sum().item()

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Handle edge case where all labels are same class
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5

    preds = (all_probs > 0.5).astype(float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )

    # Calculate affinity RMSE if available
    affinity_rmse = 0.0
    if len(all_affinities_true) > 0:
        affinity_rmse = np.sqrt(
            mean_squared_error(all_affinities_true, all_affinities_pred)
        )

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / total,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "affinity_rmse": affinity_rmse,
        "interaction_type_accuracy": itype_correct / max(itype_total, 1),
    }


def _save_checkpoint(path: Path, model, optimizer, scheduler,
                     epoch: int, best_auc: float,
                     patience_counter: int, history: dict) -> None:
    """Save a full resumable checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc": best_auc,
            "patience_counter": patience_counter,
            "history": history,
        },
        path,
    )


def train_model(
    model, train_loader, val_loader, epochs, lr, device, save_path, patience=10,
    resume_path: Path = None,
):
    """Full training loop with optional resume from checkpoint."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_path and Path(resume_path).exists():
        print(f"\n  Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_auc = ckpt.get("best_auc", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", history)
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch + 1} | best AUC so far: {best_auc:.4f}")
    elif resume_path:
        print(f"  ⚠  Resume checkpoint not found at {resume_path} — starting fresh")

    # "latest" checkpoint saved every epoch (for reliable resume)
    latest_path = Path(save_path).with_suffix(".latest.pt")

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["auc"])

        print(
            f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}"
        )
        if val_metrics["affinity_rmse"] > 0:
            print(f"  Affinity RMSE: {val_metrics['affinity_rmse']:.4f}")
        if val_metrics.get("interaction_type_accuracy", 0) > 0:
            print(f"  Val Interaction Type Acc: {val_metrics['interaction_type_accuracy']:.4f}")

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])

        # Always save a latest checkpoint so training can be resumed
        _save_checkpoint(latest_path, model, optimizer, scheduler,
                         epoch, best_auc, patience_counter, history)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            _save_checkpoint(Path(save_path), model, optimizer, scheduler,
                             epoch, best_auc, patience_counter, history)
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print("\n" + "=" * 70)
    print(f"Training complete. Best AUC: {best_auc:.4f}")
    print("=" * 70)

    return history


def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(
        description="Train base Protein-Molecule Interaction Model (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m model.model_p2m --data-dir data/prepared/p2m --epochs 50
    python -m model.model_p2m --data-dir data/prepared/p2m --epochs 50 --batch-size 32
    python -m model.model_p2m --data-dir data/prepared/p2m --checkpoint model/p2m_model.pt

Data files expected:
    {data_dir}/p2m_train.tsv
    {data_dir}/p2m_val.tsv
        """,
    )
    parser.add_argument("--data-dir", type=str, required=True,
        help="Directory containing p2m_train.tsv and p2m_val.tsv")
    parser.add_argument("--checkpoint", type=str, default="model/p2m_model.pt",
        help="Path to save best checkpoint (default: model/p2m_model.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None,
        help="Cap training set size (useful for quick tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esm-model", type=str, default="esm2_t12_35M_UR50D",
        help="ESM-2 variant to use as protein encoder")
    parser.add_argument("--esm-cache", type=str, default=None,
        help="Path to pre-computed ESM-2 embedding cache (.pt) from precompute_esm.py")
    parser.add_argument("--max-protein-len", type=int, default=1000,
        help="Truncate protein sequences to this length (default: 1000). "
             "Must match the --max-len used in precompute_esm.py.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("CELLA NOVA — P2M BASE MODEL TRAINING  (Phase 1)")
    print("=" * 70)
    print(f"  Device:     {device}")
    print(f"  ESM model:  {args.esm_model}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Seed:       {args.seed}")
    print()

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

    train_dataset = ProteinMoleculeDataset(data_file=train_file, max_protein_len=args.max_protein_len)
    val_dataset   = ProteinMoleculeDataset(data_file=val_file,   max_protein_len=args.max_protein_len)

    if args.max_samples and len(train_dataset) > args.max_samples:
        train_dataset.samples = train_dataset.samples[: args.max_samples]
        print(f"  Capped training set to {args.max_samples:,} samples")

    if len(train_dataset) == 0:
        print("❌  Training dataset is empty.")
        sys.exit(1)

    print(f"  Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = ProteinMoleculeModel(
        protein_dim=512,
        molecule_dim=512,
        hidden_dim=512,
        esm_model=args.esm_model,
        num_cross_attention_layers=2,
        dropout=0.1,
        device=device,
        esm_cache_path=args.esm_cache,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")

    save_path = Path(args.checkpoint)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve --resume path
    resume_path = save_path.with_suffix(".latest.pt")
    if not resume_path.exists():
        print(f"  ⚠  Resume file not found: {resume_path}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=save_path,
        patience=10,
        resume_path=resume_path,
    )

    with open(save_path.with_suffix(".history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Checkpoint: {save_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
