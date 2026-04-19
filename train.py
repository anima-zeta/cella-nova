#!/usr/bin/env python3
"""
Unified Training Script for Cella Nova Models
==============================================

Train any of the interaction prediction models:
- P2P: Protein-Protein Interaction
- P2D: Protein-DNA Interaction
- P2R: Protein-RNA Interaction
- P2M: Protein-Molecule Interaction

Usage:
    python train.py --model p2p --epochs 50
    python train.py --model p2d --epochs 50 --batch-size 16
    python train.py --model p2r --epochs 50 --lr 1e-4
    python train.py --model p2m --epochs 50 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Project paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "prepared"
MODEL_DIR = SCRIPT_DIR / "models"


# =============================================================================
# DATASETS
# =============================================================================


class PPIDataset(Dataset):
    """Dataset for Protein-Protein Interaction prediction"""

    def __init__(self, data_file: Path, max_seq_length: int = 1000):
        self.max_seq_length = max_seq_length
        self.samples = []

        with open(data_file, "r") as f:
            header = f.readline().strip().split("\t")

            # Find column indices
            cols = {col.lower(): i for i, col in enumerate(header)}

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                # Try different column name formats
                id1 = parts[cols.get("protein1_id", cols.get("uniprot1", 0))]
                id2 = parts[cols.get("protein2_id", cols.get("uniprot2", 1))]

                # Get sequences if available
                seq1 = (
                    parts[cols.get("protein1_seq", cols.get("seq1", -1))]
                    if "protein1_seq" in cols or "seq1" in cols
                    else ""
                )
                seq2 = (
                    parts[cols.get("protein2_seq", cols.get("seq2", -1))]
                    if "protein2_seq" in cols or "seq2" in cols
                    else ""
                )

                label = float(parts[cols.get("label", 2)])

                self.samples.append(
                    {
                        "id1": id1,
                        "id2": id2,
                        "seq1": seq1[:max_seq_length] if seq1 else "",
                        "seq2": seq2[:max_seq_length] if seq2 else "",
                        "label": label,
                    }
                )

        print(f"  Loaded {len(self.samples):,} samples from {data_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "seq1": sample["seq1"],
            "seq2": sample["seq2"],
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


class PDNADataset(Dataset):
    """Dataset for Protein-DNA Interaction prediction"""

    def __init__(
        self, data_file: Path, max_protein_length: int = 1000, max_dna_length: int = 200
    ):
        self.max_protein_length = max_protein_length
        self.max_dna_length = max_dna_length
        self.samples = []

        with open(data_file, "r") as f:
            header = f.readline().strip().split("\t")
            cols = {col.lower(): i for i, col in enumerate(header)}

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                protein_id = parts[cols.get("protein_id", 0)]
                protein_seq = parts[cols.get("protein_seq", 1)]
                dna_seq = parts[cols.get("dna_seq", 2)]
                label = float(parts[cols.get("label", 3)])

                self.samples.append(
                    {
                        "protein_id": protein_id,
                        "protein_seq": protein_seq[:max_protein_length],
                        "dna_seq": dna_seq[:max_dna_length],
                        "label": label,
                    }
                )

        print(f"  Loaded {len(self.samples):,} samples from {data_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "protein_seq": sample["protein_seq"],
            "dna_seq": sample["dna_seq"],
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


class PRNADataset(Dataset):
    """Dataset for Protein-RNA Interaction prediction"""

    def __init__(
        self, data_file: Path, max_protein_length: int = 1000, max_rna_length: int = 200
    ):
        self.max_protein_length = max_protein_length
        self.max_rna_length = max_rna_length
        self.samples = []

        with open(data_file, "r") as f:
            header = f.readline().strip().split("\t")
            cols = {col.lower(): i for i, col in enumerate(header)}

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                protein_id = parts[cols.get("protein_id", 0)]
                protein_seq = parts[cols.get("protein_seq", 1)]
                rna_seq = parts[cols.get("rna_seq", 2)]
                label = float(parts[cols.get("label", 3)])

                self.samples.append(
                    {
                        "protein_id": protein_id,
                        "protein_seq": protein_seq[:max_protein_length],
                        "rna_seq": rna_seq[:max_rna_length],
                        "label": label,
                    }
                )

        print(f"  Loaded {len(self.samples):,} samples from {data_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "protein_seq": sample["protein_seq"],
            "rna_seq": sample["rna_seq"],
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


class PMolDataset(Dataset):
    """Dataset for Protein-Molecule Interaction prediction"""

    def __init__(
        self,
        data_file: Path,
        max_protein_length: int = 1000,
        max_smiles_length: int = 200,
    ):
        self.max_protein_length = max_protein_length
        self.max_smiles_length = max_smiles_length
        self.samples = []

        with open(data_file, "r") as f:
            header = f.readline().strip().split("\t")
            cols = {col.lower(): i for i, col in enumerate(header)}

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                protein_id = parts[cols.get("protein_id", 0)]
                protein_seq = parts[cols.get("protein_seq", 1)]
                smiles = parts[cols.get("smiles", 2)]
                label = float(parts[cols.get("label", 3)])

                # Get affinity if available
                affinity = 0.0
                if "affinity" in cols and len(parts) > cols["affinity"]:
                    try:
                        affinity = float(parts[cols["affinity"]])
                    except ValueError:
                        pass

                self.samples.append(
                    {
                        "protein_id": protein_id,
                        "protein_seq": protein_seq[:max_protein_length],
                        "smiles": smiles[:max_smiles_length],
                        "label": label,
                        "affinity": affinity,
                    }
                )

        print(f"  Loaded {len(self.samples):,} samples from {data_file.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "protein_seq": sample["protein_seq"],
            "smiles": sample["smiles"],
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "affinity": torch.tensor(sample["affinity"], dtype=torch.float32),
        }


# =============================================================================
# SEQUENCE ENCODERS
# =============================================================================


class ProteinEncoder(nn.Module):
    """Simple protein sequence encoder using embeddings + CNN"""

    def __init__(
        self,
        vocab_size: int = 26,  # 20 amino acids + special tokens
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()

        self.aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWXY")}
        self.aa_to_idx["<PAD>"] = 21
        self.aa_to_idx["<UNK>"] = 22

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=21)

        # Multi-scale CNN
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k // 2)
                for k in [3, 5, 7]
            ]
        )

        self.lstm = nn.LSTM(
            hidden_dim * 3,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.1)

    def encode_sequence(
        self, sequences: List[str], device: torch.device
    ) -> torch.Tensor:
        """Encode list of sequences to tensor"""
        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)

        encoded = torch.full((batch_size, max_len), 21, dtype=torch.long, device=device)

        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                encoded[i, j] = self.aa_to_idx.get(aa.upper(), 22)

        return encoded

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        # Encode sequences
        x = self.encode_sequence(sequences, device)  # (B, L)

        # Embedding
        x = self.embedding(x)  # (B, L, E)
        x = x.transpose(1, 2)  # (B, E, L)

        # Multi-scale CNN
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)  # (B, H*3, L)
        x = x.transpose(1, 2)  # (B, L, H*3)

        # BiLSTM
        x, _ = self.lstm(x)  # (B, L, H*2)

        # Global max pooling
        x = torch.max(x, dim=1)[0]  # (B, H*2)

        # Final projection
        x = self.dropout(x)
        x = self.fc(x)  # (B, output_dim)

        return x


class DNAEncoder(nn.Module):
    """DNA sequence encoder"""

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.nt_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "<PAD>": 5}

        self.embedding = nn.Embedding(6, embed_dim, padding_idx=5)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k // 2)
                for k in [3, 5, 7, 9]
            ]
        )

        self.lstm = nn.LSTM(
            hidden_dim * 4,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def encode_sequence(
        self, sequences: List[str], device: torch.device
    ) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)

        encoded = torch.full((batch_size, max_len), 5, dtype=torch.long, device=device)

        for i, seq in enumerate(sequences):
            for j, nt in enumerate(seq):
                encoded[i, j] = self.nt_to_idx.get(nt.upper(), 4)

        return encoded

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        x = self.encode_sequence(sequences, device)
        x = self.embedding(x)
        x = x.transpose(1, 2)

        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = torch.max(x, dim=1)[0]
        x = self.fc(x)

        return x


class RNAEncoder(nn.Module):
    """RNA sequence encoder"""

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.nt_to_idx = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4, "<PAD>": 5}

        self.embedding = nn.Embedding(6, embed_dim, padding_idx=5)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k // 2)
                for k in [3, 5, 7]
            ]
        )

        self.lstm = nn.LSTM(
            hidden_dim * 3,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def encode_sequence(
        self, sequences: List[str], device: torch.device
    ) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)

        encoded = torch.full((batch_size, max_len), 5, dtype=torch.long, device=device)

        for i, seq in enumerate(sequences):
            for j, nt in enumerate(seq):
                encoded[i, j] = self.nt_to_idx.get(nt.upper(), 4)

        return encoded

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        x = self.encode_sequence(sequences, device)
        x = self.embedding(x)
        x = x.transpose(1, 2)

        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = torch.max(x, dim=1)[0]
        x = self.fc(x)

        return x


class SMILESEncoder(nn.Module):
    """SMILES molecule encoder"""

    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        # Common SMILES characters
        self.char_to_idx = {
            c: i for i, c in enumerate("CNOSPFClBrI=#@+\\/-[]()123456789%cnops")
        }
        self.char_to_idx["<PAD>"] = len(self.char_to_idx)
        self.char_to_idx["<UNK>"] = len(self.char_to_idx)

        self.embedding = nn.Embedding(
            len(self.char_to_idx), embed_dim, padding_idx=self.char_to_idx["<PAD>"]
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=k, padding=k // 2)
                for k in [3, 5, 7]
            ]
        )

        self.lstm = nn.LSTM(
            hidden_dim * 3,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def encode_sequence(
        self, sequences: List[str], device: torch.device
    ) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        batch_size = len(sequences)

        pad_idx = self.char_to_idx["<PAD>"]
        unk_idx = self.char_to_idx["<UNK>"]

        encoded = torch.full(
            (batch_size, max_len), pad_idx, dtype=torch.long, device=device
        )

        for i, seq in enumerate(sequences):
            for j, c in enumerate(seq):
                encoded[i, j] = self.char_to_idx.get(c, unk_idx)

        return encoded

    def forward(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        x = self.encode_sequence(sequences, device)
        x = self.embedding(x)
        x = x.transpose(1, 2)

        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = torch.max(x, dim=1)[0]
        x = self.fc(x)

        return x


# =============================================================================
# MODELS
# =============================================================================


class PPIModel(nn.Module):
    """Protein-Protein Interaction prediction model"""

    def __init__(self, protein_dim: int = 512, hidden_dim: int = 256):
        super().__init__()

        self.protein_encoder = ProteinEncoder(output_dim=protein_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(protein_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, seq1: List[str], seq2: List[str], device: torch.device
    ) -> torch.Tensor:
        # Encode proteins
        emb1 = self.protein_encoder(seq1, device)  # (B, D)
        emb2 = self.protein_encoder(seq2, device)  # (B, D)

        # Cross attention
        emb1_unsq = emb1.unsqueeze(1)  # (B, 1, D)
        emb2_unsq = emb2.unsqueeze(1)  # (B, 1, D)

        attn_out, _ = self.cross_attention(emb1_unsq, emb2_unsq, emb2_unsq)
        attn_out = attn_out.squeeze(1)  # (B, D)

        # Combine features
        combined = torch.cat([emb1, emb2, attn_out], dim=-1)  # (B, D*3)

        # Classify
        logits = self.classifier(combined).squeeze(-1)  # (B,)

        return logits


class PDNAModel(nn.Module):
    """Protein-DNA Interaction prediction model"""

    def __init__(
        self, protein_dim: int = 512, dna_dim: int = 256, hidden_dim: int = 256
    ):
        super().__init__()

        self.protein_encoder = ProteinEncoder(output_dim=protein_dim)
        self.dna_encoder = DNAEncoder(output_dim=dna_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # Project DNA to protein dimension for attention
        self.dna_proj = nn.Linear(dna_dim, protein_dim)

        self.classifier = nn.Sequential(
            nn.Linear(protein_dim + dna_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        protein_seqs: List[str],
        dna_seqs: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        # Encode sequences
        protein_emb = self.protein_encoder(protein_seqs, device)  # (B, D_p)
        dna_emb = self.dna_encoder(dna_seqs, device)  # (B, D_d)

        # Cross attention
        protein_unsq = protein_emb.unsqueeze(1)  # (B, 1, D_p)
        dna_proj = self.dna_proj(dna_emb).unsqueeze(1)  # (B, 1, D_p)

        attn_out, _ = self.cross_attention(protein_unsq, dna_proj, dna_proj)
        attn_out = attn_out.squeeze(1)  # (B, D_p)

        # Combine features
        combined = torch.cat([protein_emb, dna_emb, attn_out], dim=-1)

        # Classify
        logits = self.classifier(combined).squeeze(-1)

        return logits


class PRNAModel(nn.Module):
    """Protein-RNA Interaction prediction model"""

    def __init__(
        self, protein_dim: int = 512, rna_dim: int = 256, hidden_dim: int = 256
    ):
        super().__init__()

        self.protein_encoder = ProteinEncoder(output_dim=protein_dim)
        self.rna_encoder = RNAEncoder(output_dim=rna_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        self.rna_proj = nn.Linear(rna_dim, protein_dim)

        self.classifier = nn.Sequential(
            nn.Linear(protein_dim + rna_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        protein_seqs: List[str],
        rna_seqs: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        # Encode sequences
        protein_emb = self.protein_encoder(protein_seqs, device)
        rna_emb = self.rna_encoder(rna_seqs, device)

        # Cross attention
        protein_unsq = protein_emb.unsqueeze(1)
        rna_proj = self.rna_proj(rna_emb).unsqueeze(1)

        attn_out, _ = self.cross_attention(protein_unsq, rna_proj, rna_proj)
        attn_out = attn_out.squeeze(1)

        # Combine features
        combined = torch.cat([protein_emb, rna_emb, attn_out], dim=-1)

        # Classify
        logits = self.classifier(combined).squeeze(-1)

        return logits


class PMolModel(nn.Module):
    """Protein-Molecule Interaction prediction model"""

    def __init__(
        self, protein_dim: int = 512, mol_dim: int = 256, hidden_dim: int = 256
    ):
        super().__init__()

        self.protein_encoder = ProteinEncoder(output_dim=protein_dim)
        self.mol_encoder = SMILESEncoder(output_dim=mol_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=protein_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        self.mol_proj = nn.Linear(mol_dim, protein_dim)

        # Binding classification head
        self.classifier = nn.Sequential(
            nn.Linear(protein_dim + mol_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Affinity regression head
        self.affinity_head = nn.Sequential(
            nn.Linear(protein_dim + mol_dim + protein_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        protein_seqs: List[str],
        smiles: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode sequences
        protein_emb = self.protein_encoder(protein_seqs, device)
        mol_emb = self.mol_encoder(smiles, device)

        # Cross attention
        protein_unsq = protein_emb.unsqueeze(1)
        mol_proj = self.mol_proj(mol_emb).unsqueeze(1)

        attn_out, _ = self.cross_attention(protein_unsq, mol_proj, mol_proj)
        attn_out = attn_out.squeeze(1)

        # Combine features
        combined = torch.cat([protein_emb, mol_emb, attn_out], dim=-1)

        # Predict
        logits = self.classifier(combined).squeeze(-1)
        affinity = self.affinity_head(combined).squeeze(-1)

        return logits, affinity


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_epoch_ppi(model, dataloader, optimizer, device):
    """Train PPI model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        labels = batch["label"].to(device)
        logits = model(batch["seq1"], batch["seq2"], device)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def train_epoch_pdna(model, dataloader, optimizer, device):
    """Train P2D model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        labels = batch["label"].to(device)
        logits = model(batch["protein_seq"], batch["dna_seq"], device)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def train_epoch_prna(model, dataloader, optimizer, device):
    """Train P2R model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        labels = batch["label"].to(device)
        logits = model(batch["protein_seq"], batch["rna_seq"], device)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def train_epoch_pmol(model, dataloader, optimizer, device):
    """Train P2M model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()

        labels = batch["label"].to(device)
        affinities = batch["affinity"].to(device)

        logits, pred_affinity = model(batch["protein_seq"], batch["smiles"], device)

        # Combined loss: classification + affinity regression
        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
        aff_loss = F.mse_loss(pred_affinity, affinities)
        loss = cls_loss + 0.1 * aff_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / len(dataloader), "accuracy": correct / total}


def evaluate(model, dataloader, device, model_type: str):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels = batch["label"].to(device)

            if model_type == "p2p":
                logits = model(batch["seq1"], batch["seq2"], device)
            elif model_type == "p2d":
                logits = model(batch["protein_seq"], batch["dna_seq"], device)
            elif model_type == "p2r":
                logits = model(batch["protein_seq"], batch["rna_seq"], device)
            elif model_type == "p2m":
                logits, _ = model(batch["protein_seq"], batch["smiles"], device)

            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    auc = (
        roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    )
    preds = (all_probs > 0.5).astype(float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average="binary", zero_division=0
    )
    accuracy = (preds == all_labels).mean()

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: Path,
    model_type: str,
    patience: int = 10,
):
    """Full training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_auc = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    # Select training function based on model type
    train_fn = {
        "p2p": train_epoch_ppi,
        "p2d": train_epoch_pdna,
        "p2r": train_epoch_prna,
        "p2m": train_epoch_pmol,
    }[model_type]

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_fn(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, model_type)

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

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "history": history,
                },
                save_path,
            )
            print(f"  ✓ Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print("\n" + "=" * 70)
    print(f"Training complete. Best AUC: {best_auc:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Train Cella Nova Interaction Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py --model p2p --epochs 50
    python train.py --model p2d --epochs 50 --batch-size 16
    python train.py --model p2r --epochs 50 --lr 1e-4
    python train.py --model p2m --epochs 50 --device cuda
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["p2p", "p2d", "p2r", "p2m"],
        help="Model type to train",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: cuda, mps, cpu, or auto (default: auto)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: data/prepared/{model})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        help="Output directory for saved models (default: models/)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print(f"CELLA NOVA - {args.model.upper()} MODEL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()

    # Set data directory
    data_dir = args.data_dir or DATA_DIR / args.model

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"   Run: python -m prepare.prepare_{args.model}_data")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")

    train_file = data_dir / f"{args.model}_train.tsv"
    val_file = data_dir / f"{args.model}_val.tsv"
    test_file = data_dir / f"{args.model}_test.tsv"

    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        sys.exit(1)

    # Create datasets based on model type
    if args.model == "p2p":
        train_dataset = PPIDataset(train_file)
        val_dataset = PPIDataset(val_file)
        model = PPIModel().to(device)
    elif args.model == "p2d":
        train_dataset = PDNADataset(train_file)
        val_dataset = PDNADataset(val_file)
        model = PDNAModel().to(device)
    elif args.model == "p2r":
        train_dataset = PRNADataset(train_file)
        val_dataset = PRNADataset(val_file)
        model = PRNAModel().to(device)
    elif args.model == "p2m":
        train_dataset = PMolDataset(train_file)
        val_dataset = PMolDataset(val_file)
        model = PMolModel().to(device)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    save_path = args.output_dir / f"{args.model}_model.pt"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=save_path,
        model_type=args.model,
        patience=args.patience,
    )

    # Evaluate on test set
    if test_file.exists():
        print("\n" + "=" * 70)
        print("EVALUATING ON TEST SET")
        print("=" * 70)

        if args.model == "p2p":
            test_dataset = PPIDataset(test_file)
        elif args.model == "p2d":
            test_dataset = PDNADataset(test_file)
        elif args.model == "p2r":
            test_dataset = PRNADataset(test_file)
        elif args.model == "p2m":
            test_dataset = PMolDataset(test_file)

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Load best model
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = evaluate(model, test_loader, device, args.model)

        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Test F1: {test_metrics['f1']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")

    print("\n" + "=" * 70)
    print(f"Model saved to: {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
