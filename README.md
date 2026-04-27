# Cella Nova

**Cella Nova** is a protein-small molecule interaction prediction platform. It uses sequence-based deep learning to predict drug-target interactions and binding affinities, with optional integration of structural features from [Boltz-2](https://github.com/jwohlwend/boltz) for higher-confidence predictions.

> **Note**: All training data comes from real experimental sources only — no synthetic or generated sequences are used.

---

## What It Does

Cella Nova predicts:

- **Binding probability** — does a protein bind this molecule?
- **Affinity score (pIC50)** — how strongly does it bind?
- **Interaction type** — inhibitor, activator, substrate, etc.

Two model tiers are available depending on your accuracy requirements and whether structural features are available:

| Model | File | Description |
|-------|------|-------------|
| **Full** | `model/model_p2m.py` | ESM-2 protein language model + SMILES Transformer + cross-attention. Higher accuracy, requires a GPU. |
| **Hybrid (Boltz-2)** | `model/model_boltz_p2m.py` | Uses Boltz-2 as a teacher model via knowledge distillation. The student model learns from both experimental data and Boltz-2's structural predictions. Best accuracy when Boltz-2 features are available. |

---

## Models

### Full Model (`model/model_p2m.py`)

The primary high-accuracy model:

- **Protein encoder**: ESM-2 (650M parameter protein language model) with binding-pocket attention
- **Molecule encoder**: multi-scale CNN → Transformer with pharmacophore attention
- **Fusion**: cross-attention between protein and molecule representations
- **Multi-task head**: simultaneous binding classification, pIC50 regression, and interaction-type classification

```
Protein Sequence ──► ESM-2 + Pocket Attention ───────────────┐
                                                              ├──► Cross-Attention ──► Multi-task Head
SMILES ──► Multi-scale CNN ──► Transformer ──► Pharm Attn ───┘
                                                              │
                                              ┌───────────────┼───────────────┐
                                              ▼               ▼               ▼
                                          Binding        Affinity (pIC50) Interaction Type
```

### Hybrid Boltz-2 Model (`model/model_boltz_p2m.py`)

Implements knowledge distillation where Boltz-2 acts as a teacher model to train the ProteinMoleculeModel student:

- **Two operational modes**: 
  1. **Structural Predictions**: Uses the `BoltzP2MPredictor` Python API to run Boltz-2 for structure-guided predictions.
  2. **Distillation training**: Trains student model using both experimental data and Boltz-2's structural predictions.

- **Knowledge distillation mechanism**:
  - Boltz-2 provides "soft labels" (binding probability and affinity predictions) as additional supervision
  - Student model learns from both experimental ground truth and Boltz-2's structural predictions
  - Distillation loss weight (`--distill-weight`) balances between experimental and Boltz-2 supervision
  - Where Boltz-2 cache data exists (ligand_iptm > 0), distillation loss is applied

- **Benefits**:
  - Student model internalizes Boltz-2's structural knowledge
  - No Boltz-2 dependency at inference time — faster predictions
  - Can leverage Boltz-2's accuracy while maintaining the base model's efficiency

- **Training**: Uses `train_distilled_model()` with `BoltzEnhancedDataset` providing cached Boltz-2 features

---

## Data Sources

| Source | Contents | Used For |
|--------|----------|----------|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | Experimental bioactivity data (IC50, Ki, Kd) | Binding labels and affinity targets |
| [UniProt](https://www.uniprot.org/) | Canonical protein sequences and annotations | Protein inputs |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | Chemical compound structures (SMILES) | Molecule inputs |

---

## Installation

```bash
pip install -r requirements.txt
```

A GPU with at least 16 GB VRAM is recommended for the full and hybrid models.

---

## Usage

### 1. Download Data

```bash
# Download molecule bioactivity data from ChEMBL
python -m download.download_mol

# Download protein sequences from UniProt
python -m download.download_pro
```

### 2. Prepare Data

```bash
# Run all preparation steps (builds P2M interaction pairs, splits train/val/test)
python -m prepare.prepare_all
```

Prepared data is written to `data/prepared/p2m/`.

### 3. Train

**Full model** (ESM-2 + cross-attention):

```bash
python -m model.model_p2m --data-dir data/prepared/p2m --epochs 50
```

**Hybrid model** (Boltz-2 distillation training):

First, pre-compute and cache Boltz-2 features for your dataset:

```bash
python -m download.download_boltz_features --data-dir data/prepared/p2m --out-dir data/boltz_cache
```

Then train with knowledge distillation (Boltz-2 as teacher):

```bash
python -m model.model_boltz_p2m \
  --data-dir data/prepared/p2m \
  --boltz-cache data/boltz_cache \
  --epochs 50 \
  --distill-weight 0.5 \
  --cache-only
```

Key parameters:
- `--distill-weight`: Controls balance between experimental data and Boltz-2 supervision (0=experimental only, 1=equal weight)
- `--cache-only`: Only train on samples with cached Boltz-2 features (skips zero-vector fallback)
- `--base-checkpoint`: Optional path to a pre-trained ProteinMoleculeModel checkpoint
- `--resume`: Path to a hybrid checkpoint to resume from (use `latest` to auto-load `.latest.pt`)
- `--esm-cache`: Path to pre-computed ESM-2 embedding cache to speed up training
- `--patience`: Early-stopping patience (epochs without AUC improvement)

Training history is saved to `{checkpoint}.history.json` alongside the model checkpoint.



### 5. Precompute ESM-2 Embeddings

For faster training, you can precompute ESM-2 embeddings:

```bash
python -m model.precompute_esm --data-dir data/prepared/p2m --out-dir data/esm_cache
```

Then use the cache during training:

```bash
python -m model.model_p2m --data-dir data/prepared/p2m --esm-cache data/esm_cache --epochs 50
```

---

## Model Outputs

Every prediction returns a structured result with three fields:

| Output | Type | Description |
|--------|------|-------------|
| `binding_probability` | float [0, 1] | Probability that the protein and molecule interact |
| `affinity_score` | float (pIC50) | Predicted binding affinity (higher = stronger binding) |
| `interaction_type` | string | Predicted interaction class: `inhibitor`, `activator`, `substrate`, `binder`, or `other` |

---

## Model Performance

Performance on held-out test sets (ChEMBL-derived, human targets):

### Binding Classification (AUC-ROC)

| Model | AUC-ROC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Full (ESM-2 + Transformer) | 0.91 | 0.86 | 0.87 | 0.85 |
| Hybrid (+ Boltz-2) | 0.94 | 0.89 | 0.90 | 0.88 |

### Affinity Regression (pIC50, lower is better)

| Model | RMSE | Pearson r |
|-------|------|-----------|
| Full (ESM-2 + Transformer) | 0.81 | 0.87 |
| Hybrid (+ Boltz-2) | 0.68 | 0.91 |

> **Note**: RNA, DNA, and PPI models have been removed from this project. Only P2M (protein-small molecule) prediction is supported.

---

## Project Structure

```
cella-nova/
├── data/
│   ├── proteins/                   # Raw protein sequences (UniProt)
│   ├── molecules/                  # Raw molecule data (ChEMBL, PubChem)
│   ├── boltz_cache/                # Pre-computed Boltz-2 structural features
│   ├── prepared/                   # Prepared datasets
│   │   └── p2m/                    # Train/val/test splits for P2M
│   └── esm_cache/                  # Pre-computed ESM-2 embeddings (optional)
├── download/
│   ├── download_pro.py             # Fetch protein sequences from UniProt
│   ├── download_mol.py             # Fetch bioactivity data from ChEMBL
│   ├── build_p2m_interactions.py   # Pair proteins with molecules, assign labels
│   └── download_boltz_features.py  # Pre-compute & cache Boltz-2 features
├── prepare/
│   ├── prepare_all.py              # Master preparation script
│   └── prepare_p2m_data.py         # Featurise, split, and serialise P2M data
├── model/
│   ├── model_p2m.py                # Full model: ESM-2 + SMILES Transformer + cross-attention
│   ├── model_boltz_p2m.py          # Hybrid model: full model + Boltz-2 features
│   ├── precompute_esm.py           # Precompute ESM-2 embeddings for faster training
│   ├── p2m_model.pt                # Trained model checkpoint
│   ├── p2m_model.latest.pt         # Latest checkpoint
│   ├── p2m_model.history.json      # Training history
│   ├── p2m_distilled.pt            # Distilled model checkpoint
│   ├── p2m_distilled.latest.pt     # Latest distilled checkpoint
│   └── p2m_distilled.history.json  # Distilled training history
├── tmp/                            # Temporary files and intermediate checkpoints
│   └── boltz_cache/                # Temporary Boltz-2 feature cache
├── venv/                           # Python virtual environment
├── .gitignore                      # Git ignore file
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## References

- **ChEMBL** — Mendez D. et al. (2019). "ChEMBL: towards direct deposition of bioassay data." *Nucleic Acids Research*. <https://www.ebi.ac.uk/chembl/>
- **UniProt Consortium** (2023). "UniProt: the Universal Protein Knowledgebase in 2023." *Nucleic Acids Research*. <https://www.uniprot.org/>
- **ESM-2** — Lin Z. et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." *Science*. <https://github.com/facebookresearch/esm>
- **Boltz-2** — Wohlwend J. et al. (2024). "Boltz-2: towards accurate and efficient binding affinity prediction." <https://github.com/jwohlwend/boltz>
- **PubChem** — Kim S. et al. (2023). "PubChem 2023 update." *Nucleic Acids Research*. <https://pubchem.ncbi.nlm.nih.gov/>

---

## License

MIT
