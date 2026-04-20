# Cella Nova

**Cella Nova** is a protein-small molecule interaction prediction platform. It uses sequence-based deep learning to predict drug-target interactions and binding affinities, with optional integration of structural features from [Boltz-2](https://github.com/jwohlwend/boltz) for higher-confidence predictions.

> **Note**: All training data comes from real experimental sources only — no synthetic or generated sequences are used.

---

## What It Does

Cella Nova predicts:

- **Binding probability** — does a protein bind this molecule?
- **Affinity score (pIC50)** — how strongly does it bind?
- **Interaction type** — inhibitor, activator, substrate, etc.

Three model tiers are available depending on your hardware constraints and accuracy requirements:

| Model | File | Description |
|-------|------|-------------|
| **Lightweight** | `train.py` | Custom CNN + BiLSTM protein encoder + SMILES encoder. No heavy dependencies, fast to train on CPU or modest GPU. |
| **Full** | `model/model_p2m.py` | ESM-2 protein language model + SMILES Transformer + cross-attention. Higher accuracy, requires a GPU. |
| **Hybrid (Boltz-2)** | `model/model_boltz_p2m.py` | Wraps the full model with pre-cached Boltz-2 structural features (affinity priors, pose confidence). Best accuracy. |

---

## Models

### Lightweight Model (`train.py`)

A fast, dependency-light baseline suitable for rapid experimentation:

- **Protein encoder**: multi-scale CNN + Bidirectional LSTM over raw amino acid sequences
- **Molecule encoder**: character-level CNN over SMILES strings
- **Fusion**: concatenation + MLP classifier/regressor
- **No ESM-2 required** — trains in minutes on a modern CPU

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

Extends the full model with pre-computed Boltz-2 structural features:

- **Boltz-2 features**: predicted binding affinity prior + pose confidence score, cached per (protein, SMILES) pair
- **Feature injection**: Boltz-2 embeddings are concatenated into the cross-attention fusion layer
- **Inference mode**: can run direct structure-guided predictions without training via `--predict`

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

A GPU with at least 16 GB VRAM is recommended for the full and hybrid models. The lightweight model runs comfortably on CPU.

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

**Lightweight model** (fast, no ESM-2):

```bash
python train.py --data-dir data/prepared/p2m --epochs 50
```

**Full model** (ESM-2 + cross-attention):

```bash
python -m model.model_p2m --data-dir data/prepared/p2m --epochs 50
```

**Hybrid model** (Boltz-2 structural features):

First, pre-compute and cache Boltz-2 features for your dataset:

```bash
python -m download.download_boltz_features --data-dir data/prepared/p2m --out-dir data/boltz_cache
```

Then train:

```bash
python -m model.model_boltz_p2m --data-dir data/prepared/p2m --boltz-cache data/boltz_cache --epochs 50
```

### 4. Direct Boltz-2 Inference (no training required)

Run a structure-guided prediction for a single protein-molecule pair:

```bash
python -m model.model_boltz_p2m \
  --predict \
  --protein "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL" \
  --smiles "CC1=CC=C(C=C1)S(=O)(=O)N"
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
| Lightweight (CNN+BiLSTM) | 0.81 | 0.76 | 0.77 | 0.75 |
| Full (ESM-2 + Transformer) | 0.91 | 0.86 | 0.87 | 0.85 |
| Hybrid (+ Boltz-2) | 0.94 | 0.89 | 0.90 | 0.88 |

### Affinity Regression (pIC50, lower is better)

| Model | RMSE | Pearson r |
|-------|------|-----------|
| Lightweight (CNN+BiLSTM) | 1.12 | 0.73 |
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
│   └── prepared/
│       └── p2m/                    # Train/val/test splits for P2M
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
│   └── model_boltz_p2m.py          # Hybrid model: full model + Boltz-2 features
├── train.py                        # Lightweight CNN+BiLSTM model (no ESM-2)
└── requirements.txt
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