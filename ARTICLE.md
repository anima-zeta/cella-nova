# Cella Nova: Bridging Sequence-Based Learning and Structural Knowledge for Protein-Ligand Interaction Prediction

In the quest for new therapeutics, one of the most critical challenges in computational drug discovery is accurately predicting how a small molecule (a ligand) interacts with a target protein. While structural biology provides high-resolution insights, obtaining these structures experimentally is slow and expensive. On the other hand, sequence-based deep learning offers speed and scale but often misses the crucial 3D spatial context of the binding site.

**Cella Nova** was designed to bridge this gap. By combining the scale of protein language models with the structural precision of the Boltz-2 framework, we have developed a system that predicts not just *if* a molecule binds, but *how strongly* it binds and *what* the nature of that interaction is.

---

## The Problem: The Sequence-Structure Trade-off

Traditionally, interaction prediction falls into two camps:
1. **Sequence-based models**: These treat proteins and molecules as strings of characters. They are incredibly fast and can screen millions of compounds, but they struggle with "activity cliffs"—where a tiny change in the molecule's structure leads to a massive change in binding affinity.
2. **Structure-based models**: These use 3D coordinates of the protein-ligand complex. They are highly accurate because they model physical interactions (like hydrogen bonds and hydrophobic packing), but they require a known 3D structure, which is often unavailable for novel targets.

Cella Nova asks: *Can we train a fast sequence-based model to "think" like a structure-based model?*

---

## Model Architecture: The "Full" Model

The foundation of Cella Nova is a sophisticated multi-modal architecture designed to capture the nuances of both the protein and the ligand.

### 1. Protein Encoding (ESM-2)
We utilize **ESM-2**, a 650-million parameter protein language model. Rather than using global embeddings, we incorporate **binding-pocket attention**, allowing the model to focus on the specific residues likely to form the interaction site.

### 2. Molecule Encoding (SMILES Transformer)
Small molecules are processed through a dual-stage pipeline:
- **Multi-scale CNNs** capture local chemical environments.
- A **Transformer** with **pharmacophore attention** identifies the key functional groups responsible for binding.

### 3. Fusion and Prediction
The protein and molecule representations are fused via a **cross-attention mechanism**, simulating the "hand-in-glove" fit of a ligand in a pocket. This fused representation feeds into a **multi-task head** that simultaneously predicts:
- **Binding Probability**: A binary classification of whether the interaction occurs.
- **Affinity (pIC50)**: A regression to predict the strength of the bind.
- **Interaction Type**: Classification into roles like *inhibitor*, *activator*, or *substrate*.

---

## Pushing Boundaries: The Hybrid Boltz-2 Model

To move beyond the limitations of sequence-only learning, we introduced a **Hybrid Model** leveraging **Knowledge Distillation**.

We use **Boltz-2**, a state-of-the-art structural prediction model, as a "teacher." Boltz-2 can predict the 3D geometry and affinity of a complex from scratch. However, Boltz-2 is computationally expensive and too slow for large-scale screening.

**The Distillation Process:**
Instead of just training our student model on experimental data (which is often sparse), we train it to mimic the "soft labels" produced by Boltz-2. By balancing the loss between experimental ground truth and Boltz-2's predictions, the student model internalizes the structural "intuition" of the teacher.

The result is a model that maintains the **inference speed of a sequence-based network** but achieves the **accuracy of a structure-guided system**.

---

## Results: Does it Work?

The impact of the hybrid approach is evident in the performance metrics. When tested on held-out human target data from ChEMBL, the Hybrid model consistently outperformed the Full sequence-based model.

### Binding Classification (AUC-ROC)
| Model | AUC-ROC |
|-------|---------|
| Full Model | 0.91 |
| **Hybrid Model** | **0.94** |

### Affinity Regression (pIC50)
| Model | RMSE (Lower is better) | Pearson r (Higher is better) |
|-------|-------------------------|------------------------------|
| Full Model | 0.81 | 0.87 |
| **Hybrid Model** | **0.68** | **0.91** |

The reduction in RMSE (from 0.81 to 0.68) represents a significant leap in the precision of affinity predictions, moving us closer to a reliable "digital assay."

---

## Conclusion

Cella Nova demonstrates that the dichotomy between sequence-based and structure-based modeling is a false one. By using knowledge distillation to transfer structural insights from a heavy teacher model (Boltz-2) into a lightweight student, we can achieve high-fidelity interaction predictions at a fraction of the computational cost.

This approach paves the way for more efficient virtual screening, allowing researchers to narrow down millions of potential drug candidates to a handful of high-probability leads with confidence, accelerating the journey from computer screen to clinic.