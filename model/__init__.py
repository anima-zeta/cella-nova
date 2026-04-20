"""
Cella Nova Model Package

This package contains neural network models for predicting
protein-small molecule (P2M) interactions and binding affinities.

Models:
    model_p2m:       Full model — ESM-2 + SMILES Transformer + cross-attention
    model_boltz_p2m: Hybrid model — wraps model_p2m with Boltz-2 structural features
"""

from pathlib import Path

# Package metadata
__version__ = "0.2.0"
__author__ = "Cella Nova Team"

# Package directory
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Model weights directory
WEIGHTS_DIR = PACKAGE_DIR

# Default paths for saved models
DEFAULT_MODEL_PATHS = {
    "p2m": WEIGHTS_DIR / "p2m_model.pt",
    "p2m_hybrid": WEIGHTS_DIR / "p2m_hybrid_model.pt",
}


def get_model_path(model_type: str) -> Path:
    """Get the default path for a model type."""
    if model_type not in DEFAULT_MODEL_PATHS:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Available types: {list(DEFAULT_MODEL_PATHS.keys())}"
        )
    return DEFAULT_MODEL_PATHS[model_type]


# Lazy imports to avoid loading heavy dependencies until needed
def __getattr__(name):
    if name == "ProteinMoleculeModel":
        from .model_p2m import ProteinMoleculeModel
        return ProteinMoleculeModel
    if name == "HybridP2MModel":
        from .model_boltz_p2m import HybridP2MModel
        return HybridP2MModel
    if name == "BoltzP2MPredictor":
        from .model_boltz_p2m import BoltzP2MPredictor
        return BoltzP2MPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProteinMoleculeModel",
    "HybridP2MModel",
    "BoltzP2MPredictor",
    "get_model_path",
    "WEIGHTS_DIR",
    "PROJECT_ROOT",
]
