"""
Cella Nova Download Module

This module contains scripts for downloading biological data for
protein-small molecule (P2M) interaction prediction.

Scripts:
    download_pro.py:            Protein sequences from UniProt
    download_mol.py:            Small molecules and bioactivity data from ChEMBL
    build_p2m_interactions.py:  Build protein-molecule interaction pairs
    download_boltz_features.py: Pre-compute Boltz-2 structural features
"""

from pathlib import Path

# Module directory
MODULE_DIR = Path(__file__).parent

# Default data directory (relative to project root)
DEFAULT_DATA_DIR = MODULE_DIR.parent / "data"

__all__ = [
    "MODULE_DIR",
    "DEFAULT_DATA_DIR",
]
