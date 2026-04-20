#!/usr/bin/env python3
"""
Data Preparation Script for Cella Nova — Protein-Molecule (P2M) Interactions

This script prepares protein-molecule interaction data for training, including
splitting into train/val/test sets, generating negative samples, and filtering
by binding affinity threshold.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Determine project root directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_p2m_preparation(
    data_dir: Path,
    output_dir: Path,
    seed: int = 42,
    negative_ratio: float = 1.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    affinity_threshold: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Run Protein-Molecule Interaction data preparation."""
    try:
        try:
            from .prepare_p2m_data import PMolDataPreparer
        except ImportError:
            if str(SCRIPT_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPT_DIR))
            from prepare_p2m_data import PMolDataPreparer  # type: ignore

        logger.info("")
        logger.info("=" * 70)
        logger.info("PREPARING P2M (PROTEIN-MOLECULE INTERACTION) DATA")
        logger.info("=" * 70)

        preparer = PMolDataPreparer(
            data_dir=data_dir,
            output_dir=output_dir / "p2m",
            seed=seed,
            negative_ratio=negative_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            affinity_threshold=affinity_threshold,
        )

        return preparer.prepare()

    except ImportError as e:
        logger.error(f"Failed to import P2M preparer: {e}")
        return None
    except Exception as e:
        logger.error(f"P2M preparation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="P2M Data Preparation Script for Cella Nova",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare P2M data with defaults
  python prepare_all.py --data-dir data --output-dir data/prepared

  # Prepare with custom affinity threshold and seed
  python prepare_all.py --affinity-threshold 6.0 --seed 123

  # Prepare with different split ratios
  python prepare_all.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing raw downloaded data (default: {DEFAULT_DATA_DIR})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save prepared data (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive samples (default: 1.0)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation (default: 0.1)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for testing (default: 0.1)",
    )

    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=5.0,
        help="Minimum affinity (pIC50/pKd) for positive interactions (default: 5.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate and normalise split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.warning(f"Split ratios sum to {total_ratio:.4f}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " CELLA NOVA — P2M DATA PREPARATION PIPELINE ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("")
    logger.info(f"Data directory:    {args.data_dir}")
    logger.info(f"Output directory:  {args.output_dir}")
    logger.info(f"Affinity threshold: {args.affinity_threshold}")
    logger.info(f"Negative ratio:    {args.negative_ratio}")
    logger.info(
        f"Split ratios:      train={args.train_ratio:.2f}, "
        f"val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
    )
    logger.info(f"Random seed:       {args.seed}")

    result = run_p2m_preparation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        affinity_threshold=args.affinity_threshold,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("P2M PREPARATION RESULTS")
    logger.info("=" * 70)

    if result is not None:
        logger.info(f"  Status:       success")
        logger.info(f"  Interactions: {result.get('num_interactions', 0):,}")
        logger.info(f"  Proteins:     {result.get('num_proteins', 0):,}")
        logger.info(f"  Molecules:    {result.get('num_molecules', 0):,}")
        logger.info(
            f"  Train: {result.get('train_size', 0):,} | "
            f"Val: {result.get('val_size', 0):,} | "
            f"Test: {result.get('test_size', 0):,}"
        )
        logger.info(f"  Output:       {result.get('output_dir', args.output_dir / 'p2m')}")
        logger.info("=" * 70)
        logger.info("✓ P2M data preparation completed successfully!")
        sys.exit(0)
    else:
        logger.error("✗ P2M data preparation failed!")
        logger.info("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
