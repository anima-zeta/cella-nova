#!/usr/bin/env python3
"""
Pre-compute Boltz-2 Structural Features for P2M Training Data
==============================================================

Batch pre-computes Boltz-2 structural features for all (protein, ligand)
pairs in the P2M training dataset so that HybridP2MModel can consume them
during training without running Boltz-2 live.

Each pair is identified by the MD5 hash of ``"{protein_seq}||{smiles}"``
and results are stored as JSON files in ``cache_dir``.  The cache-key logic
mirrors ``BoltzP2MPredictor._boltz_key()`` so the training dataloader and
this script always agree on where each result lives.

Usage::

    # Pre-compute features for all P2M training data
    python -m download.download_boltz_features

    # Test on first 50 samples using CPU
    python -m download.download_boltz_features --max-samples 50 --accelerator cpu

    # Resume interrupted run (skips already-cached pairs)
    python -m download.download_boltz_features --skip-existing
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import BoltzP2MPredictor with a graceful fallback for direct execution
# ---------------------------------------------------------------------------
try:
    from model.model_boltz_p2m import BoltzP2MPredictor
except ImportError:
    from model_boltz_p2m import BoltzP2MPredictor  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_p2m_data(data_dir: Path) -> List[Tuple[str, str, float, float]]:
    """
    Read protein-molecule interaction pairs from *data_dir*.

    Looks for files in this order:
      1. ``p2m_all.tsv``   — full prepared dataset (from prepare_all.py)
      2. ``p2m_train.tsv`` — training split only
      3. ``protein_molecule_interactions.tsv`` — raw download file (legacy)

    Expected TSV layout (tab-separated, first row is a header)::

        protein_id   protein_seq   smiles   label   affinity   [interaction_type]   source

    Returns
    -------
    list of ``(protein_seq, smiles, label, affinity_value)`` tuples.
    Rows with missing or unparseable values are skipped with a warning.
    """
    candidates = [
        data_dir / "p2m_all.tsv",
        data_dir / "p2m_train.tsv",
        data_dir / "protein_molecule_interactions.tsv",
    ]
    tsv_path = next((p for p in candidates if p.exists()), None)
    if tsv_path is None:
        log.error("No interaction TSV found in %s", data_dir)
        log.error(
            "Expected one of: p2m_all.tsv, p2m_train.tsv, "
            "protein_molecule_interactions.tsv"
        )
        log.error(
            "Run:  python -m prepare.prepare_all "
            "--data-dir data --output-dir data/prepared"
        )
        sys.exit(1)
    log.info("Loading pairs from %s", tsv_path.name)

    records: List[Tuple[str, str, float, float]] = []
    skipped = 0

    with open(tsv_path) as fh:
        raw_header = fh.readline().strip().split("\t")
        cols = {c.lower(): i for i, c in enumerate(raw_header)}
        log.info("TSV columns detected: %s", raw_header)

        # Resolve column indices by name with positional fallbacks so the
        # script works with both the raw download layout and the prepared layout.
        idx_protein  = cols.get("protein_seq", 1)
        idx_smiles   = cols.get("smiles", 2)
        idx_label    = cols.get("label", 3)
        idx_affinity = cols.get("affinity", 4)

        for lineno, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx_protein, idx_smiles, idx_label):
                log.warning(
                    "Line %d: too few columns (%d) — skipping",
                    lineno, len(parts),
                )
                skipped += 1
                continue
            try:
                protein_seq = parts[idx_protein].strip()
                smiles      = parts[idx_smiles].strip()
                label       = float(parts[idx_label])
                affinity    = (
                    float(parts[idx_affinity])
                    if idx_affinity < len(parts) and parts[idx_affinity].strip()
                    else 0.0
                )
            except (ValueError, IndexError) as exc:
                log.warning("Line %d: parse error (%s) — skipping", lineno, exc)
                skipped += 1
                continue

            if not protein_seq or not smiles:
                log.warning("Line %d: empty protein_seq or smiles — skipping", lineno)
                skipped += 1
                continue

            records.append((protein_seq, smiles, label, affinity))

    if skipped:
        log.warning("Skipped %d malformed rows while loading TSV", skipped)
    log.info("Loaded %d valid pairs from %s", len(records), tsv_path)
    return records


# ---------------------------------------------------------------------------
# Cache-key helper  (must mirror BoltzP2MPredictor._boltz_key exactly)
# ---------------------------------------------------------------------------

def compute_cache_key(protein_seq: str, smiles: str) -> str:
    """
    Return the MD5 hex digest of ``"{protein_seq}||{smiles}"``.

    This intentionally replicates the ``_boltz_key`` / ``_md5`` logic inside
    ``BoltzP2MPredictor`` so the download script and the training dataloader
    always agree on where each JSON cache file lives.
    """
    return hashlib.md5(f"{protein_seq}||{smiles}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-compute Boltz-2 structural features for P2M training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m download.download_boltz_features
  python -m download.download_boltz_features --max-samples 50 --accelerator cpu
  python -m download.download_boltz_features --skip-existing
        """,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/prepared/p2m"),
        help="Directory containing protein_molecule_interactions.tsv (default: data/prepared/p2m)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/boltz_cache"),
        help="Where to save Boltz-2 JSON cache files (default: data/boltz_cache)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Limit to the first N samples — useful for smoke-testing",
    )
    parser.add_argument(
        "--accelerator", choices=["gpu", "mps", "cpu"], default="cpu",
        help=(
            "Hardware accelerator for Boltz-2 predictions. "
            "gpu = NVIDIA CUDA, mps = Apple Silicon, cpu = CPU only. "
            "Default: cpu (safe on all hardware — use mps on Apple Silicon "
            "for ~3-5x speedup, gpu on Linux/Windows with NVIDIA GPU)."
        ),
    )
    parser.add_argument(
        "--use-msa-server", action=argparse.BooleanOptionalAction, default=True,
        help="Pass --use_msa_server to Boltz-2; required for MSA quality (default: True)",
    )
    parser.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="Skip pairs whose cache JSON already exists — enables resumable runs (default: True)",
    )
    parser.add_argument(
        "--max-protein-len", type=int, default=800,
        help="Skip proteins longer than this many residues; Boltz-2 is slow on very long chains (default: 800)",
    )
    parser.add_argument(
        "--max-smiles-len", type=int, default=150,
        help="Skip SMILES strings longer than this; affinity module has a 128 heavy-atom limit (default: 150)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="[Future] Number of parallel workers (currently unused, default: 1)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    args = _build_parser().parse_args()

    log.info("=" * 60)
    log.info("Cella Nova — Boltz-2 Feature Pre-computation")
    log.info("=" * 60)
    log.info("  data_dir        : %s", args.data_dir)
    log.info("  cache_dir       : %s", args.cache_dir)
    log.info("  accelerator     : %s", args.accelerator)
    log.info("  use_msa_server  : %s", args.use_msa_server)
    log.info("  skip_existing   : %s", args.skip_existing)
    log.info("  max_protein_len : %d", args.max_protein_len)
    log.info("  max_smiles_len  : %d", args.max_smiles_len)
    log.info("  max_samples     : %s", args.max_samples)

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    records = load_p2m_data(args.data_dir)
    if args.max_samples is not None:
        records = records[: args.max_samples]
        log.info("Capped to first %d samples (--max-samples)", len(records))
    total = len(records)

    # ------------------------------------------------------------------
    # Build predictor
    # ------------------------------------------------------------------
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    predictor = BoltzP2MPredictor(
        cache_dir=args.cache_dir,
        use_msa_server=args.use_msa_server,
        accelerator=args.accelerator,
    )

    # ------------------------------------------------------------------
    # Pre-filter: apply length limits and skip already-cached pairs
    # ------------------------------------------------------------------
    to_process: List[Tuple[str, str, float, float]] = []
    pre_skipped = 0  # count of pairs filtered out before the main loop

    for protein_seq, smiles, label, affinity in records:
        if len(protein_seq) > args.max_protein_len:
            pre_skipped += 1
            continue
        if len(smiles) > args.max_smiles_len:
            pre_skipped += 1
            continue
        if args.skip_existing:
            cache_file = args.cache_dir / f"{compute_cache_key(protein_seq, smiles)}.json"
            if cache_file.exists():
                pre_skipped += 1
                continue
        to_process.append((protein_seq, smiles, label, affinity))

    log.info(
        "Pairs to run: %d  |  pre-filtered / already cached: %d",
        len(to_process), pre_skipped,
    )

    # ------------------------------------------------------------------
    # Main loop — run Boltz-2 for each pair, update progress bar live
    # ------------------------------------------------------------------
    done = 0
    errors = 0

    with tqdm(to_process, unit="pair", dynamic_ncols=True) as progress:
        for protein_seq, smiles, label, affinity in progress:
            # Live status line: mirrors the requested format
            progress.set_description(
                f"{done}/{len(to_process)} pairs"
                f" | {pre_skipped} skipped"
                f" | {errors} errors"
            )
            try:
                predictor.predict(protein_seq, smiles)
                done += 1
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failed — protein[:40]=%r…  smiles[:40]=%r… : %s",
                    protein_seq[:40], smiles[:40], exc,
                )
                errors += 1

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    cached_total = done + pre_skipped          # everything now available in cache
    hit_rate = cached_total / total if total else 0.0

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total pairs in dataset          : {total:,}")
    print(f"  Newly computed & cached         : {done:,}")
    print(f"  Skipped (cached / filtered out) : {pre_skipped:,}")
    print(f"  Errors                          : {errors:,}")
    print(f"  Cache hit rate                  : {hit_rate * 100:.1f}%")
    print(f"  Cache directory                 : {args.cache_dir}")
    print()

    if errors:
        log.warning("%d pair(s) failed — re-run this script to retry them.", errors)
    else:
        log.info("All pairs processed successfully.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
