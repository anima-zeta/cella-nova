#!/usr/bin/env python3
"""
Pre-compute ESM-2 Protein Embeddings — Cella Nova
==================================================

Runs ESM-2 once over all unique proteins in the prepared dataset and saves
the token-level representations to a cache file.  The ProteinEncoder will
load this cache at training time and skip the ESM-2 forward pass entirely,
reducing per-batch cost from ~80 s to <1 s on MPS.

Cache format (saved as a .pt file):
    {md5_hex_of_sequence: tensor[L, esm_dim]}
    where L = min(len(seq), 1022) and esm_dim = 480 for esm2_t12_35M_UR50D.

Usage:
    python -m model.precompute_esm
    python -m model.precompute_esm --data-dir data/prepared/p2m --esm-model esm2_t12_35M_UR50D
    python -m model.precompute_esm --batch-size 4 --device cpu
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seq_md5(seq: str) -> str:
    """Hash a sequence exactly — must match ProteinEncoder._cache_key."""
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def load_unique_sequences(data_dir: Path) -> dict[str, str]:
    """
    Return {uniprot_id: sequence} for every unique protein in the dataset.
    Reads p2m_proteins.json produced by prepare_p2m_data.py.
    Falls back to scanning the TSV files directly when the JSON is absent.
    """
    result = {}

    # Always load from proteins.json first if available
    proteins_json = data_dir / "p2m_proteins.json"
    if proteins_json.exists():
        raw = json.loads(proteins_json.read_text())
        for pid, info in raw.items():
            seq = info.get("sequence", "") if isinstance(info, dict) else str(info)
            if seq:
                result[pid] = seq
        print(f"  Loaded {len(result):,} unique proteins from p2m_proteins.json")

    # Always also scan TSVs — catches sequences not present in the JSON
    tsv_added = 0
    for split in ("p2m_train.tsv", "p2m_val.tsv", "p2m_test.tsv"):
        tsv = data_dir / split
        if not tsv.exists():
            continue
        with open(tsv) as fh:
            header = fh.readline().strip().split("\t")
            cols = {c.lower(): i for i, c in enumerate(header)}
            idx_id  = cols.get("protein_id", 0)
            idx_seq = cols.get("protein_seq", 1)
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) <= max(idx_id, idx_seq):
                    continue
                pid = parts[idx_id]
                seq = parts[idx_seq]
                if pid and seq and pid not in result:
                    result[pid] = seq
                    tsv_added += 1
    if tsv_added:
        print(f"  Found {tsv_added} additional protein(s) in TSV files")
    print(f"  Total unique proteins: {len(result):,}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute ESM-2 embeddings for all unique proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m model.precompute_esm
    python -m model.precompute_esm --batch-size 8 --device cuda
    python -m model.precompute_esm --esm-model esm2_t33_650M_UR50D
        """,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/prepared/p2m"),
        help="Directory containing p2m_proteins.json (default: data/prepared/p2m)",
    )
    parser.add_argument(
        "--cache-path", type=Path, default=None,
        help="Output .pt file path (default: {data_dir}/esm_cache.pt)",
    )
    parser.add_argument(
        "--esm-model", type=str, default="esm2_t12_35M_UR50D",
        help="ESM-2 variant (must match the one used in model_p2m.py)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Proteins per ESM-2 batch (lower if OOM; default: 8)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="cuda | mps | cpu | auto (default: auto)",
    )
    parser.add_argument(
        "--max-len", type=int, default=512,
        help="Truncate sequences to this length — must match dataset max_protein_len (default: 512)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite an existing cache file",
    )
    args = parser.parse_args()

    # ---- device ----
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    cache_path = args.cache_path or (args.data_dir / "esm_cache.pt")

    print("=" * 60)
    print("Cella Nova — ESM-2 Embedding Pre-computation")
    print("=" * 60)
    print(f"  ESM model  : {args.esm_model}")
    print(f"  Device     : {device}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Max length : {args.max_len}")
    print(f"  Cache path : {cache_path}")
    print()

    # ---- check existing cache ----
    existing_cache: dict = {}
    if cache_path.exists() and not args.overwrite:
        print(f"Loading existing cache from {cache_path} …")
        existing_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  {len(existing_cache):,} embeddings already cached")

    # ---- load proteins ----
    if not args.data_dir.exists():
        print(f"❌  data-dir not found: {args.data_dir}")
        sys.exit(1)

    proteins = load_unique_sequences(args.data_dir)
    if not proteins:
        print("❌  No proteins found. Run prepare_all first.")
        sys.exit(1)

    # ---- filter to uncached ----
    to_compute = {
        pid: seq for pid, seq in proteins.items()
        if seq_md5(seq[:args.max_len]) not in existing_cache
    }
    print(f"  To compute : {len(to_compute):,}  (skipping {len(proteins) - len(to_compute):,} already cached)")

    if not to_compute:
        print("\n✓  All embeddings already cached. Nothing to do.")
        return

    # ---- load ESM-2 ----
    print(f"\nLoading {args.esm_model} …")
    import esm as esm_lib
    esm_model, alphabet = esm_lib.pretrained.load_model_and_alphabet(args.esm_model)
    batch_converter = alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()
    esm_dim = esm_model.embed_dim
    num_layers = esm_model.num_layers
    print(f"  ESM-2 loaded — embed_dim={esm_dim}, layers={num_layers}")

    # ---- compute embeddings ----
    cache: dict = dict(existing_cache)
    items = list(to_compute.items())
    errors = 0

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, len(items), args.batch_size),
            desc="Computing embeddings",
            unit="batch",
        ):
            batch_items = items[batch_start : batch_start + args.batch_size]

            # Truncate sequences to ESM-2 max length
            data = [
                (pid, seq[: args.max_len])
                for pid, seq in batch_items
            ]

            try:
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                results = esm_model(
                    batch_tokens,
                    repr_layers=[num_layers],
                    return_contacts=False,
                )
                token_repr = results["representations"][num_layers]  # [B, L+2, D]

                for i, (pid, seq) in enumerate(data):
                    seq_len = len(seq)
                    # Slice off BOS (index 0) and EOS (index seq_len+1)
                    embedding = token_repr[i, 1 : seq_len + 1].cpu()  # [L, D]
                    key = seq_md5(seq)
                    cache[key] = embedding

            except Exception as exc:
                tqdm.write(f"  ⚠ batch starting at {batch_start} failed: {exc}")
                errors += 1
                continue

    # ---- save ----
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total proteins   : {len(proteins):,}")
    print(f"  Newly computed   : {len(to_compute) - errors * args.batch_size:,}")
    print(f"  Errors (batches) : {errors}")
    print(f"  Cache entries    : {len(cache):,}")
    print(f"  Cache saved to   : {cache_path}")
    print()
    print("Next step:")
    print("  python -m model.model_p2m \\")
    print("    --data-dir data/prepared/p2m \\")
    print("    --esm-cache data/prepared/p2m/esm_cache.pt \\")
    print("    --epochs 50")


if __name__ == "__main__":
    main()
