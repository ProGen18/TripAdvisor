"""Merge des sorties partielles (workers paralleles) en un parquet unique + agregation.

Lit tous les `sentiment_reviews*.parquet` dans data/stats/, concatene, deduplique
par review_id, sauve dans `sentiment_reviews.parquet` et regenere l'agregation
restaurant. Optionnellement supprime les fichiers _partN apres merge.

Usage :
    python scripts/sentiment_merge.py                # merge + agregation
    python scripts/sentiment_merge.py --cleanup      # + suppression _partN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import STATS_DIR  # noqa: E402
from scripts.sentiment_transformer import aggregate_by_restaurant, OUT_REVIEWS_PATH, OUT_RESTAURANT_PATH  # noqa: E402


def main(cleanup: bool = False):
    files = sorted(STATS_DIR.glob("sentiment_reviews*.parquet"))
    print(f"Fichiers trouves : {len(files)}")
    for f in files:
        n = len(pd.read_parquet(f, columns=["review_id"]))
        print(f"  {f.name}: {n:,} avis")

    if not files:
        print("Rien a merger.")
        return

    print("\nMerge...")
    dfs = [pd.read_parquet(f) for f in files]
    full = pd.concat(dfs, ignore_index=True)
    print(f"  Total avant dedup : {len(full):,}")
    full = full.drop_duplicates(subset=["review_id"], keep="first")
    print(f"  Total apres dedup : {len(full):,}")

    full.to_parquet(OUT_REVIEWS_PATH, index=False)
    print(f"  -> {OUT_REVIEWS_PATH.name}")

    print("\nAgregation par restaurant...")
    agg = aggregate_by_restaurant(full)
    agg.to_parquet(OUT_RESTAURANT_PATH, index=False)
    print(f"  -> {OUT_RESTAURANT_PATH.name} ({len(agg):,} restos)")

    if cleanup:
        print("\nNettoyage des _partN...")
        for f in files:
            if "_part" in f.name:
                f.unlink()
                print(f"  Supprime : {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge sentiment chunks")
    parser.add_argument("--cleanup", action="store_true", help="Supprime les _partN apres merge")
    args = parser.parse_args()
    main(cleanup=args.cleanup)
