"""Extrait les avis depuis les fichiers JSON locaux (/data/paris/*.json).

Structure JSON attendue :
    data.locations[0].reviewListPage.reviews[]
        - text, rating, publishedDate, language, title, id

Produit data/raw/all_reviews.parquet (dans le worktree).

Usage:
    python scripts/extract_local_reviews.py
    python scripts/extract_local_reviews.py --max-restos 200 --stratify
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import DATA_DIR

# Les JSON originaux sont dans le repo principal
MAIN_REPO = PROJECT_ROOT.parents[2]
PARIS_DATA_DIR = MAIN_REPO / "data" / "paris"

RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def extract_all_reviews(max_restos: int | None = None,
                        stratify: bool = False,
                        min_text_length: int = 20) -> pd.DataFrame:
    """Extrait tous les avis des fichiers JSON locaux.

    Args:
        max_restos: Nombre max de restaurants a traiter
        stratify: Si True, echantillonnage stratifie par arrondissement (via enriched)
        min_text_length: Longueur minimale du texte pour etre garde

    Returns:
        DataFrame avec colonnes : locationId, name, text, rating, publishedDate,
                                  language, title, review_id
    """
    json_files = sorted(PARIS_DATA_DIR.glob("*.json"))

    if max_restos and not stratify:
        json_files = json_files[:max_restos]

    reviews = []
    n_files = 0
    n_skipped = 0

    for fp in json_files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)

            locations = data.get("data", {}).get("locations", [])
            if not locations:
                n_skipped += 1
                continue

            loc = locations[0]
            location_id = loc.get("locationId")
            name = loc.get("name", "")

            review_list = loc.get("reviewListPage", {}).get("reviews", [])
            for rev in review_list:
                text = rev.get("text", "")
                if not text or not isinstance(text, str) or len(text.strip()) < min_text_length:
                    continue

                reviews.append({
                    "locationId": int(location_id),
                    "name": str(name),
                    "text": text.strip(),
                    "rating": int(rev["rating"]) if rev.get("rating") else None,
                    "publishedDate": rev.get("publishedDate", ""),
                    "language": rev.get("language", "unknown"),
                    "title": rev.get("title", ""),
                    "review_id": rev.get("id"),
                })
            n_files += 1
        except Exception:
            n_skipped += 1
            continue

    df = pd.DataFrame(reviews)

    if df.empty:
        print("Aucun avis extrait.")
        return df

    # Conversion des dates
    df["publishedDate"] = pd.to_datetime(df["publishedDate"], errors="coerce")

    # Stratification si demandee
    if stratify and max_restos:
        enriched_path = PROJECT_ROOT / "data" / "enriched" / "restaurants_enriched.parquet"
        if enriched_path.exists():
            enriched = pd.read_parquet(enriched_path)
            enriched_ids = set(enriched["locationId"].astype(int).tolist())
            df = df[df["locationId"].isin(enriched_ids)]

            if len(df["locationId"].unique()) > max_restos:
                # Stratified by arrondissement
                df_with_arr = df.merge(
                    enriched[["locationId", "arrondissement", "priceLevel", "rating"]],
                    on="locationId", how="left"
                )
                df_with_arr["rating_quartile"] = pd.qcut(
                    df_with_arr["rating"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
                )

                selected_ids = set()
                for arr in df_with_arr["arrondissement"].dropna().unique():
                    arr_df = df_with_arr[df_with_arr["arrondissement"] == arr]
                    n_per_arr = max(1, int(max_restos * len(arr_df) / len(df_with_arr)))
                    arr_ids = arr_df["locationId"].unique()
                    if len(arr_ids) > n_per_arr:
                        arr_ids = np.random.choice(arr_ids, n_per_arr, replace=False)
                    selected_ids.update(arr_ids)

                # Complete si pas assez
                remaining = max_restos - len(selected_ids)
                if remaining > 0:
                    extra = df_with_arr[~df_with_arr["locationId"].isin(selected_ids)]
                    extra_ids = extra["locationId"].unique()
                    if len(extra_ids) > remaining:
                        extra_ids = np.random.choice(extra_ids, remaining, replace=False)
                    selected_ids.update(extra_ids)

                df = df[df["locationId"].isin(selected_ids)]

    print(f"Fichiers traites : {n_files}")
    print(f"Fichiers ignores : {n_skipped}")
    print(f"Avis extraits : {len(df)}")
    print(f"Restaurants uniques : {df['locationId'].nunique()}")

    # Statistiques par langue
    if "language" in df.columns:
        lang_counts = df["language"].value_counts()
        print(f"Langues : {dict(lang_counts)}")

    # Plage de dates
    if "publishedDate" in df.columns and df["publishedDate"].notna().any():
        print(f"Dates : {df['publishedDate'].min().date()} -> {df['publishedDate'].max().date()}")

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extraire les avis depuis les JSON locaux")
    parser.add_argument("--max-restos", type=int, default=None,
                        help="Nombre max de restaurants")
    parser.add_argument("--stratify", action="store_true",
                        help="Echantillonnage stratifie")
    parser.add_argument("--output", type=str, default="all_reviews.parquet")
    args = parser.parse_args()

    print("=" * 60)
    print("EXTRACTION DES AVIS — JSON LOCAUX")
    print("=" * 60)
    print(f"Source : {PARIS_DATA_DIR}")
    print(f"Fichiers JSON trouves : {len(list(PARIS_DATA_DIR.glob('*.json')))}")

    df = extract_all_reviews(
        max_restos=args.max_restos,
        stratify=args.stratify,
    )

    if df.empty:
        print("Aucun avis extrait.")
        return

    out_path = RAW_DIR / args.output
    df.to_parquet(out_path, index=False)
    print(f"\nSauvegarde dans {out_path}")
    print(f"Taille : {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
