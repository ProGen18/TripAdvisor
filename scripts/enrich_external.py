"""Phase 2: Enrichissement - arrondissement, saisonnalite, concurrence locale."""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm
from scripts.config import (
    PROCESSED_DIR, ENRICHED_DIR, SEASONAL_PRODUCTS,
)


def extract_arrondissement(neighborhoods_str):
    """Extrait le numero d'arrondissement depuis les quartiers.
    Ex: '7e Arr. - Palais Bourbon|Ecole Militaire' -> 7
    """
    if not neighborhoods_str or not isinstance(neighborhoods_str, str):
        return None
    m = re.search(r'(\d{1,2})\s*(?:er|e|ème|eme)\s*Arr', neighborhoods_str)
    return int(m.group(1)) if m else None


def compute_competition(df):
    """Calcule le nombre de restaurants concurrents a 500m et 1km."""
    result = df[["locationId"]].copy()
    result["nb_restos_500m"] = np.nan
    result["nb_restos_1km"] = np.nan

    # Restos avec coordonnees valides
    valid = df["latitude"].notna() & df["longitude"].notna()
    coords = df.loc[valid, ["latitude", "longitude"]].values

    if len(coords) == 0:
        return result

    lats = np.radians(coords[:, 0])
    lons = np.radians(coords[:, 1])

    # Conversion lat/lon -> x,y,z sur sphere unite
    xs = np.cos(lats) * np.cos(lons)
    ys = np.cos(lats) * np.sin(lons)
    zs = np.sin(lats)
    xyz = np.column_stack([xs, ys, zs])

    tree = cKDTree(xyz)

    # Rayons en distance corde correspondant a 500m et 1km
    R = 6371.0
    chord_500 = 2.0 * np.sin(0.5 / R)
    chord_1000 = 2.0 * np.sin(1.0 / R)

    counts_500 = tree.query_ball_point(xyz, chord_500, return_length=True)
    counts_1000 = tree.query_ball_point(xyz, chord_1000, return_length=True)

    result.loc[valid, "nb_restos_500m"] = [int(c) - 1 for c in counts_500]
    result.loc[valid, "nb_restos_1km"] = [int(c) - 1 for c in counts_1000]
    return result


def compute_seasonality(items_df):
    """Calcule un score de saisonnalite par restaurant base sur les items."""
    items = items_df.copy()
    items["text"] = (items["item_title"].fillna("").str.lower()
                     + " " + items["item_description"].fillna("").str.lower())

    results = []
    for lid, group in tqdm(items.groupby("locationId"), desc="Seasonality"):
        all_text = " ".join(group["text"])
        total = max(len(group), 1)
        scores = {}
        for season, products in SEASONAL_PRODUCTS.items():
            count = sum(1 for p in products if p in all_text)
            scores[f"seasonal_{season}"] = int(count)
            scores[f"seasonal_{season}_ratio"] = count / total
        scores["locationId"] = lid
        scores["seasonal_items_total"] = int(sum(
            scores[f"seasonal_{s}"] for s in SEASONAL_PRODUCTS
        ))
        scores["seasonality_score"] = float(max(
            scores[f"seasonal_{s}_ratio"] for s in SEASONAL_PRODUCTS
        ))
        results.append(scores)

    return pd.DataFrame(results)


def main():
    # Chargement
    print("Chargement des donnees...")
    restos = pd.read_parquet(PROCESSED_DIR / "restaurants.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "menu_items.parquet")
    diversity = pd.read_parquet(PROCESSED_DIR / "menu_diversity.parquet")

    # 1. Arrondissement
    print("Extraction des arrondissements...")
    restos["arrondissement"] = restos["neighborhoods"].apply(extract_arrondissement)
    resolved = restos["arrondissement"].notna().sum()
    print(f"  Resolus : {resolved}/{len(restos)} ({100*resolved/len(restos):.1f}%)")

    # 2. Concurrence locale
    print("Calcul de la concurrence locale...")
    competition = compute_competition(restos)

    # 3. Saisonnalite
    print("Calcul de la saisonnalite...")
    seasonality = compute_seasonality(items)

    # 4. Fusion
    print("Fusion...")
    enriched = restos.merge(diversity, on="locationId", how="left")
    enriched = enriched.merge(seasonality, on="locationId", how="left")
    enriched = enriched.merge(competition, on="locationId", how="left")

    # Fill NA pour les restos sans menu
    for col in ["nb_items_total", "nb_categories", "shannon_entropy", "type_token_ratio"]:
        if col in enriched.columns:
            enriched[col] = enriched[col].fillna(0)

    output = ENRICHED_DIR / "restaurants_enriched.parquet"
    enriched.to_parquet(output, index=False)

    print(f"\nRestaurants enrichis : {len(enriched)}")
    print(f"Colonnes : {list(enriched.columns)}")
    print(f"\nDistribution par arrondissement :")
    arr_dist = enriched["arrondissement"].value_counts().sort_index()
    for arr, cnt in arr_dist.items():
        print(f"  {int(arr)}e : {cnt}")

    print(f"\nConcurrence moyenne : {enriched['nb_restos_500m'].mean():.1f} (500m), {enriched['nb_restos_1km'].mean():.1f} (1km)")
    print(f"Saisonnalite moyenne : {enriched['seasonality_score'].mean():.4f}")
    print(f"\nSauvegarde dans {output}")


if __name__ == "__main__":
    main()
