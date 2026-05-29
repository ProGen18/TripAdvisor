"""Data loader — charge toutes les donnees du pipeline pour le dashboard UI.

Usage dans Streamlit:
    from scripts.data_loader import load_all_data
    data = load_all_data()
    # data est un dict avec toutes les DataFrames pretes a l'emploi
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ENRICHED_DIR = PROJECT_ROOT / "data" / "enriched"
STATS_DIR = PROJECT_ROOT / "data" / "stats"


def _encode_price_level(series):
    """Map price-level strings (e.g. '€', '€€-€€€') to an integer level (1=cheapest)."""

    def _count_euros(val):
        if pd.isna(val):
            return np.nan
        return str(val).count("€")

    return series.apply(_count_euros)


def load_all_data():
    """Charge toutes les donnees en un seul appel. Retourne un dict."""
    data = {}

    # --- Donnees principales ---
    data["restaurants"] = pd.read_parquet(PROCESSED_DIR / "restaurants.parquet")
    data["menu_items"] = pd.read_parquet(PROCESSED_DIR / "menu_items.parquet")
    data["menu_diversity"] = pd.read_parquet(PROCESSED_DIR / "menu_diversity.parquet")
    data["enriched"] = pd.read_parquet(ENRICHED_DIR / "restaurants_enriched.parquet")

    # --- Derived columns (same logic as advanced_stats.py) ---
    enriched = data["enriched"]

    if "price_level_num" not in enriched.columns and "priceLevel" in enriched.columns:
        enriched["price_level_num"] = _encode_price_level(enriched["priceLevel"])

    if "cuisine_list" not in enriched.columns and "cuisines" in enriched.columns:
        enriched["cuisine_list"] = (
            enriched["cuisines"]
            .fillna("")
            .apply(lambda s: [c.strip() for c in str(s).split("|") if c.strip()])
        )

    # --- Statistiques pre-calculees ---
    data["descriptive"] = pd.read_parquet(STATS_DIR / "descriptive.parquet")
    data["rankings"] = pd.read_parquet(STATS_DIR / "rankings.parquet")
    data["group_stats"] = pd.read_parquet(STATS_DIR / "group_stats.parquet")
    data["price_analysis"] = pd.read_parquet(STATS_DIR / "price_analysis.parquet")
    data["distributions"] = pd.read_parquet(STATS_DIR / "distributions.parquet")
    data["spatial_clusters"] = pd.read_parquet(STATS_DIR / "spatial_clusters.parquet")
    data["spatial_cluster_centers"] = pd.read_parquet(
        STATS_DIR / "spatial_cluster_centers.parquet"
    )
    data["outliers"] = pd.read_parquet(STATS_DIR / "outliers.parquet")
    data["cuisine_analysis"] = pd.read_parquet(STATS_DIR / "cuisine_analysis.parquet")
    data["cuisine_prices"] = pd.read_parquet(STATS_DIR / "cuisine_prices.parquet")
    data["cuisine_cooccurrence"] = pd.read_parquet(
        STATS_DIR / "cuisine_cooccurrence.parquet"
    )
    data["michelin_analysis"] = pd.read_parquet(STATS_DIR / "michelin_analysis.parquet")
    data["michelin_coordinates"] = pd.read_parquet(
        STATS_DIR / "michelin_coordinates.parquet"
    )
    data["seasonality_detail"] = pd.read_parquet(
        STATS_DIR / "seasonality_detail.parquet"
    )

    # --- Fichiers JSON ---
    with open(STATS_DIR / "correlations.json", encoding="utf-8") as f:
        data["correlations"] = json.load(f)
    with open(STATS_DIR / "regressions.json", encoding="utf-8") as f:
        data["regressions"] = json.load(f)

    # --- Nouveaux artefacts (Sprint 1) ---
    narrative_path = STATS_DIR / "narrative.json"
    if narrative_path.exists():
        with open(narrative_path, encoding="utf-8") as f:
            data["narrative"] = json.load(f)
    else:
        data["narrative"] = {}

    # --- Sentiment lexical (legacy, garde pour fallback) ---
    sentiment_path = STATS_DIR / "sentiment_restaurant.parquet"
    if sentiment_path.exists():
        data["sentiment"] = pd.read_parquet(sentiment_path)
    else:
        data["sentiment"] = pd.DataFrame()

    sentiment_kw_path = STATS_DIR / "sentiment_keywords.parquet"
    if sentiment_kw_path.exists():
        data["sentiment_keywords"] = pd.read_parquet(sentiment_kw_path)
    else:
        data["sentiment_keywords"] = pd.DataFrame()

    # --- Sentiment transformer (nlptown multilingue 1-5 stars) ---
    sent_v2_path = STATS_DIR / "sentiment_restaurant_v2.parquet"
    if sent_v2_path.exists():
        data["sentiment_v2"] = pd.read_parquet(sent_v2_path)
    else:
        data["sentiment_v2"] = pd.DataFrame()

    # Lit le parquet principal + tous les _partN (run parallele en cours)
    part_files = list(STATS_DIR.glob("sentiment_reviews*.parquet"))
    if part_files:
        dfs = []
        for p in part_files:
            try:
                dfs.append(pd.read_parquet(p))
            except Exception:
                pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=["review_id"], keep="first")

            # Enrichissement dynamique avec les colonnes d'ironie
            if (
                "is_ironic" not in combined.columns
                or combined["is_ironic"].isna().any()
            ):
                combined = combined.copy()
                if "rating" in combined.columns:
                    is_ironic_mask = (combined["sent_class"] >= 4) & (
                        combined["rating"].fillna(99.0) <= 2
                    )
                    combined["is_ironic"] = is_ironic_mask
                    combined["sent_stars_corrected"] = np.where(
                        is_ironic_mask,
                        combined["rating"],
                        combined["sent_stars_expected"],
                    )
                    combined["sent_class_corrected"] = np.where(
                        is_ironic_mask, combined["rating"], combined["sent_class"]
                    )
                    combined["sent_class_corrected"] = combined[
                        "sent_class_corrected"
                    ].astype("Int64")
                else:
                    combined["is_ironic"] = False
                    combined["sent_stars_corrected"] = combined["sent_stars_expected"]
                    combined["sent_class_corrected"] = combined["sent_class"].astype(
                        "Int64"
                    )

            data["sentiment_reviews"] = combined
        else:
            data["sentiment_reviews"] = pd.DataFrame()
    else:
        data["sentiment_reviews"] = pd.DataFrame()

    # Enrichissement dynamique de sentiment_v2 (statistiques agrégées) si vide ou si les colonnes d'ironie manquent
    if "sentiment_v2" in data:
        df_v2 = data["sentiment_v2"]
        if (df_v2.empty or "sent_mean_corrected" not in df_v2.columns) and not data[
            "sentiment_reviews"
        ].empty:
            data["sentiment_v2"] = aggregate_sentiment_reviews(
                data["sentiment_reviews"]
            )

    return data


def get_available_filters(enriched_df):
    """Retourne les valeurs possibles pour les filtres UI."""
    filters = {}

    # Arrondissements (tries)
    arrs = enriched_df["arrondissement"].dropna().unique()
    filters["arrondissements"] = sorted([int(a) for a in arrs if not pd.isna(a)])

    # Niveaux de prix
    prices = enriched_df["priceLevel"].dropna().unique()
    filters["price_levels"] = sorted([p for p in prices if isinstance(p, str)])

    # Cuisines (top 30)
    all_cuisines = []
    for c in enriched_df["cuisines"].dropna():
        if isinstance(c, str):
            all_cuisines.extend(c.split("|"))
    cuisine_counts = pd.Series(all_cuisines).value_counts()
    filters["top_cuisines"] = cuisine_counts.head(30).index.tolist()

    # Intervalle de notes
    filters["rating_min"] = float(enriched_df["rating"].min())
    filters["rating_max"] = float(enriched_df["rating"].max())

    # Intervalle de prix
    menu_items = pd.read_parquet(PROCESSED_DIR / "menu_items.parquet")
    prices_valid = menu_items["item_price"].dropna()
    filters["price_min"] = float(prices_valid.quantile(0.01))
    filters["price_max"] = float(prices_valid.quantile(0.99))

    # Nombre d'avis
    filters["review_count_max"] = int(enriched_df["reviewCount"].max())

    # Concurrence locale
    filters["concurrence_max"] = int(enriched_df["nb_restos_500m"].max())

    # Categories de menu disponibles
    menu_cats = ["entree", "plat", "dessert", "boisson"]
    filters["menu_categories"] = menu_cats

    return filters


def aggregate_sentiment_reviews(scored_reviews: pd.DataFrame) -> pd.DataFrame:
    """Agrège de manière performante et vectorisée les avis par restaurant.

    Cette fonction s'exécute sans importer PyTorch/Transformers et est optimisée
    pour Pandas en évitant les lambdas Python lents.
    """
    if scored_reviews.empty:
        return pd.DataFrame(
            columns=[
                "locationId",
                "n_reviews_scored",
                "sent_mean",
                "sent_std",
                "sent_median",
                "pct_negative",
                "pct_positive",
                "confidence_mean",
                "sent_mean_corrected",
                "sent_std_corrected",
                "sent_median_corrected",
                "pct_negative_corrected",
                "pct_positive_corrected",
                "pct_ironic",
                "discordance",
                "discordance_corrected",
            ]
        )

    df = scored_reviews.copy()

    # Détection de l'ironie (sans propagation de NaN)
    df["is_ironic"] = (df["sent_class"] >= 4) & (df["rating"] <= 2)
    df["is_ironic"] = df["is_ironic"].fillna(False).astype(bool)

    # Note et classe corrigées
    df["sent_stars_corrected"] = np.where(
        df["is_ironic"], df["rating"], df["sent_stars_expected"]
    )
    df["sent_class_corrected"] = np.where(
        df["is_ironic"], df["rating"], df["sent_class"]
    )
    df["sent_class_corrected"] = df["sent_class_corrected"].astype(
        "Int64"
    )  # Nullable Int

    # Indicateurs rapides pour agrégation moyenne optimisée
    df["is_neg"] = df["sent_class"] <= 2
    df["is_pos"] = df["sent_class"] >= 4
    df["is_neg_corr"] = df["sent_class_corrected"] <= 2
    df["is_pos_corr"] = df["sent_class_corrected"] >= 4
    df["is_ironic_int"] = df["is_ironic"].astype(int)

    # Agrégation par restaurant
    g = (
        df.groupby("locationId")
        .agg(
            n_reviews_scored=("review_id", "count"),
            sent_mean=("sent_stars_expected", "mean"),
            sent_std=("sent_stars_expected", "std"),
            sent_median=("sent_stars_expected", "median"),
            pct_negative=("is_neg", "mean"),
            pct_positive=("is_pos", "mean"),
            confidence_mean=("sent_confidence", "mean"),
            # Métriques corrigées
            sent_mean_corrected=("sent_stars_corrected", "mean"),
            sent_std_corrected=("sent_stars_corrected", "std"),
            sent_median_corrected=("sent_stars_corrected", "median"),
            pct_negative_corrected=("is_neg_corr", "mean"),
            pct_positive_corrected=("is_pos_corr", "mean"),
            pct_ironic=("is_ironic_int", "mean"),
        )
        .reset_index()
    )

    g["pct_negative"] *= 100
    g["pct_positive"] *= 100
    g["pct_negative_corrected"] *= 100
    g["pct_positive_corrected"] *= 100
    g["pct_ironic"] *= 100

    # Discordance vs note moyenne
    if "rating" in df.columns:
        rating_mean = df.groupby("locationId")["rating"].mean()
        g = g.merge(
            rating_mean.rename("rating_mean_reviews"), on="locationId", how="left"
        )
        g["discordance"] = g["sent_mean"] - g["rating_mean_reviews"]
        g["discordance_corrected"] = g["sent_mean_corrected"] - g["rating_mean_reviews"]
    else:
        g["rating_mean_reviews"] = np.nan
        g["discordance"] = np.nan
        g["discordance_corrected"] = np.nan

    return g


if __name__ == "__main__":
    print("Test du chargeur de donnees...")
    data = load_all_data()
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"  {name}: {df.shape[0]} lignes x {df.shape[1]} colonnes")
        elif isinstance(df, dict):
            print(f"  {name}: dict avec cles {list(df.keys())[:5]}...")
    print("Termine.")
