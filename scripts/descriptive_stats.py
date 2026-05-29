"""Phase 3: Analyses descriptives et visualisations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.config import ENRICHED_DIR, OUTPUT_DIR


def plot_rating_distribution(df):
    """Distribution des notes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ratings = df["rating"].dropna()
    ax.hist(ratings, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(ratings.mean(), color="red", linestyle="--",
               label=f"Moy={ratings.mean():.2f}")
    ax.set_xlabel("Note")
    ax.set_ylabel("Nombre de restaurants")
    ax.set_title("Distribution des notes TripAdvisor")
    ax.legend()

    ax = axes[1]
    arr_ratings = df.groupby("arrondissement")["rating"].agg(["mean", "count"])
    arr_ratings = arr_ratings[arr_ratings["count"] >= 10].sort_values("mean")
    ax.barh(range(len(arr_ratings)), arr_ratings["mean"], color="steelblue")
    ax.set_yticks(range(len(arr_ratings)))
    ax.set_yticklabels([str(int(i)) + "e" for i in arr_ratings.index])
    ax.set_xlabel("Note moyenne")
    ax.set_title("Note moyenne par arrondissement (>=10 restos)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rating_distribution.png", dpi=150)
    plt.close(fig)


def plot_menu_analysis(df):
    """Distribution du nombre d'items et lien note-diversite."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    items = df[df["nb_items_total"] > 0]["nb_items_total"]
    ax.hist(items, bins=50, color="forestgreen", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Nombre d'items")
    ax.set_ylabel("Nombre de restaurants")
    ax.set_title("Distribution du nombre d'items par menu")

    ax = axes[0, 1]
    valid = df[(df["nb_items_total"] > 0) & df["rating"].notna()]
    ax.scatter(valid["nb_items_total"], valid["rating"], alpha=0.3, s=5, c="steelblue")
    ax.set_xlabel("Nombre d'items")
    ax.set_ylabel("Note")
    ax.set_title("Note contre Nombre d'items")

    ax = axes[1, 0]
    ax.scatter(valid["shannon_entropy"], valid["rating"], alpha=0.3, s=5, c="darkorange")
    ax.set_xlabel("Entropie de Shannon (E/P/D)")
    ax.set_ylabel("Note")
    ax.set_title("Note contre Diversite du menu")

    ax = axes[1, 1]
    cats = ["entrée", "plat", "dessert", "boisson", "formule", "autre"]
    cats_present = [c for c in cats if c in valid.columns]
    totals = [valid[c].sum() for c in cats_present]
    ax.bar(cats_present, totals,
           color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0", "#d9d9d9"])
    ax.set_ylabel("Nombre total d'items")
    ax.set_title("Distribution par categorie")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "menu_analysis.png", dpi=150)
    plt.close(fig)


def plot_competition_analysis(df):
    """Analyse lien concurrence-note."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    valid = df[df["nb_restos_500m"].notna() & df["rating"].notna()]

    ax = axes[0]
    ax.scatter(valid["nb_restos_500m"], valid["rating"], alpha=0.3, s=5)
    ax.set_xlabel("Nb restos dans 500m")
    ax.set_ylabel("Note")
    ax.set_title("Note contre Concurrence locale (500m)")

    ax = axes[1]
    ax.scatter(valid["nb_restos_1km"], valid["rating"], alpha=0.3, s=5, c="green")
    ax.set_xlabel("Nb restos dans 1km")
    ax.set_ylabel("Note")
    ax.set_title("Note contre Concurrence (1km)")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "competition_analysis.png", dpi=150)
    plt.close(fig)


def plot_seasonality(df):
    """Saisonnalite des menus."""
    seasons = ["printemps", "été", "automne", "hiver"]
    cols = [f"seasonal_{s}_ratio" for s in seasons]

    fig, ax = plt.subplots(figsize=(8, 6))
    cols_present = [c for c in cols if c in df.columns]
    valid = df[cols_present].dropna()
    if len(valid) > 0:
        means = [valid[c].mean() for c in cols_present]
        labels = [s for s, c in zip(seasons, cols) if c in df.columns]
        ax.bar(labels, means,
               color=["#90ee90", "#ff6347", "#daa520", "#87ceeb"])
        ax.set_ylabel("Ratio moyen d'items saisonniers")
        ax.set_title("Presence moyenne de produits de saison dans les menus")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "seasonality.png", dpi=150)
    plt.close(fig)


def print_stats(df):
    """Affiche les statistiques descriptives cles."""
    print("=" * 60)
    print("STATISTIQUES DESCRIPTIVES")
    print("=" * 60)

    print(f"\nNombre de restaurants: {len(df)}")
    print(f"Avec note: {df['rating'].notna().sum()}")
    print(f"Avec menu: {(df['nb_items_total'] > 0).sum()}")

    print(f"\nNote moyenne: {df['rating'].mean():.2f} (+/-{df['rating'].std():.2f})")
    print(f"Mediane: {df['rating'].median():.1f}")
    print(f"Min/Max: {df['rating'].min():.1f} / {df['rating'].max():.1f}")

    valid = df[df["nb_items_total"] > 0]
    print(f"\nItems moyen par menu: {valid['nb_items_total'].mean():.1f} "
          f"(+/-{valid['nb_items_total'].std():.1f})")
    print(f"Categories distinctes (E/P/D) moyen: {valid['nb_categories'].mean():.2f}")
    print(f"Entropie Shannon moyenne: {valid['shannon_entropy'].mean():.3f}")

    print(f"\n--- Correlations ---")
    print(f"Note ~ nb_items:     {valid['rating'].corr(valid['nb_items_total']):.3f}")
    print(f"Note ~ entropie:     {valid['rating'].corr(valid['shannon_entropy']):.3f}")
    if "nb_restos_500m" in valid.columns:
        print(f"Note ~ concurrence 500m: {valid['rating'].corr(valid['nb_restos_500m'].fillna(0)):.3f}")
    if "seasonality_score" in valid.columns:
        print(f"Note ~ saisonnalite: {valid['rating'].corr(valid['seasonality_score'].fillna(0)):.3f}")

    print(f"\n--- Top 5 arrondissements par note moyenne ---")
    arr_stats = df.groupby("arrondissement")["rating"].agg(["mean", "count"])
    arr_stats = arr_stats[arr_stats["count"] >= 10]
    print(arr_stats.sort_values("mean", ascending=False).head(5).to_string())

    print(f"\n--- Top 5 restos avec le plus d'items ---")
    top_items = df.nlargest(5, "nb_items_total")[
        ["name", "nb_items_total", "rating", "arrondissement"]
    ]
    print(top_items.to_string(index=False))


def main():
    print("Loading enriched data...")
    df = pd.read_parquet(ENRICHED_DIR / "restaurants_enriched.parquet")

    print_stats(df)

    print("\nGenerating plots...")
    plot_rating_distribution(df)
    print("  -> rating_distribution.png")
    plot_menu_analysis(df)
    print("  -> menu_analysis.png")
    plot_competition_analysis(df)
    print("  -> competition_analysis.png")
    plot_seasonality(df)
    print("  -> seasonality.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
