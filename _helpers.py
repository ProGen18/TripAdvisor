"""Helpers partages par tous les modules d'onglets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def ensure_data_loaded():
    """Charge les donnees dans st.session_state si elles n'y sont pas deja.

    A appeler en haut de chaque page multi-page Streamlit pour permettre
    le rechargement direct d'une sous-page (refresh ou URL directe) sans
    devoir d'abord passer par app_v2_pages.py.
    """
    if "data" in st.session_state and "filtered" in st.session_state:
        return

    from scripts.data_loader import load_all_data, get_available_filters

    @st.cache_data(show_spinner="Chargement des donnees...", ttl=3600)
    def _cached():
        d = load_all_data()
        fm = get_available_filters(d["enriched"])
        return d, fm

    data, filters_meta = _cached()
    st.session_state["data"] = data
    st.session_state["filters_meta"] = filters_meta
    if "filtered" not in st.session_state:
        st.session_state["filtered"] = data["enriched"]
    if "filters" not in st.session_state:
        st.session_state["filters"] = {}

# Palette coherente
COLOR_SEQ = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

PARIS_CENTER = {"lat": 48.8566, "lon": 2.3522}


def format_arr(n) -> str:
    if pd.isna(n):
        return "Inconnu"
    try:
        return f"Arr {int(n)}"
    except (TypeError, ValueError):
        return str(n)


@st.cache_data(show_spinner=False)
def filter_menu_items(_data_id: str, location_ids: tuple) -> pd.DataFrame:
    raise RuntimeError("Utiliser load_menu_items_filtered(data, filtered) a la place")


def load_menu_items_filtered(data: dict, filtered: pd.DataFrame) -> pd.DataFrame:
    """Filtre menu_items pour ne garder que les restaurants presents dans filtered."""
    if filtered.empty:
        return data["menu_items"].iloc[0:0]
    ids = set(filtered["locationId"].astype(int).tolist())
    mi = data["menu_items"]
    return mi[mi["locationId"].isin(ids)]


def explode_cuisines(df: pd.DataFrame, top_n: int | None = None) -> pd.Series:
    """Renvoie une Series des cuisines individuelles (apres split sur |)."""
    cuisines = []
    for c in df["cuisines"].dropna():
        if isinstance(c, str):
            cuisines.extend([x.strip() for x in c.split("|") if x.strip()])
    s = pd.Series(cuisines).value_counts()
    if top_n:
        s = s.head(top_n)
    return s


def empty_warning(filtered: pd.DataFrame) -> bool:
    if filtered.empty:
        st.warning("Aucun restaurant ne correspond aux filtres. Elargis la selection dans la sidebar.")
        return True
    return False


# ---------------------------------------------------------------------------
# KPI dynamiques
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, global_df: pd.DataFrame | None = None) -> dict:
    """Calcule les KPI dynamiques sur la DataFrame filtree."""
    if global_df is None:
        global_df = df

    kpis = {}

    # Ensure price_level_num exists (compute from priceLevel if missing)
    if "price_level_num" not in df.columns and "priceLevel" in df.columns:
        def _count_euros(val):
            if pd.isna(val):
                return np.nan
            return str(val).count("€")
        df = df.copy()
        df["price_level_num"] = df["priceLevel"].apply(_count_euros)

    if all(c in df.columns for c in ["rating", "price_level_num"]):
        ratio_df = df[df["price_level_num"] > 0].copy()
        if len(ratio_df) > 0:
            rating_norm = (ratio_df["rating"] - ratio_df["rating"].min()) / (ratio_df["rating"].max() - ratio_df["rating"].min() + 0.01)
            price_norm = ratio_df["price_level_num"] / ratio_df["price_level_num"].max()
            ratio_df["qp"] = rating_norm / price_norm
            best_idx = ratio_df["qp"].idxmax()
            best_name = ratio_df.loc[best_idx].get("name", f"Loc #{int(ratio_df.loc[best_idx, 'locationId'])}")
            kpis["best_qp"] = {
                "label": "Meilleur rapport qualite/prix",
                "value": str(best_name)[:30],
                "delta": None,
            }

    if "rating" in df.columns:
        avg_rating = df["rating"].mean()
        global_avg = global_df["rating"].mean()
        kpis["avg_rating"] = {
            "label": "Note moyenne",
            "value": f"{avg_rating:.2f}",
            "delta": round(float(avg_rating - global_avg), 2),
        }

    if "cuisine_list" in df.columns:
        cuisine_std = df.explode("cuisine_list").groupby("cuisine_list")["rating"].std()
        if len(cuisine_std) >= 2:
            most_pol = cuisine_std.idxmax()
            kpis["polarizing_cuisine"] = {
                "label": "Cuisine la plus polarisante",
                "value": str(most_pol),
                "delta": f"etendue {cuisine_std.max() - cuisine_std.min():.2f}",
            }

    if "cuisine_list" in df.columns:
        all_cuisines = df["cuisine_list"].explode().dropna()
        cuisine_counts = all_cuisines.value_counts()
        props = cuisine_counts / cuisine_counts.sum()
        shannon = -sum(props * np.log(props))
        kpis["diversity"] = {
            "label": "Diversite culinaire",
            "value": f"{shannon:.2f}",
            "delta": None,
        }

    df_copy = df.copy()
    df_copy["z_rating"] = (df_copy["rating"] - df_copy["rating"].mean()) / df_copy["rating"].std()
    df_copy["z_reviews"] = (df_copy["reviewCount"] - df_copy["reviewCount"].mean()) / df_copy["reviewCount"].std()
    most_popular_idx = (df_copy["z_rating"] + df_copy["z_reviews"]).idxmax()
    most_pop_name = df_copy.loc[most_popular_idx].get("name", f"#{int(df_copy.loc[most_popular_idx, 'locationId'])}")
    kpis["most_popular"] = {
        "label": "Restaurant le plus populaire",
        "value": str(most_pop_name)[:35],
        "delta": None,
    }

    return kpis


def render_kpi_bar(kpis: dict, n_cols: int = 5):
    """Affiche une barre de KPI avec st.metric."""
    if not kpis:
        return
    cols = st.columns(min(n_cols, len(kpis)))
    for i, (key, kpi) in enumerate(kpis.items()):
        with cols[i % n_cols]:
            st.metric(
                label=kpi["label"],
                value=kpi["value"],
                delta=kpi.get("delta"),
            )


# ---------------------------------------------------------------------------
# Narration automatique
# ---------------------------------------------------------------------------

def render_narrative(narrative: dict):
    """Affiche la synthese narrative dans un bloc st.info."""
    if not narrative or "summary" not in narrative:
        return

    summary = narrative["summary"]
    if summary.get("info"):
        st.info(summary["info"])
        return

    bullets = []

    bullets.append(
        f"{summary.get('n_restaurants', '?')} restaurants analyses "
        f"dans {summary.get('n_arrondissements', '?')} arrondissements — "
        f"note moyenne {summary.get('rating_mean', '?')}/5, "
        f"mediane {summary.get('rating_median', '?')}/5"
    )

    for c in narrative.get("contrasts", [])[:4]:
        sig = " (significatif)" if c.get("significant") else ""
        bullets.append(f"{c['description']}{sig}")

    for ac in narrative.get("anticorrelations", [])[:2]:
        bullets.append(ac["description"])

    for rec in narrative.get("recommendations", [])[:2]:
        if "description" in rec:
            bullets.append(rec["description"])
        elif "items" in rec:
            bullets.append(f"**{rec['title']}** : {', '.join(rec['items'][:3])}")

    with st.container():
        st.info("### Ce qu'il faut retenir")
        for b in bullets:
            st.write(f"• {b}")


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

def get_csv_download_link(df: pd.DataFrame, filename: str = "restaurants_filtres.csv") -> bytes:
    """Prepare un CSV pour telechargement."""
    export_cols = [
        c for c in ["locationId", "name", "rating", "reviewCount", "priceLevel",
                     "arrondissement", "cuisines", "nb_items_total",
                     "shannon_entropy", "hasMichelin", "latitude", "longitude"]
        if c in df.columns
    ]
    return df[export_cols].to_csv(index=False).encode("utf-8")
