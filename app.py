"""Dashboard Streamlit — TripAdvisor Paris (2717 restaurants).

Architecture multi-page Streamlit :
    app.py                    → page d'accueil + sidebar globale + cache
    pages/1_Synthese.py       → KPI + narration + carte
    pages/2_Exploration.py    → distributions cles
    pages/3_Croisements.py    → heatmaps, ANOVA, scatters, Sankey
    pages/4_Sentiments.py     → sentiment maps, word clouds (Sprint 2)
    pages/5_Export.py         → CSV + PDF

Lancement :
    cd <worktree>
    $env:PYTHONPATH = "."
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from scripts.data_loader import load_all_data, get_available_filters

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TripAdvisor Paris — Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Chargement des donnees...", ttl=3600)
def cached_load_all():
    data = load_all_data()
    filters_meta = get_available_filters(data["enriched"])
    return data, filters_meta


if "data" not in st.session_state:
    data, filters_meta = cached_load_all()
    st.session_state["data"] = data
    st.session_state["filters_meta"] = filters_meta
else:
    data = st.session_state["data"]
    filters_meta = st.session_state["filters_meta"]


# ---------------------------------------------------------------------------
# Sidebar — filtres globaux
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🍽️ Filtres globaux")
    st.caption(
        f"Source : {len(data['enriched'])} restaurants · {len(data['menu_items'])} items de menu"
    )

    arr_selected = st.multiselect(
        "Arrondissements",
        options=filters_meta["arrondissements"],
        default=[],
        format_func=lambda n: f"Arr {n}",
        help="Vide = tous",
    )

    price_selected = st.multiselect(
        "Niveaux de prix",
        options=filters_meta["price_levels"],
        default=[],
        help="Vide = tous",
    )

    cuisine_selected = st.multiselect(
        "Cuisines (top 30)",
        options=filters_meta["top_cuisines"],
        default=[],
        help="Vide = toutes. Un resto compte si au moins une de ses cuisines est cochee.",
    )

    rating_min, rating_max = st.slider(
        "Intervalle de notes",
        min_value=float(filters_meta["rating_min"]),
        max_value=float(filters_meta["rating_max"]),
        value=(float(filters_meta["rating_min"]), float(filters_meta["rating_max"])),
        step=0.1,
    )

    price_min, price_max = st.slider(
        "Intervalle de prix items (€)",
        min_value=float(filters_meta["price_min"]),
        max_value=float(filters_meta["price_max"]),
        value=(float(filters_meta["price_min"]), float(filters_meta["price_max"])),
        step=1.0,
        help="Applique uniquement aux graphes par item de menu.",
    )

    michelin_only = st.checkbox("Restaurants Michelin uniquement", value=False)

    st.divider()

    review_count_min = st.number_input(
        "Nombre d'avis minimum",
        min_value=1,
        max_value=int(filters_meta["review_count_max"]),
        value=1,
        step=10,
        help="Filtre les restaurants avec au moins X avis.",
    )

    concurrence_max = st.slider(
        "Concurrence max a 500m",
        min_value=0,
        max_value=int(filters_meta["concurrence_max"]),
        value=int(filters_meta["concurrence_max"]),
        step=5,
        help="Nombre max de restaurants concurrents dans un rayon de 500m.",
    )

    menu_cat_selected = st.multiselect(
        "Categories de menu disponibles",
        options=filters_meta["menu_categories"],
        default=[],
        help="Vide = toutes.",
    )

    st.divider()
    st.caption("Les filtres s'appliquent a toutes les pages.")


# ---------------------------------------------------------------------------
# Application des filtres
# ---------------------------------------------------------------------------
def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if arr_selected:
        out = out[out["arrondissement"].isin(arr_selected)]

    if price_selected:
        out = out[out["priceLevel"].isin(price_selected)]

    if cuisine_selected:
        def _match(c: str) -> bool:
            if not isinstance(c, str):
                return False
            parts = [p.strip() for p in c.split("|")]
            return any(p in cuisine_selected for p in parts)

        out = out[out["cuisines"].apply(_match)]

    out = out[(out["rating"] >= rating_min) & (out["rating"] <= rating_max)]

    if michelin_only:
        out = out[out["hasMichelin"] == True]  # noqa: E712

    out = out[out["reviewCount"] >= review_count_min]

    out = out[out["nb_restos_500m"] <= concurrence_max]

    if menu_cat_selected:
        for cat in menu_cat_selected:
            out = out[out[cat] > 0]

    return out


filtered = _apply_filters(data["enriched"])

# Stocker dans la session state pour les pages
st.session_state["filtered"] = filtered
st.session_state["filters"] = {
    "arrondissements": arr_selected or None,
    "price_levels": price_selected or None,
    "cuisines": cuisine_selected or None,
    "rating_range": (rating_min, rating_max),
    "price_range": (price_min, price_max),
    "michelin_only": michelin_only,
    "review_count_min": review_count_min,
    "concurrence_max": concurrence_max,
    "menu_categories": menu_cat_selected or None,
}


# ---------------------------------------------------------------------------
# Page d'accueil
# ---------------------------------------------------------------------------
st.title("🍽️ TripAdvisor Paris — Analyse des restaurants")
st.caption(
    f"Selection courante : {len(filtered):,} / {len(data['enriched']):,} restaurants — "
    f"notes {rating_min:.1f} - {rating_max:.1f}"
)

st.divider()

st.markdown("""
### Bienvenue dans le dashboard d'analyse des restaurants parisiens

Ce dashboard vous permet d'explorer les donnees de **{n_total}** restaurants parisiens
extraites de TripAdvisor.

#### Pages disponibles

| Page | Contenu |
|------|---------|
| 📊 **Synthese** | KPI dynamiques, narration automatique, carte par arrondissement |
| ⭐ **Exploration descriptive** | Distributions des notes, prix, avis |
| 📈 **Croisements avances** | Tests statistiques (Khi², ANOVA, Spearman), heatmaps, Sankey |
| 💬 **Sentiments & Avis** | Analyse de sentiment des avis clients (a venir — Sprint 2) |
| 📥 **Export** | Telechargement CSV des donnees filtrees |

#### Comment naviguer ?

Utilisez la **barre laterale** (←) pour filtrer les restaurants, puis naviguez
entre les pages via le menu de navigation Streamlit dans la barre laterale.
""".format(n_total=len(data["enriched"])))

st.info(
    "👈 Commencez par selectionner vos filtres dans la barre laterale, "
    "puis naviguez vers les pages d'analyse."
)

# Resume rapide
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Restaurants", len(data["enriched"]))
with col2:
    st.metric("Arrondissements", data["enriched"]["arrondissement"].nunique())
with col3:
    st.metric("Note moyenne", f"{data['enriched']['rating'].mean():.2f}")
with col4:
    st.metric("Avis total", f"{int(data['enriched']['reviewCount'].sum()):,}")
with col5:
    st.metric("Michelin", f"{data['enriched']['hasMichelin'].sum():,}")
