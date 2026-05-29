"""Page 1 — Synthese : KPI, narration auto, carte choroplethe, top 10."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from _helpers import compute_kpis, render_kpi_bar, empty_warning, ensure_data_loaded, COLOR_SEQ
from viz.maps import choropleth_arrondissement

st.set_page_config(page_title="Synthese", page_icon="📊", layout="wide")

ensure_data_loaded()
data = st.session_state["data"]
filtered = st.session_state["filtered"]
filters = st.session_state.get("filters", {})

st.title("📊 Synthese")
st.markdown("""
Cette page offre une **vue d'ensemble** de votre selection. Vous y trouverez les indicateurs de performance cles (KPI) 
et une cartographie des performances par arrondissement.
""")
st.caption(f"{len(filtered)} restaurants selectionnes")

if empty_warning(filtered):
    st.stop()

# --- KPI Bar ---
st.subheader("Indicateurs cles")
kpis = compute_kpis(filtered, data["enriched"])
render_kpi_bar(kpis)

st.divider()



# --- Carte choroplèthe ---
st.subheader("Note moyenne par arrondissement")
col1, col2 = st.columns([2, 1])

with col1:
    geojson_path = ROOT / "data" / "external" / "arrondissements.geojson"
    fig_map = choropleth_arrondissement(
        filtered,
        geojson_path=str(geojson_path) if geojson_path.exists() else None,
        color_col="rating",
        agg_func="mean",
        title="Note moyenne par arrondissement",
    )
    st.plotly_chart(fig_map, width='stretch')

with col2:
    st.subheader("Top 10 — Note")
    top10 = filtered.nlargest(10, "rating")[
        ["name", "rating", "reviewCount", "arrondissement", "priceLevel"]
    ]
    top10_disp = top10.rename(columns={
        "name": "Nom", "rating": "Note", "reviewCount": "Nombre d'avis", 
        "arrondissement": "Arrondissement", "priceLevel": "Niveau de prix"
    })
    st.dataframe(top10_disp, width='stretch', hide_index=True)

    st.subheader("Top 10 — Avis")
    top_reviews = filtered.nlargest(10, "reviewCount")[
        ["name", "reviewCount", "rating", "arrondissement"]
    ]
    top_reviews_disp = top_reviews.rename(columns={
        "name": "Nom", "reviewCount": "Nombre d'avis", "rating": "Note", "arrondissement": "Arrondissement"
    })
    st.dataframe(top_reviews_disp, width='stretch', hide_index=True)

# --- Repartition ---
st.divider()
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Restaurants Michelin", f"{filtered['hasMichelin'].sum()}")
with col_b:
    st.metric("Note mediane", f"{filtered['rating'].median():.2f}")
with col_c:
    st.metric("Avis total", f"{int(filtered['reviewCount'].sum()):,}")
