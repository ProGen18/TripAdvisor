"""Page 3 — Croisements avances : heatmaps Khi², ANOVA, scatters, Sankey."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats

from _helpers import empty_warning, explode_cuisines, ensure_data_loaded
from scripts.stat_tests import (
    chi2_contingency, kruskal_wallis, spearman_r, anova, format_stat_for_ui
)
from viz.scatters import scatter_with_trendline, scatter_facet, hexbin_dense
from viz.crosstabs import heatmap_contingency, sankey_cuisine_prix_qualite, grouped_bar_comparison

st.set_page_config(page_title="Croisements avances", page_icon="📈", layout="wide")

ensure_data_loaded()
data = st.session_state["data"]
filtered = st.session_state["filtered"]

st.title("📈 Croisements avances")
st.markdown("""
Cet onglet explore les **relations et correlations** entre plusieurs variables (ex: note vs prix, diversite du menu vs satisfaction). 
Des tests statistiques de significativite (Khi², Kruskal-Wallis, correlation de Spearman) viennent appuyer les visualisations 
pour confirmer ou infirmer les tendances observees.
""")
st.caption(f"{len(filtered)} restaurants selectionnes")

if empty_warning(filtered):
    st.stop()

# --- 1. Note × Prix × Cuisine ---
st.subheader("Note × Prix × Cuisine")

col1, col2 = st.columns([2, 1])
with col1:
    if all(c in filtered.columns for c in ["rating", "priceLevel", "reviewCount"]):
        fig = scatter_with_trendline(
            filtered, x_col="rating", y_col="reviewCount",
            color_col="priceLevel",
            title="Note vs Nombre d'avis (par niveau de prix)",
            trendline="ols",
            log_x=False,
        )
        st.plotly_chart(fig, width='stretch')

with col2:
    # Spearman note vs avis
    mask = filtered["rating"].notna() & filtered["reviewCount"].notna()
    if mask.sum() >= 5:
        result = spearman_r(
            filtered.loc[mask, "reviewCount"].values,
            filtered.loc[mask, "rating"].values
        )
        st.markdown(format_stat_for_ui(result))

# --- Note vs Arrondissement ---
st.subheader("Note × Arrondissement")

arr_groups = [
    filtered[filtered["arrondissement"] == a]["rating"].dropna().values
    for a in sorted(filtered["arrondissement"].dropna().unique())
]
arr_groups = [g for g in arr_groups if len(g) >= 2]
arr_names = sorted(filtered["arrondissement"].dropna().unique())

if len(arr_groups) >= 2:
    kw_result = kruskal_wallis(*arr_groups, group_names=[str(int(a)) for a in arr_names])
    st.markdown(format_stat_for_ui(kw_result))

# --- 2. Cuisines × Arrondissement (Khi²) ---
st.subheader("Cuisines × Arrondissement")

if "cuisine_list" in filtered.columns:
    exploded = filtered[["arrondissement", "cuisine_list"]].explode("cuisine_list")
    exploded = exploded.dropna()

    top_cuisines = exploded["cuisine_list"].value_counts().head(8).index.tolist()
    cuisine_arr = exploded[exploded["cuisine_list"].isin(top_cuisines)]

    contingency = pd.crosstab(
        cuisine_arr["cuisine_list"],
        cuisine_arr["arrondissement"],
    )

    if not contingency.empty and contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = heatmap_contingency(contingency, show_residuals=True,
                                      title="Residus de Khi² : Cuisine × Arrondissement")
            st.plotly_chart(fig, width='stretch')

        with col2:
            chi_result = chi2_contingency(contingency)
            st.markdown(format_stat_for_ui(chi_result))

# --- 3. Note vs Diversite menu ---
st.subheader("Note vs Diversite du menu")

col1, col2 = st.columns([2, 1])
with col1:
    if all(c in filtered.columns for c in ["shannon_entropy", "rating", "priceLevel"]):
        fig = scatter_with_trendline(
            filtered, x_col="shannon_entropy", y_col="rating",
            color_col="priceLevel",
            title="Diversite du menu (Shannon) vs Note",
            trendline="lowess",
        )
        st.plotly_chart(fig, width='stretch')

with col2:
    mask = filtered["shannon_entropy"].notna() & filtered["rating"].notna()
    if mask.sum() >= 5:
        result = spearman_r(
            filtered.loc[mask, "shannon_entropy"].values,
            filtered.loc[mask, "rating"].values
        )
        st.markdown(format_stat_for_ui(result))

# --- 4. Concurrence 500m × Rating ---
st.subheader("Concurrence 500m × Note")

if all(c in filtered.columns for c in ["nb_restos_500m", "rating"]):
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = hexbin_dense(
            filtered, x_col="nb_restos_500m", y_col="rating",
            title="Concurrence a 500m vs Note",
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        mask = filtered["nb_restos_500m"].notna() & filtered["rating"].notna()
        if mask.sum() >= 5:
            result = spearman_r(
                filtered.loc[mask, "nb_restos_500m"].values,
                filtered.loc[mask, "rating"].values
            )
            st.markdown(format_stat_for_ui(result))

# --- 5. Sankey Cuisine → Prix → Qualite ---
st.subheader("Flux Cuisine → Prix → Qualite")
fig = sankey_cuisine_prix_qualite(filtered)
st.plotly_chart(fig, width='stretch')

# --- 6. Rapport qualite/prix ---
st.subheader("Top 20 — Rapport qualite/prix")

if "price_level_num" in filtered.columns:
    qp_df = filtered[filtered["price_level_num"] > 0].copy()
    if len(qp_df) > 0:
        rating_norm = (qp_df["rating"] - qp_df["rating"].min()) / (qp_df["rating"].max() - qp_df["rating"].min() + 0.01)
        price_norm = qp_df["price_level_num"] / qp_df["price_level_num"].max()
        qp_df["qp_score"] = rating_norm / price_norm
        top_qp = qp_df.nlargest(20, "qp_score")[
            ["name", "rating", "priceLevel", "arrondissement", "qp_score"]
        ] if "name" in qp_df.columns else qp_df.nlargest(20, "qp_score")[
            ["locationId", "rating", "priceLevel", "arrondissement", "qp_score"]
        ]
        top_qp_disp = top_qp.rename(columns={
            "name": "Nom", "locationId": "ID", "rating": "Note",
            "priceLevel": "Niveau de prix", "arrondissement": "Arrondissement", "qp_score": "Score Qualité/Prix"
        })
        st.dataframe(top_qp_disp, width='stretch', hide_index=True)
