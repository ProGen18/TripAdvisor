"""Page 2 — Exploration descriptive : distributions cles + filtres univaries."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from _helpers import empty_warning, ensure_data_loaded
from viz.distributions import boxplot_by_group, histogram_conditionnel, cdf_comparison

st.set_page_config(page_title="Exploration descriptive", page_icon="⭐", layout="wide")

ensure_data_loaded()
data = st.session_state["data"]
filtered = st.session_state["filtered"]

st.title("⭐ Exploration descriptive")
st.markdown("""
Cette section vous permet de comprendre la **distribution** des principales caracteristiques des restaurants (notes, niveaux de prix, popularite). 
Grace a ces graphiques, vous pouvez facilement identifier les tendances dominantes et reperer d'eventuelles valeurs atypiques dans votre selection.
""")
st.caption(f"{len(filtered)} restaurants selectionnes")

if empty_warning(filtered):
    st.stop()

# --- Distributions cles ---
tab1, tab2, tab3 = st.tabs(["Notes", "Prix", "Avis & Popularite"])

with tab1:
    st.subheader("Distribution des notes")

    col1, col2 = st.columns(2)
    with col1:
        fig = histogram_conditionnel(
            filtered, col="rating", nbins=25,
            title="Distribution des notes",
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        if "arrondissement" in filtered.columns:
            fig = boxplot_by_group(
                filtered, x_col="arrondissement", y_col="rating",
                title="Notes par arrondissement",
                show_points=True,
            )
            st.plotly_chart(fig, width='stretch')

    if "priceLevel" in filtered.columns:
        fig = boxplot_by_group(
            filtered, x_col="priceLevel", y_col="rating",
            title="Notes par niveau de prix",
            show_points=True,
        )
        st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("Analyse des prix")

    col1, col2 = st.columns(2)
    with col1:
        if "priceLevel" in filtered.columns:
            price_counts = filtered["priceLevel"].value_counts().reset_index()
            price_counts.columns = ["Niveau de prix", "Nombre"]
            import plotly.express as px
            fig = px.bar(price_counts, x="Niveau de prix", y="Nombre",
                         title="Repartition par niveau de prix",
                         color="Niveau de prix")
            st.plotly_chart(fig, width='stretch')

    with col2:
        menu_f = data["menu_items"]
        f_ids = set(filtered["locationId"].astype(int).tolist())
        menu_f = menu_f[menu_f["locationId"].isin(f_ids)]
        if len(menu_f) > 0:
            fig = histogram_conditionnel(
                menu_f, col="item_price", nbins=40,
                title="Distribution des prix des items (€)",
            )
            st.plotly_chart(fig, width='stretch')

with tab3:
    st.subheader("Avis et popularite")

    col1, col2 = st.columns(2)
    with col1:
        fig = histogram_conditionnel(
            filtered[filtered["reviewCount"] > 0],
            col="reviewCount", nbins=40,
            title="Nombre d'avis par restaurant",
        )
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, width='stretch')

    with col2:
        if "nb_items_total" in filtered.columns:
            fig = histogram_conditionnel(
                filtered, col="nb_items_total", nbins=30,
                title="Nombre d'items au menu",
            )
            st.plotly_chart(fig, width='stretch')

# --- Table descriptive ---
st.divider()
st.subheader("Statistiques descriptives")
desc_cols = ["rating", "reviewCount", "nb_items_total", "shannon_entropy", "price_level_num"]
desc_cols = [c for c in desc_cols if c in filtered.columns]
if desc_cols:
    desc = filtered[desc_cols].describe().T
    desc["IQR"] = filtered[desc_cols].quantile(0.75) - filtered[desc_cols].quantile(0.25)
    desc = desc.rename(index={
        "rating": "Note",
        "reviewCount": "Nombre d'avis",
        "nb_items_total": "Nombre d'items au menu",
        "shannon_entropy": "Diversité du menu",
        "price_level_num": "Niveau de prix (numérique)"
    })
    st.dataframe(desc, width='stretch')
