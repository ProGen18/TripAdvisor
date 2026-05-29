"""Tableaux croises : heatmaps, Sankey, residus de Khi²."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats

COLOR_SEQ = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def heatmap_contingency(
    contingency: pd.DataFrame,
    show_residuals: bool = True,
    title: str = "Residus de Khi²",
    cmap: str = "RdBu_r",
) -> go.Figure:
    """Heatmap des residus standardises de Khi².

    Args:
        contingency: Table de contingence
        show_residuals: Si True, affiche les residus ; sinon les comptages bruts
        title: Titre
        cmap: Echelle de couleur
    """
    if show_residuals:
        chi2, p, dof, expected = scipy_stats.chi2_contingency(contingency.values)
        residuals = (contingency.values - expected) / np.sqrt(expected)
        data = residuals
        color_label = "Residu standardise"
        center = 0.0
    else:
        data = contingency.values
        color_label = "Effectif"
        center = None

    fig = px.imshow(
        data,
        x=contingency.columns.tolist(),
        y=contingency.index.tolist(),
        color_continuous_scale=cmap,
        color_continuous_midpoint=center if show_residuals else None,
        labels=dict(x=contingency.columns.name or "", y=contingency.index.name or "",
                     color=color_label),
        title=title,
        text_auto=".2f" if show_residuals else ".0f",
        aspect="auto",
    )
    fig.update_layout(height=500)
    return fig


def heatmap_corr(
    corr_matrix: np.ndarray,
    labels: list[str],
    title: str = "Matrice de correlations",
    cmap: str = "RdBu_r",
) -> go.Figure:
    """Heatmap de correlation.

    Args:
        corr_matrix: Matrice de correlation (n x n)
        labels: Noms des variables
        title: Titre
        cmap: Echelle de couleur
    """
    fig = px.imshow(
        corr_matrix,
        x=labels,
        y=labels,
        color_continuous_scale=cmap,
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
        title=title,
    )
    fig.update_layout(height=550)
    return fig


def sankey_cuisine_prix_qualite(
    df: pd.DataFrame,
    title: str = "Flux Cuisine → Prix → Qualite",
) -> go.Figure:
    """Diagramme Sankey cuisine → niveau de prix → qualite.

    Args:
        df: DataFrame avec colonnes 'cuisine_list', 'priceLevel', 'rating'
    """
    if "cuisine_list" not in df.columns or "priceLevel" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Donnees insuffisantes pour le Sankey", showarrow=False)
        return fig

    exploded = df[["cuisine_list", "priceLevel", "rating"]].explode("cuisine_list")
    exploded = exploded.dropna()
    if len(exploded) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Donnees insuffisantes pour le Sankey", showarrow=False)
        return fig

    exploded["qualite"] = pd.cut(
        exploded["rating"],
        bins=[0, 3.5, 4.0, 4.5, 5.0],
        labels=["Faible (<3.5)", "Moyen (3.5-4)", "Bon (4-4.5)", "Excellent (>4.5)"],
    )

    # Top cuisines pour lisibilite
    top_cuisines = exploded["cuisine_list"].value_counts().head(8).index.tolist()
    sankey_df = exploded[exploded["cuisine_list"].isin(top_cuisines)]

    # Cuisine -> Prix
    cp = sankey_df.groupby(["cuisine_list", "priceLevel"]).size().reset_index(name="count")
    # Prix -> Qualite
    pq = sankey_df.groupby(["priceLevel", "qualite"]).size().reset_index(name="count")

    all_nodes = list(cp["cuisine_list"].unique()) + list(cp["priceLevel"].unique()) + list(pq["qualite"].unique())
    all_nodes = list(dict.fromkeys(all_nodes))  # deduplicate preserving order

    node_idx = {n: i for i, n in enumerate(all_nodes)}

    sources = []
    targets = []
    values = []

    for _, row in cp.iterrows():
        sources.append(node_idx[row["cuisine_list"]])
        targets.append(node_idx[row["priceLevel"]])
        values.append(row["count"])

    for _, row in pq.iterrows():
        sources.append(node_idx[row["priceLevel"]])
        targets.append(node_idx[row["qualite"]])
        values.append(row["count"])

    node_colors = []
    for n in all_nodes:
        if n in top_cuisines:
            node_colors.append("rgba(31, 119, 180, 0.8)")
        elif n in cp["priceLevel"].unique():
            node_colors.append("rgba(255, 127, 14, 0.8)")
        else:
            node_colors.append("rgba(44, 160, 44, 0.8)")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        ),
    ))
    fig.update_layout(title=title, height=500)
    return fig


def grouped_bar_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = "",
    horizontal: bool = False,
) -> go.Figure:
    """Barplot groupe comparant deux categories.

    Args:
        df: DataFrame
        x_col: Colonne pour l'axe des categories
        y_col: Colonne pour les valeurs
        group_col: Colonne de groupement (couleur)
        title: Titre
        horizontal: Barres horizontales
    """
    fig = px.bar(
        df,
        x=y_col if horizontal else x_col,
        y=x_col if horizontal else y_col,
        color=group_col,
        barmode="group",
        title=title,
        color_discrete_sequence=COLOR_SEQ,
        orientation="h" if horizontal else "v",
    )
    fig.update_layout(height=450)
    return fig
