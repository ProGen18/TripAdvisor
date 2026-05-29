"""Distributions : boxplots, violons, histogrammes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_SEQ = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def boxplot_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str = "",
    show_points: bool = True,
    horizontal: bool = False,
) -> go.Figure:
    """Boxplot conditionnel trie par mediane.

    Args:
        df: DataFrame
        x_col: Colonne de groupement (x)
        y_col: Colonne de valeur (y)
        color_col: Colonne de couleur optionnelle
        title: Titre
        show_points: Afficher les suspected outliers
        horizontal: Boxplot horizontal
    """
    point_mode = "suspectedoutliers" if show_points else False

    fig = px.box(
        df,
        x=x_col if not horizontal else y_col,
        y=y_col if not horizontal else x_col,
        color=color_col,
        points=point_mode,
        title=title,
        color_discrete_sequence=COLOR_SEQ if not color_col else None,
    )
    fig.update_layout(height=450, boxmode="group")
    return fig


def violin_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str = "",
    box: bool = True,
) -> go.Figure:
    """Violin plot par groupe.

    Args:
        df: DataFrame
        x_col: Colonne de groupement
        y_col: Colonne de valeur
        color_col: Colonne de couleur optionnelle
        title: Titre
        box: Afficher une boite a moustaches a l'interieur
    """
    fig = px.violin(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        box=box,
        points=False,
        title=title,
        color_discrete_sequence=COLOR_SEQ if not color_col else None,
    )
    fig.update_layout(height=450)
    return fig


def histogram_conditionnel(
    df: pd.DataFrame,
    col: str,
    color_col: str | None = None,
    nbins: int = 30,
    title: str = "",
    marginal: str | None = None,
) -> go.Figure:
    """Histogramme conditionnel.

    Args:
        df: DataFrame
        col: Colonne a afficher
        color_col: Colonne de couleur optionnelle
        nbins: Nombre de bins
        title: Titre
        marginal: 'rug', 'box', 'violin' ou None
    """
    fig = px.histogram(
        df,
        x=col,
        color=color_col,
        nbins=nbins,
        marginal=marginal,
        title=title,
        color_discrete_sequence=COLOR_SEQ if not color_col else None,
        opacity=0.75,
        barmode="overlay" if color_col else "relative",
    )
    fig.update_layout(height=400)
    return fig


def cdf_comparison(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    title: str = "",
    top_n: int = 8,
) -> go.Figure:
    """Courbes CDF comparatives par groupe.

    Args:
        df: DataFrame
        col: Colonne de valeur
        group_col: Colonne de groupement
        title: Titre
        top_n: Nombre max de groupes a afficher
    """
    top_groups = df[group_col].value_counts().head(top_n).index.tolist()

    fig = go.Figure()
    for i, group in enumerate(top_groups):
        subset = df[df[group_col] == group][col].dropna().sort_values()
        if len(subset) < 2:
            continue
        cdf = np.arange(1, len(subset) + 1) / len(subset)

        fig.add_trace(go.Scatter(
            x=subset,
            y=cdf,
            mode="lines",
            name=str(group),
            line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)], width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=col,
        yaxis_title="Proportion cumulee",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


import numpy as np
