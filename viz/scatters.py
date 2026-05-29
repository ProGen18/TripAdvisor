"""Scatters avec trendline, facettes, et lowess."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_SEQ = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def scatter_with_trendline(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    title: str = "",
    trendline: str = "lowess",
    opacity: float = 0.6,
    log_x: bool = False,
) -> go.Figure:
    """Scatter plot avec trendline optionnelle.

    Args:
        df: DataFrame
        x_col: Variable en abscisse
        y_col: Variable en ordonnee
        color_col: Variable de couleur
        size_col: Variable de taille des points
        title: Titre
        trendline: 'ols', 'lowess' ou None
        opacity: Opacite des points
        log_x: Echelle log sur l'axe x
    """
    hover_cols = [x_col, y_col]
    if color_col and color_col in df.columns:
        hover_cols.append(color_col)
    if size_col and size_col in df.columns:
        hover_cols.append(size_col)
    if "name" in df.columns:
        hover_cols.append("name")

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        size_max=15,
        trendline=trendline if trendline in ("ols", "lowess") else None,
        opacity=opacity,
        hover_data=[c for c in hover_cols if c in df.columns][:6],
        title=title,
        color_discrete_sequence=COLOR_SEQ if not color_col else None,
    )

    if log_x:
        fig.update_xaxes(type="log")

    fig.update_layout(height=480)
    return fig


def scatter_facet(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    facet_col: str,
    color_col: str | None = None,
    facet_ncol: int = 3,
    title: str = "",
    trendline: str | None = None,
    max_facets: int = 9,
) -> go.Figure:
    """Scatter avec facettes.

    Args:
        df: DataFrame
        x_col: Variable en abscisse
        y_col: Variable en ordonnee
        facet_col: Colonne de facette
        color_col: Variable de couleur
        facet_ncol: Nombre de colonnes de facettes
        title: Titre
        trendline: 'ols', 'lowess' ou None
        max_facets: Nombre max de facettes (garde les plus frequentes)
    """
    top_facets = df[facet_col].value_counts().head(max_facets).index.tolist()
    facet_df = df[df[facet_col].isin(top_facets)]

    fig = px.scatter(
        facet_df,
        x=x_col,
        y=y_col,
        color=color_col,
        facet_col=facet_col,
        facet_col_wrap=facet_ncol,
        trendline=trendline if trendline in ("ols", "lowess") else None,
        opacity=0.6,
        title=title,
        color_discrete_sequence=COLOR_SEQ if not color_col else None,
    )
    fig.update_layout(height=200 * ((len(top_facets) + facet_ncol - 1) // facet_ncol) + 100)
    return fig


def scatter_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str = "",
) -> go.Figure:
    """Scatter avec droite de regression OLS et intervalle de confiance.

    Args:
        df: DataFrame
        x_col: Variable en abscisse
        y_col: Variable en ordonnee
        color_col: Variable de couleur
        title: Titre
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline="ols",
        trendline_scope="overall",
        opacity=0.6,
        title=title,
        color_discrete_sequence=COLOR_SEQ if not color_col else None,
    )
    fig.update_layout(height=450)
    return fig


def hexbin_dense(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    gridsize: int = 30,
) -> go.Figure:
    """Hexbin pour donnees denses.

    Args:
        df: DataFrame
        x_col: Variable en abscisse
        y_col: Variable en ordonnee
        title: Titre
        gridsize: Taille de la grille hexagonale
    """
    fig = go.Figure(go.Histogram2dContour(
        x=df[x_col],
        y=df[y_col],
        colorscale="Viridis",
        contours=dict(showlabels=True),
        xbins=dict(size=(df[x_col].max() - df[x_col].min()) / gridsize) if len(df) > 0 else None,
        ybins=dict(size=(df[y_col].max() - df[y_col].min()) / gridsize) if len(df) > 0 else None,
    ))

    # Ajouter les points individuels avec faible opacite si peu de donnees
    if len(df) <= 500:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            marker=dict(color="black", size=3, opacity=0.3),
            showlegend=False,
        ))

    fig.update_layout(title=title, height=450)
    return fig
