"""Cartes : choroplethe par arrondissement, scatter mapbox."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLOR_SEQ = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def choropleth_arrondissement(
    df: pd.DataFrame,
    geojson_path: str | None = None,
    color_col: str = "rating",
    agg_func: str = "mean",
    title: str = "Note moyenne par arrondissement",
) -> go.Figure:
    """Carte choroplethe par arrondissement.

    Args:
        df: DataFrame avec colonnes 'arrondissement' et color_col
        geojson_path: Chemin vers le fichier GeoJSON APUR
        color_col: Colonne a agreger pour la couleur
        agg_func: Fonction d'agregation ('mean', 'median', 'count')
        title: Titre du graphique
    """
    agg = df.groupby("arrondissement")[color_col].agg(agg_func).reset_index()
    agg["arrondissement"] = agg["arrondissement"].astype(int)
    agg = agg.rename(columns={color_col: f"{color_col}_{agg_func}"})

    # Fallback si pas de GeoJSON : carte simple avec plotly
    if geojson_path and Path(geojson_path).exists():
        with open(geojson_path, encoding="utf-8") as f:
            geojson = json.load(f)

        fig = px.choropleth_mapbox(
            agg,
            geojson=geojson,
            locations="arrondissement",
            featureidkey="properties.c_ar",
            color=f"{color_col}_{agg_func}",
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            zoom=11,
            center={"lat": 48.8566, "lon": 2.3522},
            opacity=0.7,
            labels={f"{color_col}_{agg_func}": f"{color_col} ({agg_func})"},
            title=title,
            height=550,
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    else:
        # Fallback: bar chart trie par arrondissement
        agg_sorted = agg.sort_values(f"{color_col}_{agg_func}", ascending=True)
        fig = px.bar(
            agg_sorted,
            x="arrondissement",
            y=f"{color_col}_{agg_func}",
            color=f"{color_col}_{agg_func}",
            color_continuous_scale="Viridis",
            labels={"arrondissement": "Arrondissement"},
            title=title,
        )
        fig.update_layout(height=450, showlegend=False)

    return fig


def scatter_mapbox_restaurants(
    df: pd.DataFrame,
    color_col: str = "rating",
    size_col: str = "reviewCount",
    hover_name: str = "name",
    title: str = "Carte des restaurants",
    hexbin: bool = False,
) -> go.Figure:
    """Scatter mapbox des restaurants.

    Args:
        df: DataFrame avec latitude, longitude, color_col, size_col
        color_col: Colonne pour la couleur des points
        size_col: Colonne pour la taille des points
        hover_name: Colonne pour le nom au survol
        title: Titre
        hexbin: Si True, utilise hexbin au lieu de scatter
    """
    if hexbin and len(df) > 50:
        fig = go.Figure(go.Densitymapbox(
            lat=df["latitude"],
            lon=df["longitude"],
            z=df[color_col] if color_col in df.columns else None,
            radius=15,
            colorscale="Viridis",
            colorbar=dict(title=color_col),
        ))
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=11,
            mapbox_center={"lat": 48.8566, "lon": 2.3522},
            title=title,
            height=550,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
    else:
        size_max = 15 if len(df) < 200 else 8
        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color=color_col,
            size=size_col if size_col in df.columns else None,
            size_max=size_max,
            hover_name=hover_name if hover_name in df.columns else None,
            hover_data=["arrondissement", "rating", "priceLevel"] if "priceLevel" in df.columns else ["arrondissement", "rating"],
            color_continuous_scale="Viridis",
            zoom=11,
            center={"lat": 48.8566, "lon": 2.3522},
            mapbox_style="carto-positron",
            title=title,
            height=550,
            opacity=0.7,
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    return fig


def competition_map(
    df: pd.DataFrame,
    title: str = "Concurrence locale (restos a 500m)",
) -> go.Figure:
    """Carte de la concurrence locale.

    Args:
        df: DataFrame avec nb_restos_500m, latitude, longitude
    """
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="nb_restos_500m",
        size="nb_restos_500m",
        size_max=12,
        hover_name="name" if "name" in df.columns else None,
        hover_data=["arrondissement", "nb_restos_500m"],
        color_continuous_scale="Reds",
        zoom=11,
        center={"lat": 48.8566, "lon": 2.3522},
        mapbox_style="carto-positron",
        title=title,
        height=500,
        opacity=0.6,
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig
