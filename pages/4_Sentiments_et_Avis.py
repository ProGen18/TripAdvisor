"""Page 4 — Sentiments & Avis : analyse transformer multilingue (nlptown 1-5*).

Structure en 7 tabs :
  1. Vue d'ensemble  — KPI, distribution, distribution classes
  2. Calibration     — sentiment prédit vs note officielle, qualité, par langue
  3. Temporel        — évolution mensuelle, saisonnalité
  4. Carte           — geographique par arrondissement
  5. Discordances    — sur-cotés / sous-cotés (téléchargeable)
  6. Comparaisons    — Michelin, prix, cuisine, langue
  7. Polarisation    — restos clivants, confiance modèle
  8. Drill-down      — par restaurant : avis + wordclouds
"""
# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _helpers import empty_warning, ensure_data_loaded

st.set_page_config(page_title="Sentiments & Avis", page_icon="💬", layout="wide")

ensure_data_loaded()
data = st.session_state["data"]
filtered = st.session_state["filtered"]

st.title("💬 Sentiments & Avis — Transformer multilingue")
st.markdown("""
Cette page s'appuie sur un modele d'Intelligence Artificielle (NLP) pour analyser le contenu textuel des avis de maniere multilingue. 
L'objectif est d'aller au-dela de la note globale en mesurant la **tonalite reelle** des commentaires (positif, neutre, negatif) 
afin de detecter les restaurants polarisants, sur-cotes ou sous-cotes.
""")
st.caption(
    f"{len(filtered):,} restaurants selectionnes · Modele : "
    "`nlptown/bert-base-multilingual-uncased-sentiment` (FR/EN/IT/ES/PT/DE/NL) · "
    "5 classes 1-5 etoiles"
)

if empty_warning(filtered):
    st.stop()

# ---------------------------------------------------------------------------
# Données sources
# ---------------------------------------------------------------------------
sent_v2 = data.get("sentiment_v2", pd.DataFrame())
sent_reviews = data.get("sentiment_reviews", pd.DataFrame())

if sent_v2.empty and sent_reviews.empty:
    st.warning(
        "Sentiment transformer non disponible. Lancez :\n\n"
        "```bash\npython scripts/sentiment_transformer.py --batch-size 64 --quantize --threads 10\n```"
    )
    st.stop()

# Agrégation à la volée si parts en cours
if sent_v2.empty and not sent_reviews.empty:
    st.info(f"Scoring en cours — {len(sent_reviews):,} avis. Agrégation a la volée.")
    from scripts.data_loader import aggregate_sentiment_reviews
    sent_v2 = aggregate_sentiment_reviews(sent_reviews)

# Merge avec filtered
merged = filtered.merge(sent_v2, on="locationId", how="inner")
if merged.empty:
    st.warning("Aucun restaurant filtré n'a de sentiment scoré.")
    st.stop()

ids_filtered = set(merged["locationId"].astype(int).tolist())
reviews_f = sent_reviews[sent_reviews["locationId"].isin(ids_filtered)].copy()


@st.cache_data(show_spinner="Chargement des dates des avis...")
def load_review_dates() -> pd.DataFrame:
    """Charge uniquement les colonnes utiles depuis all_reviews.parquet."""
    raw = pd.read_parquet(
        ROOT / "data" / "raw" / "all_reviews.parquet",
        columns=["review_id", "publishedDate"],
    )
    raw["publishedDate"] = pd.to_datetime(raw["publishedDate"], errors="coerce")
    return raw


@st.cache_data(show_spinner="Chargement des textes des avis...")
def load_review_texts() -> pd.DataFrame:
    df = pd.read_parquet(
        ROOT / "data" / "raw" / "all_reviews.parquet",
        columns=["review_id", "text", "title", "language", "publishedDate"],
    )
    df["publishedDate"] = pd.to_datetime(df["publishedDate"], errors="coerce").dt.date
    return df


# ---------------------------------------------------------------------------
# Bandeau KPI
# ---------------------------------------------------------------------------
n_reviews_total = int(merged["n_reviews_scored"].sum())
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Restaurants", f"{len(merged):,}")
c2.metric("Avis scorés", f"{n_reviews_total:,}")
c3.metric("Sentiment moyen", f"{merged['sent_mean'].mean():.2f} ★")
c4.metric("% avis positifs (≥4★)", f"{merged['pct_positive'].mean():.1f}%")
c5.metric("% avis négatifs (≤2★)", f"{merged['pct_negative'].mean():.1f}%")
c6.metric("Discordance |moy|", f"{merged['discordance'].abs().mean():.2f} ★")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_overview, tab_calib, tab_time, tab_map, tab_disc, tab_comp, tab_polar, tab_irony, tab_drill = st.tabs([
    "📊 Vue d'ensemble",
    "🎯 Calibration",
    "📅 Temporel",
    "🗺️ Carte",
    "⚖️ Discordances",
    "🔬 Comparaisons",
    "⚡ Polarisation",
    "🕵️‍♂️ Ironie & Sarcasme",
    "🔍 Drill-down",
])

# ---------------------------------------------------------------------------
# TAB 1 — Vue d'ensemble
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Distribution globale du sentiment")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            merged, x="sent_mean", nbins=40,
            labels={"sent_mean": "Sentiment moyen prédit (★)"},
            title=f"Distribution par restaurant (n={len(merged):,})",
            color_discrete_sequence=["#2a9d8f"],
        )
        fig.add_vline(x=merged["sent_mean"].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Moy: {merged['sent_mean'].mean():.2f}")
        st.plotly_chart(fig, width='stretch')

    with col2:
        if not reviews_f.empty:
            class_counts = reviews_f["sent_class"].value_counts().sort_index()
            fig = px.bar(
                x=[f"{int(c)}★" for c in class_counts.index],
                y=class_counts.values,
                labels={"x": "Classe prédite", "y": "Nb d'avis"},
                title=f"Classes prédites ({len(reviews_f):,} avis)",
                text=[f"{v/len(reviews_f)*100:.1f}%" for v in class_counts.values],
                color=class_counts.values,
                color_continuous_scale="RdYlGn",
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, width='stretch')

    # Top restos négatifs et positifs
    st.subheader("Top restaurants")
    min_reviews_top = st.slider("Nb min d'avis scorés", 5, 200, 30, key="top_min")
    eligible = merged[merged["n_reviews_scored"] >= min_reviews_top]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚨 Plus d'avis négatifs (≤2★)**")
        top_neg = eligible.nlargest(15, "pct_negative")[
            ["name", "rating", "sent_mean", "pct_negative", "n_reviews_scored"]
        ]
        top_neg.columns = ["Nom", "Note★", "Sent.★", "% négatifs", "N avis"]
        st.dataframe(top_neg.round(2), width='stretch', hide_index=True)

    with col2:
        st.markdown("**⭐ Sentiment le plus positif**")
        top_pos = eligible.nlargest(15, "sent_mean")[
            ["name", "rating", "sent_mean", "pct_positive", "n_reviews_scored"]
        ]
        top_pos.columns = ["Nom", "Note★", "Sent.★", "% positifs", "N avis"]
        st.dataframe(top_pos.round(2), width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2 — Calibration
# ---------------------------------------------------------------------------
with tab_calib:
    st.subheader("Sentiment prédit vs note TripAdvisor")
    st.caption(
        "Chaque point = un restaurant. Diagonale rouge = accord parfait. "
        "Au-dessus : sentiment plus généreux que la note ; en-dessous : sentiment plus sévère."
    )

    plot_df = merged.dropna(subset=["sent_mean", "rating"]).copy()
    fig = px.scatter(
        plot_df, x="rating", y="sent_mean",
        hover_data=["name", "n_reviews_scored", "pct_negative", "pct_positive"],
        opacity=0.4,
        color="n_reviews_scored",
        color_continuous_scale="Viridis",
        labels={"rating": "Note TripAdvisor (★)", "sent_mean": "Sentiment prédit (★)",
                "n_reviews_scored": "N avis"},
        height=500,
    )
    fig.add_shape(type="line", x0=1, y0=1, x1=5, y1=5,
                  line=dict(color="red", dash="dash", width=2))
    fig.update_xaxes(range=[1, 5.2])
    fig.update_yaxes(range=[1, 5.2])
    st.plotly_chart(fig, width='stretch')

    # Métriques calibration
    mae = (plot_df["sent_mean"] - plot_df["rating"]).abs().mean()
    corr = plot_df[["sent_mean", "rating"]].corr().iloc[0, 1]
    bias = (plot_df["sent_mean"] - plot_df["rating"]).mean()
    within_05 = ((plot_df["sent_mean"] - plot_df["rating"]).abs() <= 0.5).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (★)", f"{mae:.2f}")
    c2.metric("Corr. Pearson", f"{corr:.3f}")
    c3.metric("Biais (sent − note)", f"{bias:+.2f}")
    c4.metric("% à ±0.5★", f"{within_05:.1f}%")

    # Calibration par langue (au niveau avis)
    if not reviews_f.empty and "language" in reviews_f.columns:
        st.subheader("Qualité par langue (au niveau avis)")
        rev_valid = reviews_f.dropna(subset=["rating"])
        lang_stats = []
        for lang in rev_valid["language"].value_counts().head(8).index:
            sub = rev_valid[rev_valid["language"] == lang]
            if len(sub) < 50:
                continue
            mae_l = (sub["sent_stars_expected"] - sub["rating"]).abs().mean()
            bias_l = (sub["sent_stars_expected"] - sub["rating"]).mean()
            corr_l = sub[["sent_stars_expected", "rating"]].corr().iloc[0, 1]
            lang_stats.append({
                "Langue": lang,
                "Nb avis": len(sub),
                "MAE": round(mae_l, 3),
                "Biais": round(bias_l, 3),
                "Corrélation": round(corr_l, 3),
            })
        if lang_stats:
            st.dataframe(pd.DataFrame(lang_stats), width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# TAB 3 — Temporel
# ---------------------------------------------------------------------------
with tab_time:
    st.subheader("Évolution temporelle du sentiment")

    if reviews_f.empty:
        st.info("Pas d'avis scorés à analyser.")
    else:
        with st.spinner("Jonction des dates..."):
            dates = load_review_dates()
            rev_dated = reviews_f.merge(dates, on="review_id", how="left")
            rev_dated = rev_dated.dropna(subset=["publishedDate"])
            rev_dated["year_month"] = rev_dated["publishedDate"].dt.to_period("M").dt.to_timestamp()
            rev_dated["year"] = rev_dated["publishedDate"].dt.year
            rev_dated["month"] = rev_dated["publishedDate"].dt.month

        # Évolution mensuelle
        monthly = rev_dated.groupby("year_month").agg(
            sent_mean=("sent_stars_expected", "mean"),
            n=("review_id", "count"),
            pct_neg=("sent_class", lambda c: (c <= 2).mean() * 100),
            pct_pos=("sent_class", lambda c: (c >= 4).mean() * 100),
        ).reset_index()
        # Filtre périodes avec assez d'avis
        monthly = monthly[monthly["n"] >= 50]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["year_month"], y=monthly["sent_mean"],
            mode="lines+markers", name="Sentiment moyen ★",
            line=dict(color="#2a9d8f", width=2),
        ))
        fig.update_layout(
            title="Sentiment moyen par mois (≥50 avis/mois)",
            xaxis_title="Date", yaxis_title="Sentiment moyen (★)",
            height=400, hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

        # Empilé positifs / neutres / négatifs
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["year_month"], y=monthly["pct_pos"],
            mode="lines", name="% positifs (≥4★)",
            line=dict(color="#2a9d8f"),
        ))
        fig.add_trace(go.Scatter(
            x=monthly["year_month"], y=monthly["pct_neg"],
            mode="lines", name="% négatifs (≤2★)",
            line=dict(color="#e63946"),
        ))
        fig.update_layout(
            title="Part positifs vs négatifs par mois",
            xaxis_title="Date", yaxis_title="% des avis",
            height=400, hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

        # Saisonnalité (moyenne sur tous les mois)
        col1, col2 = st.columns(2)
        with col1:
            monthly_pattern = rev_dated.groupby("month")["sent_stars_expected"].agg(["mean", "count"]).reset_index()
            monthly_pattern["month_name"] = monthly_pattern["month"].map({
                1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Juin",
                7: "Juil", 8: "Août", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc",
            })
            fig = px.bar(
                monthly_pattern, x="month_name", y="mean",
                hover_data=["count"],
                title="Saisonnalité — sentiment moyen par mois (tous les ans)",
                labels={"mean": "Sentiment moyen (★)", "month_name": "Mois"},
                color="mean", color_continuous_scale="RdYlGn",
                range_color=[monthly_pattern["mean"].min() - 0.05,
                             monthly_pattern["mean"].max() + 0.05],
            )
            fig.update_yaxes(range=[monthly_pattern["mean"].min() - 0.05,
                                    monthly_pattern["mean"].max() + 0.05])
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Volume par année
            yearly_vol = rev_dated.groupby("year").size().reset_index(name="n_avis")
            fig = px.bar(
                yearly_vol, x="year", y="n_avis",
                title="Volume d'avis par année",
                labels={"year": "Année", "n_avis": "Nb d'avis"},
                color_discrete_sequence=["#264653"],
            )
            st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# TAB 4 — Carte
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader("Carte du sentiment")

    if "latitude" in merged.columns and "longitude" in merged.columns:
        map_df = merged.dropna(subset=["latitude", "longitude"]).copy()
        map_df = map_df[map_df["n_reviews_scored"] >= 5]

        view_mode = st.radio(
            "Affichage",
            ["Points par restaurant", "Heatmap par arrondissement"],
            horizontal=True,
        )

        if view_mode == "Points par restaurant":
            fig = px.scatter_map(
                map_df, lat="latitude", lon="longitude",
                color="sent_mean",
                size="n_reviews_scored", size_max=18,
                color_continuous_scale="RdYlGn",
                range_color=[2.5, 5.0],
                hover_name="name",
                hover_data={"sent_mean": ":.2f", "rating": ":.2f",
                            "pct_negative": ":.1f", "n_reviews_scored": True,
                            "latitude": False, "longitude": False},
                zoom=11.5,
                map_style="carto-positron",
                height=600,
                center=dict(lat=48.8566, lon=2.3522),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            arr_agg = map_df.groupby("arrondissement").agg(
                sent_mean=("sent_mean", "mean"),
                n_restos=("locationId", "count"),
                pct_neg=("pct_negative", "mean"),
                pct_pos=("pct_positive", "mean"),
                lat=("latitude", "mean"),
                lon=("longitude", "mean"),
            ).reset_index()
            arr_agg = arr_agg[arr_agg["n_restos"] >= 3]
            arr_agg["arrondissement"] = arr_agg["arrondissement"].astype(int)
            arr_agg["label"] = arr_agg["arrondissement"].astype(str) + "ᵉ"

            fig = px.scatter_map(
                arr_agg, lat="lat", lon="lon",
                color="sent_mean",
                size="n_restos", size_max=50,
                color_continuous_scale="RdYlGn",
                hover_name="label",
                hover_data={"sent_mean": ":.2f", "pct_neg": ":.1f",
                            "pct_pos": ":.1f", "n_restos": True,
                            "lat": False, "lon": False},
                zoom=11, map_style="carto-positron",
                height=600,
                center=dict(lat=48.8566, lon=2.3522),
            )
            st.plotly_chart(fig, width='stretch')

            # Tableau ordonné
            arr_show = arr_agg[["label", "sent_mean", "pct_neg", "pct_pos", "n_restos"]].sort_values("sent_mean", ascending=False)
            arr_show.columns = ["Arr", "Sent.★", "% nég", "% pos", "N restos"]
            st.dataframe(arr_show.round(2), width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5 — Discordances
# ---------------------------------------------------------------------------
with tab_disc:
    st.subheader("Discordances note officielle vs sentiment prédit")
    st.caption(
        "Discordance = sentiment_prédit − note_moyenne. "
        "Permet de détecter (1) restos sur-cotés sur TripAdvisor (avis enthousiastes mais sentiment plus tiède), "
        "(2) sous-cotés où le sentiment du modèle est plus positif que la note."
    )

    min_rev_d = st.slider("Nb min d'avis scorés", 10, 200, 30, key="disc_min")
    disc = merged.dropna(subset=["discordance"])
    disc = disc[disc["n_reviews_scored"] >= min_rev_d]

    # Distribution
    fig = px.histogram(
        disc, x="discordance", nbins=50,
        title=f"Distribution des discordances ({len(disc):,} restos)",
        labels={"discordance": "Discordance (sent. − note)"},
        color_discrete_sequence=["#a8dadc"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, width='stretch')

    cols_show = ["name", "rating", "sent_mean", "discordance", "pct_negative",
                 "pct_positive", "n_reviews_scored"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📉 Sur-cotés (note > sentiment)**")
        st.caption("Pourquoi : avis enthousiastes en surface mais sentiment plus tiède → potentiel fake reviews ou inflation des notes.")
        over = disc.nsmallest(20, "discordance")[cols_show].copy()
        over.columns = ["Nom", "Note★", "Sent.★", "Δ", "% nég", "% pos", "N avis"]
        st.dataframe(over.round(2), width='stretch', hide_index=True)

    with col2:
        st.markdown("**📈 Sous-cotés (sentiment > note)**")
        st.caption("Pourquoi : avis textuellement positifs mais note basse → effet \"je donne 4★ par défaut\", attentes hautes, ou critères externes (prix, attente).")
        under = disc.nlargest(20, "discordance")[cols_show].copy()
        under.columns = ["Nom", "Note★", "Sent.★", "Δ", "% nég", "% pos", "N avis"]
        st.dataframe(under.round(2), width='stretch', hide_index=True)

    # Export
    st.divider()
    export_df = disc.sort_values("discordance")[cols_show].round(3)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger toutes les discordances (CSV)",
        data=csv_bytes,
        file_name="discordances_sentiment.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# TAB 6 — Comparaisons
# ---------------------------------------------------------------------------
with tab_comp:
    st.subheader("Comparaisons par segment")

    # 6.1 Michelin
    if "hasMichelin" in merged.columns:
        st.markdown("### 🌟 Michelin vs non-Michelin")
        mich_stats = merged.groupby("hasMichelin").agg(
            sent_mean=("sent_mean", "mean"),
            sent_std=("sent_mean", "std"),
            pct_neg=("pct_negative", "mean"),
            pct_pos=("pct_positive", "mean"),
            n_restos=("locationId", "count"),
            n_reviews=("n_reviews_scored", "sum"),
        ).reset_index()
        mich_stats["Catégorie"] = mich_stats["hasMichelin"].map({True: "Michelin", False: "Non-Michelin"})

        col1, col2 = st.columns([1, 2])
        with col1:
            mich_disp = mich_stats[["Catégorie", "n_restos", "n_reviews", "sent_mean",
                        "pct_neg", "pct_pos"]].rename(columns={
                "n_restos": "Nombre de restos",
                "n_reviews": "Nombre d'avis",
                "sent_mean": "Sentiment moyen",
                "pct_neg": "% négatifs",
                "pct_pos": "% positifs"
            })
            st.dataframe(
                mich_disp.round(2),
                width='stretch', hide_index=True,
            )

            # Test stat : Welch's t-test sur sent_mean
            from scipy import stats as sci
            a = merged[merged["hasMichelin"].eq(True)]["sent_mean"].dropna()
            b = merged[merged["hasMichelin"].eq(False)]["sent_mean"].dropna()
            if len(a) >= 10 and len(b) >= 10:
                t, p = sci.ttest_ind(a, b, equal_var=False)
                st.caption(f"Welch's t-test : t={t:.2f}, p={p:.4g} "
                           f"{'(significatif)' if p < 0.05 else '(non significatif)'}")

        with col2:
            fig = px.box(
                merged, x="hasMichelin", y="sent_mean",
                color="hasMichelin",
                title="Distribution sentiment moyen",
                labels={"hasMichelin": "Michelin", "sent_mean": "Sentiment (★)"},
                color_discrete_map={True: "#e9c46a", False: "#264653"},
            )
            st.plotly_chart(fig, width='stretch')

    st.divider()

    # 6.2 Prix
    if "priceLevel" in merged.columns:
        st.markdown("### 💰 Sentiment par niveau de prix")
        price_stats = merged.dropna(subset=["priceLevel"]).groupby("priceLevel").agg(
            sent_mean=("sent_mean", "mean"),
            n=("locationId", "count"),
            pct_neg=("pct_negative", "mean"),
        ).reset_index().sort_values("priceLevel")
        fig = px.bar(
            price_stats, x="priceLevel", y="sent_mean",
            color="sent_mean", color_continuous_scale="RdYlGn",
            hover_data=["n", "pct_neg"],
            labels={"priceLevel": "Niveau de prix", "sent_mean": "Sentiment moyen (★)"},
        )
        st.plotly_chart(fig, width='stretch')

    st.divider()

    # 6.3 Par cuisine
    if "cuisine_list" in merged.columns:
        st.markdown("### 🍽️ Sentiment par cuisine (≥15 restos)")
        cuisine_rows = []
        for _, row in merged.iterrows():
            cuisines = row.get("cuisine_list", [])
            if not isinstance(cuisines, (list, np.ndarray)):
                continue
            for c in cuisines:
                cuisine_rows.append({
                    "cuisine": c, "sent_mean": row["sent_mean"],
                    "pct_negative": row["pct_negative"],
                    "pct_positive": row["pct_positive"],
                })
        if cuisine_rows:
            cdf = pd.DataFrame(cuisine_rows)
            cui_stats = cdf.groupby("cuisine").agg(
                sent_mean=("sent_mean", "mean"),
                n_restos=("sent_mean", "count"),
                pct_neg=("pct_negative", "mean"),
                pct_pos=("pct_positive", "mean"),
            ).reset_index()
            cui_stats = cui_stats[cui_stats["n_restos"] >= 15].sort_values("sent_mean")

            fig = px.bar(
                cui_stats, x="sent_mean", y="cuisine",
                orientation="h",
                color="sent_mean", color_continuous_scale="RdYlGn",
                hover_data=["n_restos", "pct_neg", "pct_pos"],
                labels={"cuisine": "Cuisine", "sent_mean": "Sentiment moyen (★)"},
                height=max(400, 22 * len(cui_stats)),
            )
            st.plotly_chart(fig, width='stretch')

    st.divider()

    # 6.4 Par langue (sentiment moyen au niveau avis)
    if not reviews_f.empty and "language" in reviews_f.columns:
        st.markdown("### 🌐 Sentiment par langue d'avis")
        st.caption("Détecte si les touristes étrangers vs locaux ont des perceptions différentes des mêmes restos.")
        lang_avg = reviews_f.groupby("language").agg(
            sent_mean=("sent_stars_expected", "mean"),
            rating_mean=("rating", "mean"),
            n=("review_id", "count"),
            pct_neg=("sent_class", lambda c: (c <= 2).mean() * 100),
        ).reset_index()
        lang_avg = lang_avg[lang_avg["n"] >= 200].sort_values("sent_mean", ascending=False)

        fig = px.bar(
            lang_avg, x="language", y="sent_mean",
            color="sent_mean", color_continuous_scale="RdYlGn",
            hover_data=["n", "rating_mean", "pct_neg"],
            labels={"language": "Langue", "sent_mean": "Sentiment moyen (★)"},
        )
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# TAB 7 — Polarisation & confiance
# ---------------------------------------------------------------------------
with tab_polar:
    st.subheader("Restaurants polarisants (avis clivés)")
    st.caption(
        "L'écart-type intra-resto du sentiment mesure le clivage : "
        "haute valeur = beaucoup d'avis très positifs ET très négatifs."
    )

    pol_df = merged.dropna(subset=["sent_std"]).copy()
    pol_df = pol_df[pol_df["n_reviews_scored"] >= 30]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            pol_df, x="sent_std", nbins=40,
            title="Distribution de l'écart-type intra-resto",
            labels={"sent_std": "Écart-type sentiment"},
            color_discrete_sequence=["#9d4edd"],
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = px.scatter(
            pol_df, x="sent_mean", y="sent_std",
            hover_data=["name", "n_reviews_scored", "pct_negative"],
            opacity=0.5,
            color="pct_negative", color_continuous_scale="Reds",
            labels={"sent_mean": "Sentiment moyen (★)",
                    "sent_std": "Écart-type intra-resto",
                    "pct_negative": "% négatifs"},
            title="Polarisation vs sentiment moyen",
        )
        st.plotly_chart(fig, width='stretch')

    st.markdown("**Top 15 restos les plus polarisants**")
    top_pol = pol_df.nlargest(15, "sent_std")[
        ["name", "sent_mean", "sent_std", "pct_negative", "pct_positive", "n_reviews_scored"]
    ]
    top_pol.columns = ["Nom", "Sent.★", "Écart-type", "% nég", "% pos", "N avis"]
    st.dataframe(top_pol.round(2), width='stretch', hide_index=True)

    st.divider()

    # Confiance du modèle
    st.subheader("Confiance du modèle")
    st.caption(
        "Confiance = max(probabilités classes). "
        "Haute = modèle sûr de sa prédiction ; basse = avis ambigu pour le modèle."
    )
    if not reviews_f.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                reviews_f, x="sent_confidence", nbins=40,
                title=f"Distribution de la confiance ({len(reviews_f):,} avis)",
                labels={"sent_confidence": "Confiance max"},
                color_discrete_sequence=["#118ab2"],
            )
            st.plotly_chart(fig, width='stretch')
        with col2:
            # Confiance par classe prédite
            conf_by_class = reviews_f.groupby("sent_class")["sent_confidence"].mean().reset_index()
            conf_by_class["sent_class"] = conf_by_class["sent_class"].astype(int).astype(str) + "★"
            fig = px.bar(
                conf_by_class, x="sent_class", y="sent_confidence",
                title="Confiance moyenne par classe prédite",
                labels={"sent_class": "Classe", "sent_confidence": "Confiance moyenne"},
                color="sent_confidence", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, width='stretch')

        avg_conf = reviews_f["sent_confidence"].mean()
        low_conf = (reviews_f["sent_confidence"] < 0.4).sum()
        st.metric("Confiance moyenne", f"{avg_conf:.2%}",
                  delta=f"{low_conf:,} avis très ambigus (<40%)")

# ---------------------------------------------------------------------------
# TAB 7.5 — Ironie & Sarcasme
# ---------------------------------------------------------------------------
with tab_irony:
    st.subheader("🕵️‍♂️ Détection de l'Ironie & Sarcasme")
    st.markdown("""
    Cette section identifie les avis suspectés d'**ironie ou de sarcasme**. Un avis est classé comme *ironique* 
    si l'analyse du sentiment textuel prédit un ton très positif (≥ 4★) alors que la note officielle attribuée par le client est très basse (≤ 2★).
    Dans ce cas, le sentiment prédit est probablement erroné en raison de tournures sarcastiques, et est corrigé pour correspondre à la note réelle du client.
    """)

    if reviews_f.empty:
        st.info("Aucun avis disponible pour l'analyse de l'ironie.")
    else:
        # 1. KPIs
        n_ironic = int(reviews_f["is_ironic"].fillna(False).sum())
        pct_ironic = reviews_f["is_ironic"].fillna(False).mean() * 100
        avg_sent = reviews_f["sent_stars_expected"].mean()
        avg_sent_corr = reviews_f["sent_stars_corrected"].mean()
        delta_sent = avg_sent_corr - avg_sent

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avis ironiques", f"{n_ironic:,}")
        c2.metric("Part de l'ironie", f"{pct_ironic:.2f}%")
        c3.metric("Sentiment moyen initial", f"{avg_sent:.2f} ★")
        c4.metric("Sentiment moyen corrigé", f"{avg_sent_corr:.2f} ★", delta=f"{delta_sent:+.2f} ★", delta_color="inverse")

        st.write("---")

        # 2. Top restaurants ciblés par l'ironie
        st.subheader("🚨 Top restaurants ciblés par l'ironie")
        st.markdown("Restaurants ayant le plus fort taux ou volume d'avis ironiques.")
        
        min_rev_irony = st.slider(
            "Nombre minimum d'avis pour figurer dans le classement",
            min_value=5,
            max_value=100,
            value=15,
            key="irony_min_reviews_slider"
        )
        
        # S'assurer que la colonne n_ironic est présente
        merged["n_ironic"] = (merged["n_reviews_scored"] * merged["pct_ironic"].fillna(0) / 100).round().astype(int)
        
        eligible_irony = merged[merged["n_reviews_scored"] >= min_rev_irony].copy()
        
        if eligible_irony.empty:
            st.info("Aucun restaurant n'a assez d'avis pour la sélection actuelle.")
        else:
            sort_by = st.radio(
                "Trier par",
                ["Taux d'ironie (%)", "Nombre d'avis ironiques"],
                horizontal=True,
                key="irony_sort_radio"
            )
            
            if sort_by == "Taux d'ironie (%)":
                top_irony = eligible_irony.nlargest(15, "pct_ironic")
            else:
                top_irony = eligible_irony.nlargest(15, "n_ironic")
                
            top_irony_display = top_irony[[
                "name", "rating", "sent_mean", "sent_mean_corrected", "pct_ironic", "n_ironic", "n_reviews_scored"
            ]].copy()
            
            top_irony_display.columns = [
                "Nom", "Note moyenne ★", "Sent. initial ★", "Sent. corrigé ★", "% Ironie", "Nb avis ironiques", "Total avis scorés"
            ]
            
            st.dataframe(
                top_irony_display.round(2),
                width='stretch',
                hide_index=True
            )

        st.write("---")

        # 3. Exploration par restaurant
        st.subheader("🔍 Exploration des avis ironiques par restaurant")
        
        resto_with_irony = merged[merged["n_ironic"] > 0].sort_values("name")
        
        if resto_with_irony.empty:
            st.info("Aucun restaurant n'a d'avis ironique détecté.")
        else:
            options_irony = [
                f"{row['name']} ({int(row['n_ironic'])} avis ironiques / {int(row['n_reviews_scored'])} avis total)"
                for _, row in resto_with_irony.iterrows()
            ]
            idx_irony_map = {disp: row["locationId"] for disp, (_, row) in zip(options_irony, resto_with_irony.iterrows())}
            
            sel_resto_irony = st.selectbox(
                "Sélectionner un restaurant à analyser",
                options=options_irony,
                index=None,
                placeholder="Rechercher un restaurant...",
                key="irony_resto_select"
            )
            
            if sel_resto_irony:
                sel_id = idx_irony_map[sel_resto_irony]
                resto_reviews = reviews_f[reviews_f["locationId"] == sel_id].copy()
                
                with st.spinner("Chargement des textes des avis..."):
                    try:
                        raw_full = load_review_texts()
                        resto_reviews = resto_reviews.merge(raw_full, on="review_id", how="left")
                    except Exception as e:
                        st.error(f"Erreur lors du chargement des textes : {e}")
                
                ironic_only = resto_reviews[resto_reviews["is_ironic"].fillna(False)].copy()
                
                c1_resto, c2_resto, c3_resto = st.columns(3)
                resto_info = resto_with_irony[resto_with_irony["locationId"] == sel_id].iloc[0]
                c1_resto.metric("Note moyenne TA", f"{resto_info['rating']:.2f} ★")
                c1_resto.metric("Total avis scorés", f"{int(resto_info['n_reviews_scored'])}")
                c2_resto.metric("Sentiment initial moyen", f"{resto_info['sent_mean']:.2f} ★")
                c2_resto.metric("Nb avis ironiques", f"{int(resto_info['n_ironic'])}")
                c3_resto.metric("Sentiment corrigé moyen", f"{resto_info['sent_mean_corrected']:.2f} ★")
                c3_resto.metric("Taux d'ironie", f"{resto_info['pct_ironic']:.1f}%")
                
                st.write("---")
                st.markdown(f"#### 💬 Liste des {len(ironic_only)} avis ironiques détectés")
                st.caption("Ces avis ont été classés positivement (≥4★) par l'IA mais ont reçu une note très basse (≤2★) de l'utilisateur.")
                
                cols_to_show = [c for c in ["publishedDate", "sent_stars_expected", "rating", "language", "title", "text"] if c in ironic_only.columns]
                
                rename_cols_irony = {
                    "publishedDate": "Date",
                    "sent_stars_expected": "Sentiment prédit (IA)",
                    "rating": "Note réelle (TA)",
                    "language": "Langue",
                    "title": "Titre",
                    "text": "Contenu de l'avis"
                }
                
                if not ironic_only.empty:
                    st.dataframe(
                        ironic_only[cols_to_show].rename(columns=rename_cols_irony),
                        width='stretch',
                        hide_index=True
                    )
                else:
                    st.info("Aucun avis ironique trouvé pour ce restaurant dans les avis filtrés.")

# ---------------------------------------------------------------------------
# TAB 8 — Drill-down
# ---------------------------------------------------------------------------
with tab_drill:
    st.subheader("🔍 Inspection des avis d'un restaurant")

    if reviews_f.empty:
        st.info("Pas d'avis scorés.")
    else:
        options = merged.sort_values("name")[["locationId", "name", "sent_mean", "rating", "n_reviews_scored"]]
        options_display = [
            f"{row['name']} (sent {row['sent_mean']:.2f}★ / note {row['rating']:.2f}★ · {int(row['n_reviews_scored'])} avis)"
            for _, row in options.iterrows()
        ]
        idx_map = {disp: i for disp, i in zip(options_display, options["locationId"])}

        sel = st.selectbox(
            "Sélectionner un restaurant",
            options=options_display,
            index=None,
            placeholder="Tapez un nom...",
        )

        if sel:
            sel_id = idx_map[sel]
            revs = reviews_f[reviews_f["locationId"] == sel_id].copy()
            revs = revs.sort_values("sent_stars_expected")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avis scorés", len(revs))
            c2.metric("Sentiment moyen", f"{revs['sent_stars_expected'].mean():.2f} ★")
            c3.metric("% négatifs (≤2★)", f"{(revs['sent_class'] <= 2).mean() * 100:.1f}%")
            c4.metric("% positifs (≥4★)", f"{(revs['sent_class'] >= 4).mean() * 100:.1f}%")

            # Distribution interne
            col1, col2 = st.columns(2)
            with col1:
                class_dist = revs["sent_class"].value_counts().sort_index()
                fig = px.bar(
                    x=[f"{int(c)}★" for c in class_dist.index],
                    y=class_dist.values,
                    title="Répartition des classes",
                    color=class_dist.values, color_continuous_scale="RdYlGn",
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Évolution dans le temps si possible
                try:
                    dates = load_review_dates()
                    revs_dated = revs.merge(dates, on="review_id", how="left").dropna(subset=["publishedDate"])
                    if len(revs_dated) >= 20:
                        revs_dated["year"] = revs_dated["publishedDate"].dt.year
                        yearly = revs_dated.groupby("year")["sent_stars_expected"].mean().reset_index()
                        fig = px.line(
                            yearly, x="year", y="sent_stars_expected",
                            markers=True,
                            title="Sentiment par année",
                            labels={"sent_stars_expected": "Sentiment moyen (★)"},
                        )
                        st.plotly_chart(fig, width='stretch')
                except Exception:
                    pass

            # Avis textuels
            try:
                raw_full = load_review_texts()
                revs = revs.merge(raw_full, on="review_id", how="left")
            except Exception:
                pass

            cols_show = [c for c in ["publishedDate", "sent_stars_expected", "rating",
                                     "language", "title", "text"] if c in revs.columns]
            
            rename_cols = {
                "publishedDate": "Date",
                "sent_stars_expected": "Sentiment prédit",
                "rating": "Note",
                "language": "Langue",
                "title": "Titre",
                "text": "Avis"
            }

            st.markdown("**🔻 5 avis les plus négatifs**")
            st.dataframe(revs.head(5)[cols_show].rename(columns=rename_cols), width='stretch', hide_index=True)

            st.markdown("**🔺 5 avis les plus positifs**")
            st.dataframe(revs.tail(5)[cols_show].iloc[::-1].rename(columns=rename_cols), width='stretch', hide_index=True)

            # Wordcloud négatifs / positifs
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                if "text" in revs.columns:
                    st.subheader("Nuages de mots (texte des avis)")
                    col1, col2 = st.columns(2)
                    stopwords_fr = set([
                        "le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "a", "au",
                        "aux", "en", "est", "ce", "que", "qui", "pour", "pas", "ne", "nous",
                        "vous", "ils", "elles", "il", "elle", "on", "se", "sa", "son", "ses",
                        "mais", "ou", "y", "leur", "leurs", "dans", "par", "sur", "avec", "plus",
                        "très", "tres", "tout", "tous", "toute", "toutes", "cette", "ces",
                        "etait", "était", "ont", "ete", "été", "j'ai", "ca", "ça", "c'est",
                        "il", "y", "a", "comme", "aussi", "bien", "fait", "faire", "même",
                        "meme", "deja", "déjà", "the", "and", "of", "to", "is", "in", "for",
                        "we", "this", "with", "was", "very", "but", "had", "have", "they",
                        "our", "are", "us", "it", "be", "i", "you", "on", "as", "if", "from",
                    ])
                    neg_text = " ".join(revs[revs["sent_class"] <= 2]["text"].dropna().astype(str).tolist()[:200])
                    pos_text = " ".join(revs[revs["sent_class"] >= 4]["text"].dropna().astype(str).tolist()[:200])

                    if neg_text:
                        with col1:
                            st.markdown("**Avis négatifs (≤2★)**")
                            wc = WordCloud(width=600, height=300, background_color="white",
                                           stopwords=stopwords_fr, max_words=50,
                                           colormap="Reds").generate(neg_text)
                            fig, ax = plt.subplots(figsize=(7, 3.5))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)
                    if pos_text:
                        with col2:
                            st.markdown("**Avis positifs (≥4★)**")
                            wc = WordCloud(width=600, height=300, background_color="white",
                                           stopwords=stopwords_fr, max_words=50,
                                           colormap="Greens").generate(pos_text)
                            fig, ax = plt.subplots(figsize=(7, 3.5))
                            ax.imshow(wc, interpolation="bilinear")
                            ax.axis("off")
                            st.pyplot(fig)
            except ImportError:
                pass
