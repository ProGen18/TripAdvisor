"""Page 5 — Export : CSV filtre + rapport PDF (Sprint 3)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from _helpers import empty_warning, get_csv_download_link, ensure_data_loaded

st.set_page_config(page_title="Export", page_icon="📥", layout="wide")

ensure_data_loaded()
data = st.session_state["data"]
filtered = st.session_state["filtered"]

st.title("📥 Export")
st.caption(f"{len(filtered)} restaurants selectionnes")

if empty_warning(filtered):
    st.stop()

# --- Export CSV ---
st.subheader("Export CSV")

st.write(f"{len(filtered)} lignes × {len(filtered.columns)} colonnes")

col1, col2 = st.columns(2)
with col1:
    available_cols = [
        c for c in ["locationId", "name", "rating", "reviewCount", "priceLevel",
                     "arrondissement", "cuisines", "nb_items_total",
                     "shannon_entropy", "hasMichelin", "latitude", "longitude",
                     "nb_restos_500m", "nb_restos_1km"]
        if c in filtered.columns
    ]
    
    rename_export = {
        "locationId": "ID Restaurant",
        "name": "Nom",
        "rating": "Note",
        "reviewCount": "Nombre d'avis",
        "priceLevel": "Niveau de prix",
        "arrondissement": "Arrondissement",
        "cuisines": "Cuisines",
        "nb_items_total": "Nombre d'items",
        "shannon_entropy": "Diversité menu",
        "hasMichelin": "Michelin",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "nb_restos_500m": "Concurrence 500m",
        "nb_restos_1km": "Concurrence 1km"
    }

    selected_cols = st.multiselect(
        "Colonnes a exporter",
        options=available_cols,
        default=available_cols[:8],
        format_func=lambda x: rename_export.get(x, x)
    )

with col2:
    if selected_cols:
        csv_bytes = filtered[selected_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Telecharger CSV",
            data=csv_bytes,
            file_name="restaurants_paris_filtres.csv",
            mime="text/csv",
        )
        st.caption(f"Taille : {len(csv_bytes) / 1024:.1f} KB")

# --- Apercu des donnees ---
st.divider()
st.subheader("Apercu des donnees filtrees")
if selected_cols:
    st.dataframe(filtered[selected_cols].head(50).rename(columns=rename_export), width='stretch')
else:
    st.dataframe(filtered.head(50).rename(columns=rename_export), width='stretch')

# --- Export PDF (Sprint 3) ---
st.divider()
st.subheader("Rapport PDF")
st.info(
    "L'export PDF sera disponible au Sprint 3. "
    "Le module `scripts/export_reports.py` generera un rapport complet avec "
    "page de garde, KPI, figures cles et tableaux filtres."
)
