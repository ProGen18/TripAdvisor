# TripAdvisor Paris — Pipeline & Dashboard restaurants

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-viz-3F4F75?logo=plotly&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white)

Analyse des restaurants parisiens extraits de TripAdvisor (~2 700 restaurants) :
menus, prix, notes, cuisines, géographie, saisonnalité, label Michelin et
analyse de sentiment des avis clients.

Le projet se compose de deux parties :

1. un **pipeline de traitement** (`scripts/`) qui transforme les JSON bruts en
   fichiers Parquet et statistiques pré-calculées ;
2. un **dashboard Streamlit multi-pages** (`app.py` + `pages/`) qui visualise ces
   données.

> [!IMPORTANT]
> **Les données brutes volumineuses ne sont PAS versionnées dans ce dépôt.**
> Le dossier `data/` est ignoré par Git. Chaque collaborateur doit créer ce dossier
> localement et y placer les fichiers JSON des restaurants (voir [Données](#données-à-placer-manuellement)).

---

## Sommaire

- [Fonctionnalités clés](#fonctionnalités-clés)
- [Stack technique](#stack-technique)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Données (à placer manuellement)](#données-à-placer-manuellement)
- [Flux de données](#flux-de-données)
- [Lancement du pipeline](#lancement-du-pipeline)
- [Lancement du dashboard](#lancement-du-dashboard)
- [Structure du projet](#structure-du-projet)
- [Tests](#tests)
- [Dépannage](#dépannage)

---

## Fonctionnalités clés

- **Consolidation** de centaines de JSON bruts (menus + avis) en un dataset
  Parquet unique, propre et typé.
- **Classification des menus** (entrées / plats / desserts / boissons) et calcul
  d'indices de diversité.
- **Enrichissement externe** : arrondissement, saisonnalité, intensité
  concurrentielle, croisement avec le label Michelin.
- **Statistiques descriptives** et tests inférentiels (Khi², ANOVA, Spearman).
- **Analyse de sentiment** des avis clients via un transformer multilingue, avec
  reprise sur incident, parallélisation par chunks et quantization CPU.
- **Dashboard interactif** Streamlit : KPI, cartes par arrondissement, croisements
  avancés, nuages de mots et export CSV.

---

## Stack technique

| Domaine | Outils |
|---------|--------|
| Données | pandas, numpy, pyarrow (Parquet) |
| Statistiques | scipy, statsmodels, scikit-learn |
| NLP / sentiment | spaCy (`fr_core_news_sm`), transformers + PyTorch *(optionnel)* |
| Visualisation | Plotly, Folium / streamlit-folium, wordcloud |
| Dashboard | Streamlit (multi-pages) |

---

## Prérequis

- Python 3.10 ou supérieur
- pip
- Git
- ~2 Go d'espace disque libre (données + modèles)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/ProGen18/TripAdvisor.git
cd TripAdvisor
```

### 2. Créer et activer l'environnement virtuel

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Modèle spaCy français
python -m spacy download fr_core_news_sm
```

### 4. (Optionnel) Analyse de sentiment par transformer

Les scripts `sentiment_transformer.py` et `sentiment_merge.py` nécessitent
PyTorch et Transformers, **non inclus** dans `requirements.txt` :

```bash
pip install torch transformers
```

---

## Données (à placer manuellement)

Les données brutes doivent être partagées séparément entre collaborateurs.
Créez le dossier `data/` à la racine et placez-y les JSON :

```
data/
├── menus_by_location/    # 1 JSON par restaurant (locationId) — menus
│   ├── 695062.json
│   └── ...
└── paris/                # 1 JSON par restaurant — avis (reviews)
    ├── 695062.json
    └── ...
```

Les sous-dossiers suivants sont créés **automatiquement** par le pipeline :

```
data/raw/         data/processed/   data/enriched/
data/external/    data/output/      data/stats/      data/lexicons/
```

---

## Flux de données

```
   data/menus_by_location/  +  data/paris/        (JSON bruts)
                    │
   ┌────────────────┴──────────────── pipeline principal ────────────────┐
   │                                                                      │
   ▼                                                                      │
 build_restaurants.py   →  data/processed/restaurants.parquet             │
   ▼                                                                      │
 classify_menu.py       →  data/processed/menu_diversity.parquet          │
   ▼                                                                      │
 enrich_external.py     →  data/enriched/restaurants_enriched.parquet     │
   ▼                                                                      │
 descriptive_stats.py   →  data/stats/ + data/output/ (graphes, stats)    │
   │                                                                      │
   └──────────────────────────────────────────────────────────────────► Dashboard
                                                                       (app.py + pages/)
   ┌──────────────────── analyse de sentiment (optionnelle) ────────────────────┐
   │ extract_local_reviews.py → data/raw/all_reviews.parquet                    │
   │ sentiment_transformer.py → scores de sentiment (Parquet)                   │
   │ sentiment_merge.py       → fusion des scores dans le dataset enrichi       │
   └────────────────────────────────────────────────────────────────────────────┘
```

---

## Lancement du pipeline

Le pipeline transforme les JSON bruts en Parquet et statistiques.
Lancez-le depuis la racine du projet, environnement virtuel activé.

### Option A — tout d'un coup (orchestrateur)

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."
python scripts/pipeline.py

# macOS / Linux
PYTHONPATH=. python scripts/pipeline.py
```

Étapes exécutées dans l'ordre : `build → classify → enrich → stats`.
Chaque étape est **ignorée si son fichier de sortie existe déjà** (supprimer le
fichier pour la rejouer).

Reprendre à une étape précise :

```bash
python scripts/pipeline.py --from enrich
```

### Option B — étape par étape

```bash
python scripts/build_restaurants.py   # consolide les JSON en restaurants.parquet
python scripts/classify_menu.py       # classification des items de menu
python scripts/enrich_external.py     # arrondissement, saisonnalité, concurrence
python scripts/descriptive_stats.py   # statistiques descriptives + graphes
```

### Analyse de sentiment (optionnelle)

```bash
# 1. Extraire les avis depuis les JSON locaux → data/raw/all_reviews.parquet
python scripts/extract_local_reviews.py                 # toutes les variantes : --max-restos N, --stratify

# 2. Scorer le sentiment (le cache est repris automatiquement)
python scripts/sentiment_transformer.py --sample 1000   # test rapide sur 1000 avis
python scripts/sentiment_transformer.py                 # run complet (reprend le cache existant)
python scripts/sentiment_transformer.py --no-resume     # ignore le cache et repart de zéro
python scripts/sentiment_transformer.py --quantize      # quantization int8 (plus rapide sur CPU)

# 3. Fusionner les scores dans le dataset
python scripts/sentiment_merge.py                       # --cleanup pour supprimer les fichiers _partN
```

> Pour les très gros volumes, le scoring peut être parallélisé en plusieurs
> workers via `--num-chunks N` / `--chunk-id i`, puis recombiné par `sentiment_merge.py`.

---

## Lancement du dashboard

Une fois le pipeline exécuté (les fichiers Parquet doivent exister dans
`data/processed/`, `data/enriched/` et `data/stats/`) :

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."
streamlit run app.py

# macOS / Linux
PYTHONPATH=. streamlit run app.py
```

L'application s'ouvre dans le navigateur (par défaut http://localhost:8501).

Pages disponibles :

| Page | Contenu |
|------|---------|
| Synthèse | KPI, narration automatique, carte par arrondissement |
| Exploration descriptive | distributions des notes, prix, avis |
| Croisements avancés | Khi², ANOVA, Spearman, heatmaps, Sankey |
| Sentiments & Avis | analyse de sentiment, nuages de mots, drill-down par restaurant |
| Export | téléchargement CSV des données filtrées |

Les filtres de la barre latérale (arrondissement, prix, cuisine, note, Michelin…)
s'appliquent à toutes les pages.

---

## Structure du projet

```
TripAdvisor/
│
├── app.py                      # point d'entrée du dashboard Streamlit
├── _helpers.py                 # fonctions utilitaires partagées (chargement session)
├── requirements.txt
├── .gitignore
│
├── pages/                      # pages Streamlit multi-pages
│   ├── 1_Synthese.py
│   ├── 2_Exploration_descriptive.py
│   ├── 3_Croisements_avances.py
│   ├── 4_Sentiments_et_Avis.py
│   └── 5_Export.py
│
├── scripts/                    # logique du pipeline
│   ├── pipeline.py             # orchestrateur (--from <étape>)
│   ├── config.py               # chemins & configuration centralisée
│   ├── build_restaurants.py    # consolidation JSON → restaurants.parquet
│   ├── classify_menu.py        # classification des items de menu
│   ├── enrich_external.py      # enrichissement géographique / saisonnier
│   ├── descriptive_stats.py    # statistiques + graphes
│   ├── data_loader.py          # chargement des données pour le dashboard
│   ├── extract_local_reviews.py
│   ├── sentiment_transformer.py
│   ├── sentiment_merge.py
│   └── stat_tests.py
│
├── viz/                        # fonctions de visualisation
│   ├── crosstabs.py            # tableaux croisés
│   ├── distributions.py        # histogrammes, box plots
│   ├── maps.py                 # cartes Folium
│   └── scatters.py             # nuages de points
│
├── tests/                      # tests unitaires (pytest)
│   ├── test_classify.py
│   ├── test_irony.py
│   └── test_sentiment_transformer.py
│
└── data/                       # NON versionné — données locales
    ├── menus_by_location/
    └── paris/
```

---

## Tests

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."
pytest

# macOS / Linux
PYTHONPATH=. pytest
```

---

## Dépannage

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError: scripts` | `PYTHONPATH` non défini | Lancer avec `$env:PYTHONPATH = "."` (Windows) ou `PYTHONPATH=.` (Linux/Mac) |
| `FileNotFoundError: *.parquet` au démarrage du dashboard | Pipeline non exécuté | Lancer d'abord `scripts/pipeline.py` |
| `OSError: [E050] Can't find model 'fr_core_news_sm'` | Modèle spaCy absent | `python -m spacy download fr_core_news_sm` |
| Erreurs torch / transformers | Dépendances optionnelles absentes | `pip install torch transformers` |
| Une étape du pipeline est systématiquement ignorée | Fichier de sortie déjà présent | Supprimer le `.parquet`/`.png` concerné, ou utiliser `--from <étape>` |
