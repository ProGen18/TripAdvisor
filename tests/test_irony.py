import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_loader import aggregate_sentiment_reviews


def test_irony_detection_and_correction():
    # Jeu de données simulé couvrant tous les cas
    scored_reviews = pd.DataFrame(
        {
            "review_id": [1, 2, 3, 4, 5],
            "locationId": [101, 101, 101, 102, 102],
            "rating": [1.0, 5.0, np.nan, 2.0, 4.0],
            "sent_stars_expected": [4.5, 4.8, 4.2, 3.8, 1.5],
            "sent_class": [5, 5, 4, 4, 1],
            "sent_confidence": [0.9, 0.95, 0.8, 0.75, 0.85],
        }
    )

    # Exécution de l'agrégation
    aggregated = aggregate_sentiment_reviews(scored_reviews)

    # Vérifications pour le Restaurant 101
    # Review 1: Ironique (class 5 >= 4, rating 1 <= 2) -> corrigé = 1.0, is_ironic = True
    # Review 2: Non Ironique (class 5 >= 4, rating 5 > 2) -> corrigé = 4.8, is_ironic = False
    # Review 3: Non Ironique (rating est NaN) -> corrigé = 4.2, is_ironic = False
    # Moyenne attendue corrigée : (1.0 + 4.8 + 4.2) / 3 = 3.3333...
    # Taux d'ironie attendu : 1 / 3 = 33.3333%
    r101 = aggregated[aggregated["locationId"] == 101].iloc[0]
    assert np.isclose(r101["sent_mean"], 4.5)
    assert np.isclose(r101["sent_mean_corrected"], 10.0 / 3.0)
    assert np.isclose(r101["pct_ironic"], 100.0 / 3.0)

    # Vérifications pour le Restaurant 102
    # Review 4: Ironique (class 4 >= 4, rating 2 <= 2) -> corrigé = 2.0, is_ironic = True
    # Review 5: Non Ironique (class 1 < 4, rating 4 > 2) -> corrigé = 1.5, is_ironic = False
    # Moyenne attendue corrigée : (2.0 + 1.5) / 2 = 1.75
    # Taux d'ironie attendu : 1 / 2 = 50.0%
    r102 = aggregated[aggregated["locationId"] == 102].iloc[0]
    assert np.isclose(r102["sent_mean"], 2.65)
    assert np.isclose(r102["sent_mean_corrected"], 1.75)
    assert np.isclose(r102["pct_ironic"], 50.0)
