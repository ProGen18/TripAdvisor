import sys
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sentiment_transformer import (
    SUPPORTED_LANGS,
    split_reviews_by_language,
    compute_sentiment_metrics,
)

def test_routing_split():
    # Create mock reviews
    df = pd.DataFrame({
        "review_id": [1, 2, 3, 4],
        "language": ["fr", "ru", "en", "ja"],
        "text": ["Excellent", "Плохо", "Good", "美味しい"]
    })
    
    # Perform splitting
    df_primary, df_fallback = split_reviews_by_language(df)
    
    assert len(df_primary) == 2
    assert set(df_primary["language"]) == {"fr", "en"}
    assert len(df_fallback) == 2
    assert set(df_fallback["language"]) == {"ru", "ja"}

def test_routing_missing_language():
    # Create mock reviews missing the language column
    df = pd.DataFrame({
        "review_id": [1, 2],
        "text": ["Excellent", "Good"]
    })
    
    # Safe handling check
    df_primary, df_fallback = split_reviews_by_language(df)
        
    assert len(df_primary) == 2
    assert len(df_fallback) == 0

def test_output_math_consistency():
    # Both models output probabilities for 5 classes (argmax + 1 = class, expected = dot product)
    probs = np.array([
        [0.1, 0.1, 0.1, 0.2, 0.5], # expected: 3.9 stars, class: 5
        [0.8, 0.1, 0.1, 0.0, 0.0]  # expected: 1.3 stars, class: 1
    ])
    
    expected, predicted_class, confidence = compute_sentiment_metrics(probs)
    
    assert np.allclose(expected, [3.9, 1.3])
    assert np.array_equal(predicted_class, [5, 1])
    assert np.allclose(confidence, [0.5, 0.8])

def test_score_reviews_irony_and_correction():
    from unittest.mock import patch, MagicMock
    from scripts.sentiment_transformer import score_reviews
    
    with patch("scripts.sentiment_transformer.score_batch") as mock_score_batch, \
         patch("scripts.sentiment_transformer.save_cache") as mock_save_cache:
         
        mock_score_batch.return_value = np.array([
            [0.0, 0.05, 0.05, 0.1, 0.8], # expected: 4.65 stars, class: 5
            [0.8, 0.1, 0.1, 0.0, 0.0]    # expected: 1.3 stars, class: 1
        ])
        
        def dummy_save_cache(scored, existing, path=None):
            return pd.concat([existing, scored], ignore_index=True)
        mock_save_cache.side_effect = dummy_save_cache
        
        df = pd.DataFrame({
            "review_id": [1, 2],
            "locationId": [101, 101],
            "text": ["Very good!", "Bad"],
            "rating": [1.0, 4.0],
            "language": ["en", "en"]
        })
        
        res = score_reviews(
            reviews=df,
            tokenizer=MagicMock(),
            model=MagicMock(),
            device="cpu",
            batch_size=2,
            save_every=10
        )
        
        assert len(res) == 2
        
        # Review 1 check (ironic)
        r1 = res[res["review_id"] == 1].iloc[0]
        assert r1["is_ironic"] == True
        assert r1["sent_stars_corrected"] == 1.0
        assert r1["sent_class_corrected"] == 1
        
        # Review 2 check (not ironic)
        r2 = res[res["review_id"] == 2].iloc[0]
        assert r2["is_ironic"] == False
        assert np.isclose(r2["sent_stars_corrected"], 1.3)
        assert r2["sent_class_corrected"] == 1
