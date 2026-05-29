"""Pipeline de sentiment par transformer multilingue.

Modele : nlptown/bert-base-multilingual-uncased-sentiment
- Pre-entraine sur des avis produit en 6 langues (FR, EN, IT, ES, PT, DE)
- Sort une classe 1-5 etoiles (directement comparable au rating TripAdvisor)
- Couvre 97% des avis du dataset Paris (FR+EN+IT+ES+PT+DE)

Strategie:
- Batching (defaut 32) pour vitesse maximale sur CPU
- Cache parquet par review_id : reprise possible apres interruption
- Mode --sample N : test rapide avant full run
- Mode --resume : ne re-score que les avis non encore traites

Entree : data/raw/all_reviews.parquet (colonnes locationId, review_id, text, rating, language)
Sortie :
- data/stats/sentiment_reviews.parquet      (par avis : score 1-5 + proba + classe predite)
- data/stats/sentiment_restaurant_v2.parquet (agrege par restaurant)

Usage :
    python scripts/sentiment_transformer.py --sample 1000     # test
    python scripts/sentiment_transformer.py --resume          # full incremental
    python scripts/sentiment_transformer.py --batch-size 64   # plus rapide si RAM ok
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import STATS_DIR  # noqa: E402
from scripts.data_loader import aggregate_sentiment_reviews  # noqa: E402

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
FALLBACK_MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
SUPPORTED_LANGS = {"fr", "en", "it", "es", "pt", "de", "nl"}
MAX_LEN = 128  # tokens — couvre ~85% des avis sans truncation excessive; vitesse x2-3 vs 256
RAW_REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "all_reviews.parquet"
OUT_REVIEWS_PATH = STATS_DIR / "sentiment_reviews.parquet"
OUT_RESTAURANT_PATH = STATS_DIR / "sentiment_restaurant_v2.parquet"


def get_output_paths(suffix: str) -> tuple[Path, Path]:
    """Construit les chemins de sortie avec suffixe optionnel pour workers paralleles."""
    if suffix:
        rev = STATS_DIR / f"sentiment_reviews{suffix}.parquet"
        agg = STATS_DIR / f"sentiment_restaurant_v2{suffix}.parquet"
        return rev, agg
    return OUT_REVIEWS_PATH, OUT_RESTAURANT_PATH


def load_model(model_name: str, device: str, quantize: bool = False):
    """Charge le tokenizer et le modele en eval mode."""
    print(f"  Chargement modele {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    if quantize and device == "cpu":
        print("  Quantization dynamique int8 (Linear layers)...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    print(f"  Device : {device}{' (quantized)' if quantize else ''}")
    return tokenizer, model


@torch.no_grad()
def score_batch(texts: list[str], tokenizer, model, device: str) -> np.ndarray:
    """Score un batch de textes. Retourne array (N, 5) de probabilites."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs


def filter_supported_languages(df: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les avis dans une langue supportee par le modele."""
    if "language" not in df.columns:
        return df
    before = len(df)
    df = df[df["language"].isin(SUPPORTED_LANGS)].copy()
    print(f"  Filtre langues supportees : {len(df):,}/{before:,} ({len(df)/before*100:.1f}%)")
    return df


def split_reviews_by_language(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits reviews into primary (supported) and fallback subsets.
    If the language column is missing, all reviews are treated as primary.
    """
    if "language" not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns)
    primary_mask = df["language"].isin(SUPPORTED_LANGS)
    return df[primary_mask].copy(), df[~primary_mask].copy()


def compute_sentiment_metrics(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates expected stars, predicted class, and confidence from probabilities.
    
    Returns:
        tuple containing:
            - expected: expected stars (float rating from 1.0 to 5.0)
            - predicted_class: argmax class + 1 (int rating from 1 to 5)
            - confidence: max probability
    """
    stars = np.arange(1, 6)
    expected = (probs * stars).sum(axis=1)
    predicted_class = probs.argmax(axis=1) + 1
    confidence = probs.max(axis=1)
    return expected, predicted_class, confidence


def load_existing_cache(path: Path | None = None) -> pd.DataFrame:
    """Charge le cache des avis deja scores."""
    p = path or OUT_REVIEWS_PATH
    if p.exists():
        cache = pd.read_parquet(p)
        print(f"  Cache existant ({p.name}) : {len(cache):,} avis deja scores")
        return cache
    return pd.DataFrame()


def load_global_done_ids() -> set:
    """Charge tous les review_id deja scores (main + parts) pour eviter les doublons."""
    done = set()
    for p in sorted(STATS_DIR.glob("sentiment_reviews*.parquet")):
        try:
            done.update(pd.read_parquet(p, columns=["review_id"])["review_id"].tolist())
        except Exception:
            pass
    return done


def save_cache(scored: pd.DataFrame, existing: pd.DataFrame, path: Path | None = None):
    """Concatene et sauve le cache complet."""
    p = path or OUT_REVIEWS_PATH
    full = pd.concat([existing, scored], ignore_index=True)
    full = full.drop_duplicates(subset=["review_id"], keep="last")
    p.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(p, index=False)
    return full


def score_reviews(
    reviews: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    batch_size: int = 32,
    save_every: int = 5000,
    existing_cache: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Score tous les avis avec checkpoints periodiques."""
    if existing_cache is None:
        existing_cache = pd.DataFrame()

    all_scores = []
    n_since_save = 0
    pbar = tqdm(total=len(reviews), desc="Scoring", unit="avis")

    texts = reviews["text"].astype(str).tolist()
    ids = reviews["review_id"].tolist()
    loc_ids = reviews["locationId"].tolist()
    ratings = reviews["rating"].tolist() if "rating" in reviews.columns else [None] * len(reviews)
    langs = reviews["language"].tolist() if "language" in reviews.columns else ["?"] * len(reviews)

    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            probs = score_batch(batch_texts, tokenizer, model, device)
        except Exception as e:
            print(f"\n  Erreur batch i={i}: {e}. Saute.")
            pbar.update(len(batch_texts))
            continue

        # Score = E[note] = sum(p_k * (k+1))
        expected, predicted_class, confidence = compute_sentiment_metrics(probs)

        for j, (rev_id, loc_id, rating, lang) in enumerate(
            zip(
                ids[i : i + batch_size],
                loc_ids[i : i + batch_size],
                ratings[i : i + batch_size],
                langs[i : i + batch_size],
            )
        ):
            sent_class_val = int(predicted_class[j])
            expected_val = float(expected[j])
            
            is_ironic = False
            if rating is not None and not pd.isna(rating):
                is_ironic = (sent_class_val >= 4) and (rating <= 2)
            
            sent_stars_corrected = float(rating) if is_ironic else expected_val
            sent_class_corrected = int(rating) if is_ironic else sent_class_val

            all_scores.append({
                "review_id": rev_id,
                "locationId": loc_id,
                "rating": rating,
                "language": lang,
                "sent_stars_expected": expected_val,
                "sent_class": sent_class_val,
                "sent_confidence": float(confidence[j]),
                "p_1star": float(probs[j, 0]),
                "p_2star": float(probs[j, 1]),
                "p_3star": float(probs[j, 2]),
                "p_4star": float(probs[j, 3]),
                "p_5star": float(probs[j, 4]),
                "is_ironic": is_ironic,
                "sent_stars_corrected": sent_stars_corrected,
                "sent_class_corrected": sent_class_corrected,
            })

        pbar.update(len(batch_texts))
        n_since_save += len(batch_texts)

        # Checkpoint
        if n_since_save >= save_every:
            partial = pd.DataFrame(all_scores)
            existing_cache = save_cache(partial, existing_cache, path=output_path)
            all_scores = []
            n_since_save = 0
            elapsed = time.time() - t0
            rate = (i + batch_size) / max(elapsed, 1)
            pbar.set_postfix({"rate": f"{rate:.0f} avis/s", "cache": f"{len(existing_cache):,}"})

    pbar.close()

    # Final save
    if all_scores:
        partial = pd.DataFrame(all_scores)
        existing_cache = save_cache(partial, existing_cache, path=output_path)

    return existing_cache


def aggregate_by_restaurant(scored: pd.DataFrame) -> pd.DataFrame:
    """Agrege les scores par restaurant."""
    print("\n  Agregation par restaurant...")
    return aggregate_sentiment_reviews(scored)


def main(
    sample: int | None = None,
    batch_size: int = 32,
    resume: bool = True,
    save_every: int = 5000,
    quantize: bool = False,
    threads: int | None = None,
    chunk_id: int = 0,
    num_chunks: int = 1,
    output_suffix: str = "",
    input_path: str | None = None,
    max_len: int = 128,
):
    global MAX_LEN
    if max_len is not None:
        MAX_LEN = max_len
        
    out_reviews_path, out_restaurant_path = get_output_paths(output_suffix)
    is_worker = num_chunks > 1

    tag = f"[worker {chunk_id}/{num_chunks}] " if is_worker else ""
    print("=" * 60)
    print(f"{tag}PIPELINE SENTIMENT TRANSFORMER (nlptown multilingual)")
    print("=" * 60)

    # Device et threads
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if threads and device == "cpu":
        torch.set_num_threads(threads)
        print(f"  {tag}Threads PyTorch : {threads}")

    # Donnees
    print("\n1. Chargement des avis...")
    in_path = Path(input_path) if input_path else RAW_REVIEWS_PATH
    if not in_path.exists():
        print(f"  ERREUR : {in_path} introuvable.")
        return
    reviews = pd.read_parquet(in_path)
    print(f"  Total : {len(reviews):,} avis · {reviews['locationId'].nunique():,} restos")

    if "language" in reviews.columns:
        primary_count = reviews["language"].isin(SUPPORTED_LANGS).sum()
        fallback_count = len(reviews) - primary_count
        print(f"  Répartition langues initiales : {primary_count:,} principales (nlptown), {fallback_count:,} replis (tabularisai)")

    # Nettoyage texte
    reviews = reviews[reviews["text"].astype(str).str.len() > 5].copy()
    print(f"  Avis avec texte non vide : {len(reviews):,}")

    # Trier par review_id pour que le partitionnement par modulo soit deterministe
    reviews = reviews.sort_values("review_id").reset_index(drop=True)

    # Resume : exclure tous les avis deja scores (main + parts d'autres workers)
    existing = load_existing_cache(out_reviews_path) if resume else pd.DataFrame()
    if resume and not existing.empty:
        if (
            "is_ironic" not in existing.columns
            or "sent_stars_corrected" not in existing.columns
            or "sent_class_corrected" not in existing.columns
            or existing["is_ironic"].isna().any()
        ):
            print("  Enrichissement dynamique du cache existant avec les colonnes d'ironie...")
            existing = existing.copy()
            if "rating" in existing.columns:
                is_ironic_mask = (existing["sent_class"] >= 4) & (
                    existing["rating"].fillna(99.0) <= 2
                )
                existing["is_ironic"] = is_ironic_mask
                existing["sent_stars_corrected"] = np.where(
                    is_ironic_mask,
                    existing["rating"],
                    existing["sent_stars_expected"],
                )
                existing["sent_class_corrected"] = np.where(
                    is_ironic_mask, existing["rating"], existing["sent_class"]
                )
                existing["sent_class_corrected"] = existing[
                    "sent_class_corrected"
                ].astype("Int64")
            else:
                existing["is_ironic"] = False
                existing["sent_stars_corrected"] = existing["sent_stars_expected"]
                existing["sent_class_corrected"] = existing["sent_class"].astype(
                    "Int64"
                )
    if resume:
        all_done_ids = load_global_done_ids()
        if all_done_ids:
            before = len(reviews)
            reviews = reviews[~reviews["review_id"].isin(all_done_ids)]
            print(f"  Apres exclusion (cache global) : {len(reviews):,} a scorer (skip {before - len(reviews):,})")

    # Partitionnement par chunk (apres resume pour repartition equitable du restant)
    if is_worker:
        before = len(reviews)
        reviews = reviews.iloc[chunk_id::num_chunks].reset_index(drop=True)
        print(f"  {tag}Chunk : {len(reviews):,} / {before:,} avis")

    # Sample
    if sample is not None and sample < len(reviews):
        reviews = reviews.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"  Echantillon : {len(reviews):,} avis")

    if reviews.empty:
        print("  Rien a scorer.")
        if not existing.empty and not is_worker:
            print("\n2. Reagregation par restaurant...")
            agg = aggregate_by_restaurant(existing)
            agg.to_parquet(out_restaurant_path, index=False)
            print(f"  -> {out_restaurant_path}")
        return

    # Split reviews based on language
    reviews_primary, reviews_fallback = split_reviews_by_language(reviews)

    full_scored = existing

    # 1. Score primary reviews
    if not reviews_primary.empty:
        print("\n" + "=" * 50)
        print(f"DEBUT SCORING PRINCIPAL ({MODEL_NAME})")
        print(f"Nombre d'avis : {len(reviews_primary):,} · batch={batch_size}")
        print("=" * 50)
        
        tokenizer, model = load_model(MODEL_NAME, device, quantize=quantize)
        t0 = time.time()
        full_scored = score_reviews(
            reviews_primary, tokenizer, model, device,
            batch_size=batch_size, save_every=save_every,
            existing_cache=full_scored,
            output_path=out_reviews_path,
        )
        elapsed = time.time() - t0
        rate = len(reviews_primary) / max(elapsed, 1)
        print(f"  Termine scoring principal en {elapsed/60:.1f} min ({rate:.0f} avis/s)")
        
        # Free memory
        del model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # 2. Score fallback reviews
    if not reviews_fallback.empty:
        print("\n" + "=" * 50)
        print(f"DEBUT SCORING REPLI ({FALLBACK_MODEL_NAME})")
        print(f"Nombre d'avis : {len(reviews_fallback):,} · batch={batch_size}")
        print("=" * 50)
        
        tokenizer_fb, model_fb = load_model(FALLBACK_MODEL_NAME, device, quantize=quantize)
        t0 = time.time()
        full_scored = score_reviews(
            reviews_fallback, tokenizer_fb, model_fb, device,
            batch_size=batch_size, save_every=save_every,
            existing_cache=full_scored,
            output_path=out_reviews_path,
        )
        elapsed = time.time() - t0
        rate = len(reviews_fallback) / max(elapsed, 1)
        print(f"  Termine scoring de repli en {elapsed/60:.1f} min ({rate:.0f} avis/s)")
        
        # Free memory
        del model_fb, tokenizer_fb
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Agregation (skip si worker — sera fait apres merge)
    if is_worker:
        print(f"  {tag}Worker termine. Sortie : {out_reviews_path.name}")
        return

    print("\n4. Agregation par restaurant...")
    agg = aggregate_by_restaurant(full_scored)
    agg.to_parquet(out_restaurant_path, index=False)
    print(f"  -> {out_restaurant_path} ({len(agg):,} restos)")

    # Resume
    print("\n" + "=" * 60)
    print("RESUME")
    print("=" * 60)
    print(f"Avis scores au total      : {len(full_scored):,}")
    print(f"Restaurants couverts      : {full_scored['locationId'].nunique():,}")
    print(f"Note moyenne predite      : {full_scored['sent_stars_expected'].mean():.2f}")
    if "rating" in full_scored.columns:
        valid = full_scored.dropna(subset=["rating"])
        if not valid.empty:
            mae = (valid["sent_stars_expected"] - valid["rating"]).abs().mean()
            corr = valid[["sent_stars_expected", "rating"]].corr().iloc[0, 1]
            within_1 = ((valid["sent_stars_expected"] - valid["rating"]).abs() <= 1).mean() * 100
            print(f"MAE vs note reelle        : {mae:.3f} etoiles")
            print(f"Correlation Pearson       : {corr:.3f}")
            print(f"% predictions a +/-1 etoile: {within_1:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment transformer multilingue")
    parser.add_argument("--sample", type=int, default=None, help="Test sur N avis")
    parser.add_argument("--batch-size", type=int, default=32, help="Taille batch (defaut 32)")
    parser.add_argument("--no-resume", action="store_true", help="Ne pas reutiliser le cache")
    parser.add_argument("--save-every", type=int, default=5000, help="Checkpoint tous les N avis")
    parser.add_argument("--quantize", action="store_true", help="Quantization dynamique int8 (2-4x plus rapide sur CPU)")
    parser.add_argument("--threads", type=int, default=None, help="Nombre de threads PyTorch (defaut: auto)")
    parser.add_argument("--chunk-id", type=int, default=0, help="ID du chunk pour parallelisation (0..num_chunks-1)")
    parser.add_argument("--num-chunks", type=int, default=1, help="Nombre total de chunks (workers paralleles)")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffixe sortie (ex: _part0). Auto si num-chunks>1")
    parser.add_argument("--input-path", type=str, default=None, help="Chemin vers le fichier parquet des avis d'entrée (défaut: raw/all_reviews.parquet)")
    parser.add_argument("--max-len", type=int, default=128, help="Longueur maximale des séquences de tokens")
    args = parser.parse_args()
    suffix = args.output_suffix
    if args.num_chunks > 1 and not suffix:
        suffix = f"_part{args.chunk_id}"
    main(
        sample=args.sample,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        save_every=args.save_every,
        quantize=args.quantize,
        threads=args.threads,
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks,
        output_suffix=suffix,
        input_path=args.input_path,
        max_len=args.max_len,
    )
