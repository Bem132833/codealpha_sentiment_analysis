"""
data_loader.py
---------------
Loads the three source datasets (Amazon reviews, Twitter Airline
Sentiment, Financial PhraseBank), maps each one's original label scheme
onto the project's unified schema (negative / neutral / positive), tags
every row with its source domain, and concatenates everything into a
single DataFrame.

If a raw file is missing, a small synthetic sample is generated instead
so the rest of the pipeline (preprocessing, feature engineering, training,
evaluation) can be run and tested end-to-end before you've downloaded the
real datasets. Replace the synthetic data by dropping the real CSVs into
data/raw/<domain>/ using the filenames in config.py.

Dataset sources (download manually — see docs/dataset_description.md):
  - Amazon Reviews:        Kaggle "Amazon Reviews for Sentiment Analysis"
  - Twitter Airline:       Kaggle "Twitter US Airline Sentiment"
  - Financial PhraseBank:  Kaggle / HuggingFace "financial_phrasebank"
"""

import random
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)

random.seed(config.RANDOM_STATE)


# ---------------------------------------------------------------------------
# Synthetic fallback generation
# ---------------------------------------------------------------------------

_POS_SNIPPETS = [
    "absolutely love this, works perfectly and exceeded expectations",
    "great experience overall, would definitely recommend to others",
    "fantastic quality, fast delivery, very happy with the purchase",
    "excellent service and the staff were extremely helpful",
    "this made my day, couldn't be more satisfied",
    "impressive results, clearly well made and worth the price",
]
_NEG_SNIPPETS = [
    "terrible experience, would not recommend this at all",
    "very disappointed, broke after just two days of use",
    "worst customer service I have ever dealt with",
    "waste of money, does not work as advertised",
    "flight was delayed for hours with no explanation",
    "poor quality control, arrived damaged and unusable",
]
_NEU_SNIPPETS = [
    "the item arrived on the scheduled delivery date",
    "the company reported quarterly earnings in line with estimates",
    "the meeting has been rescheduled to next Thursday",
    "the product comes in three different sizes",
    "the report outlines standard procedures for the department",
    "the flight departed and landed at the listed times",
]


def _synthetic_domain(domain: str, n_rows: int) -> pd.DataFrame:
    """Generate a small labeled synthetic sample for one domain."""
    rows = []
    for _ in range(n_rows):
        label = random.choice(config.LABELS)
        snippet = {
            "positive": random.choice(_POS_SNIPPETS),
            "negative": random.choice(_NEG_SNIPPETS),
            "neutral": random.choice(_NEU_SNIPPETS),
        }[label]
        rows.append({"text": snippet, "label": label, "domain": domain})
    logger.warning(
        f"Raw file for '{domain}' not found — generated {n_rows} synthetic "
        f"rows instead. Download the real dataset to replace this."
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-domain loaders
# ---------------------------------------------------------------------------


def load_amazon_reviews(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Amazon product reviews and map star ratings to sentiment:
      1-2 stars -> negative, 3 stars -> neutral, 4-5 stars -> positive.
    Expected columns (Kaggle "Amazon Reviews for Sentiment Analysis"):
      'reviewText' (or 'text') and 'overall' (or 'rating' / 'score').
    """
    path = path or config.AMAZON_RAW_FILE
    if not path.exists():
        return _synthetic_domain("amazon_reviews", config.SYNTHETIC_ROWS_PER_DOMAIN)

    df = pd.read_csv(path)
    text_col = next(
        (c for c in ("reviewText", "text", "review_text") if c in df.columns), None
    )
    rating_col = next(
        (c for c in ("overall", "rating", "score", "stars") if c in df.columns), None
    )
    if text_col is None or rating_col is None:
        raise ValueError(
            f"amazon_reviews.csv is missing expected columns. Found: {list(df.columns)}. "
            f"Expected a text column (reviewText/text/review_text) and a rating "
            f"column (overall/rating/score/stars)."
        )

    df = df[[text_col, rating_col]].rename(
        columns={text_col: "text", rating_col: "rating"}
    )
    df = df.dropna(subset=["text", "rating"])

    if config.AMAZON_SAMPLE_SIZE and len(df) > config.AMAZON_SAMPLE_SIZE:
        df = df.sample(n=config.AMAZON_SAMPLE_SIZE, random_state=config.RANDOM_STATE)

    def rating_to_label(r: float) -> str:
        if r <= 2:
            return "negative"
        elif r == 3:
            return "neutral"
        return "positive"

    df["label"] = df["rating"].astype(float).apply(rating_to_label)
    df["domain"] = "amazon_reviews"
    return df[["text", "label", "domain"]].reset_index(drop=True)


def load_twitter_airline(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Twitter US Airline Sentiment.
    Expected columns: 'text' and 'airline_sentiment'
    (values already 'positive' / 'negative' / 'neutral').
    """
    path = path or config.TWITTER_RAW_FILE
    if not path.exists():
        return _synthetic_domain("twitter_airline", config.SYNTHETIC_ROWS_PER_DOMAIN)

    df = pd.read_csv(path)
    text_col = next((c for c in ("text", "tweet") if c in df.columns), None)
    label_col = next(
        (c for c in ("airline_sentiment", "sentiment", "label") if c in df.columns),
        None,
    )
    if text_col is None or label_col is None:
        raise ValueError(
            f"twitter_airline_sentiment.csv is missing expected columns. "
            f"Found: {list(df.columns)}. Expected 'text' and 'airline_sentiment'."
        )

    df = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(config.LABELS)]

    if config.TWITTER_SAMPLE_SIZE and len(df) > config.TWITTER_SAMPLE_SIZE:
        df = df.sample(n=config.TWITTER_SAMPLE_SIZE, random_state=config.RANDOM_STATE)

    df["domain"] = "twitter_airline"
    return df[["text", "label", "domain"]].reset_index(drop=True)


def load_financial_phrasebank(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Financial PhraseBank.
    Expected columns: 'sentence' (or 'text') and 'label'
    (values 'positive' / 'negative' / 'neutral').
    """
    path = path or config.FINANCIAL_RAW_FILE
    if not path.exists():
        return _synthetic_domain(
            "financial_phrasebank", config.SYNTHETIC_ROWS_PER_DOMAIN
        )

    df = pd.read_csv(path)
    text_col = next((c for c in ("sentence", "text") if c in df.columns), None)
    label_col = next((c for c in ("label", "sentiment") if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"financial_phrasebank.csv is missing expected columns. "
            f"Found: {list(df.columns)}. Expected 'sentence' and 'label'."
        )

    df = df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(config.LABELS)]

    if config.FINANCIAL_SAMPLE_SIZE and len(df) > config.FINANCIAL_SAMPLE_SIZE:
        df = df.sample(n=config.FINANCIAL_SAMPLE_SIZE, random_state=config.RANDOM_STATE)

    df["domain"] = "financial_phrasebank"
    return df[["text", "label", "domain"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------


@timeit
def load_all_domains() -> pd.DataFrame:
    """Load, unify, and concatenate all three domains into one DataFrame."""
    config.ensure_directories()

    loaders = {
        "amazon_reviews": load_amazon_reviews,
        "twitter_airline": load_twitter_airline,
        "financial_phrasebank": load_financial_phrasebank,
    }

    frames = []
    for domain, loader_fn in loaders.items():
        df = loader_fn()
        logger.info(f"Loaded {len(df):,} rows from '{domain}'")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(
        f"Combined dataset: {len(combined):,} rows across {len(loaders)} domains"
    )
    logger.info(f"Class balance:\n{combined['label'].value_counts()}")
    return combined


if __name__ == "__main__":
    data = load_all_domains()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.PROCESSED_FILE, index=False)
    logger.info(f"Saved unified dataset to {config.PROCESSED_FILE}")
