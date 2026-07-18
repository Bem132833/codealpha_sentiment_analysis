"""
clean_reviews.py
------------------
Cleans a raw Amazon (or similar) reviews CSV: normalizes text, removes
HTML/duplicates/empty rows, and derives a 3-class sentiment label from
the star rating.

Expects a CSV with at least "Review Text" and "Rating" columns.
"""

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

RAW_FILE = Path("data/raw/amazon_reviews.csv")
CLEAN_FILE = Path("data/processed/amazon_reviews_clean.csv")

TEXT_COLUMN = "Review Text"
RATING_COLUMN = "Rating"

_HTML_TAG_RE = re.compile(r"<.*?>")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """
    Clean a single review's text:
      - handle missing values
      - normalize Unicode (fixes ambiguous/lookalike characters)
      - strip HTML tags
      - lowercase
      - collapse repeated whitespace
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG_RE.sub("", text)
    text = text.lower()
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


_RATING_NUMBER_RE = re.compile(r"[\d.]+")


def rating_to_sentiment(rating) -> str:
    if pd.isna(rating):
        return "neutral"

    try:
        if isinstance(rating, str):
            match = _RATING_NUMBER_RE.search(rating)
            if not match:
                return "neutral"
            rating = float(match.group())
        else:
            rating = float(rating)
    except (ValueError, TypeError):
        return "neutral"

    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


def clean_dataset(raw_path: Path = RAW_FILE, clean_path: Path = CLEAN_FILE) -> pd.DataFrame:
    """Load, clean, label, and save the reviews dataset. Returns the cleaned DataFrame."""
    logger.info(f"Loading dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    logger.info(f"Original shape: {df.shape}")

    missing_cols = [c for c in (TEXT_COLUMN, RATING_COLUMN) if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected column(s) {missing_cols}. Found columns: {list(df.columns)}"
        )

    # Remove exact duplicate rows and rows with no review text
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TEXT_COLUMN])
    logger.info(f"Dropped {before - len(df)} duplicate/empty rows")

    # Clean text and derive sentiment label
    df["clean_review"] = df[TEXT_COLUMN].apply(clean_text)
    df = df[df["clean_review"].str.strip().astype(bool)].reset_index(drop=True)
    df["sentiment"] = df[RATING_COLUMN].apply(rating_to_sentiment)

    logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False, encoding="utf-8")

    logger.info("Cleaning completed!")
    logger.info(f"Cleaned shape: {df.shape}")
    logger.info(f"Saved to: {clean_path}")

    return df


if __name__ == "__main__":
    clean_dataset()