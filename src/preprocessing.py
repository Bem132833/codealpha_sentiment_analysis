"""
preprocessing.py
-----------------
Text cleaning and normalization shared by every domain. Handles the
messiness specific to each source (URLs and @mentions in tweets, HTML
fragments in reviews, boilerplate in financial sentences) with one
general-purpose pipeline, then tokenizes, removes stopwords, and
lemmatizes.

Uses NLTK for tokenization, stopwords, and lemmatization. Run
`python -m src.preprocessing --download-nltk` once to fetch the required
NLTK data packages before first use.
"""

import argparse
import re
import string
from typing import List

import pandas as pd

from src import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

_lemmatizer = None
_stopwords = None


def _ensure_nltk_resources() -> None:
    """Lazily import and cache NLTK objects; fail with a clear message if data is missing."""
    global _lemmatizer, _stopwords
    if _lemmatizer is not None and _stopwords is not None:
        return

    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:
        raise ImportError(
            "nltk is required for preprocessing. Install it with "
            "`pip install nltk` (already in requirements.txt)."
        ) from exc

    try:
        _stopwords = set(stopwords.words("english"))
        _lemmatizer = WordNetLemmatizer()
        _lemmatizer.lemmatize("test")  # forces wordnet data to load, raises if missing
    except LookupError as exc:
        raise LookupError(
            "Required NLTK data is missing. Run once:\n"
            "  python -m src.preprocessing --download-nltk"
        ) from exc


def download_nltk_data() -> None:
    """One-time download of NLTK corpora needed by this module."""
    import nltk
    for package in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"):
        nltk.download(package, quiet=True)
    logger.info("NLTK data downloaded.")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/HTML/punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)  # keep hashtag word, drop the '#'
    text = text.translate(str.maketrans("", "", string.digits))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize_and_normalize(text: str) -> List[str]:
    """Tokenize cleaned text, optionally remove stopwords, and lemmatize."""
    _ensure_nltk_resources()
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if len(t) >= config.MIN_TOKEN_LENGTH]

    if config.REMOVE_STOPWORDS:
        tokens = [t for t in tokens if t not in _stopwords]

    if config.APPLY_LEMMATIZATION:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return tokens


def preprocess_text(text: str) -> str:
    """Full pipeline: clean -> tokenize -> normalize -> rejoin into a string."""
    cleaned = clean_text(text)
    tokens = tokenize_and_normalize(cleaned)
    return " ".join(tokens)


@timeit
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply preprocess_text to every row and drop rows that end up empty."""
    df = df.copy()
    _ensure_nltk_resources()

    df["clean_text"] = df[text_col].apply(preprocess_text)
    before = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows that became empty after cleaning")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text preprocessing utilities")
    parser.add_argument(
        "--download-nltk", action="store_true",
        help="Download required NLTK corpora and exit"
    )
    args = parser.parse_args()

    if args.download_nltk:
        download_nltk_data()
    else:
        data = pd.read_csv(config.PROCESSED_FILE) if config.PROCESSED_FILE.exists() else None
        if data is None:
            logger.error(
                f"{config.PROCESSED_FILE} not found. Run `python -m src.data_loader` first."
            )
        else:
            processed = preprocess_dataframe(data)
            processed.to_csv(config.PROCESSED_FILE, index=False)
            logger.info(f"Saved preprocessed dataset to {config.PROCESSED_FILE}")
