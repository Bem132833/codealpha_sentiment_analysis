"""
config.py
---------
Central configuration for the CodeAlpha Sentiment Analysis project.

All file paths, dataset schemas, and hyperparameters live here so that no
other module hardcodes a path or a magic number. If you move a folder or
swap a dataset, this is the only file you should need to touch.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & directory layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

# ---------------------------------------------------------------------------
# Raw dataset locations (per domain)
# ---------------------------------------------------------------------------
AMAZON_RAW_DIR = RAW_DIR / "amazon_reviews"
TWITTER_RAW_DIR = RAW_DIR / "twitter_airline"
FINANCIAL_RAW_DIR = RAW_DIR / "financial_phrasebank"

# Expected raw filenames. If your Kaggle/HuggingFace download uses a
# different filename, either rename the file or update these constants.
AMAZON_RAW_FILE = AMAZON_RAW_DIR / "amazon_reviews.csv"
TWITTER_RAW_FILE = TWITTER_RAW_DIR / "twitter_airline_sentiment.csv"
FINANCIAL_RAW_FILE = FINANCIAL_RAW_DIR / "financial_phrasebank.csv"

# Unified processed dataset (all three domains combined & cleaned)
PROCESSED_FILE = PROCESSED_DIR / "unified_sentiment_dataset.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Output artifacts
# ---------------------------------------------------------------------------
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.csv"
METRICS_FILE = METRICS_DIR / "metrics.json"
CROSS_DOMAIN_METRICS_FILE = METRICS_DIR / "cross_domain_metrics.json"
LOG_FILE = LOGS_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Domain / label schema
# ---------------------------------------------------------------------------
DOMAINS = ("amazon_reviews", "twitter_airline", "financial_phrasebank")
LABELS = ("negative", "neutral", "positive")
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Preprocessing settings
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_TOKEN_LENGTH = 2
REMOVE_STOPWORDS = True
APPLY_LEMMATIZATION = True

# ---------------------------------------------------------------------------
# Feature engineering settings (TF-IDF)
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 15000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
# Two candidate models are trained and compared; the better one (by
# macro-F1 on the held-out test set) is persisted as the final model.
MODEL_CANDIDATES = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "linear_svm": {
        "C": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "max_iter": 5000,
    },
}

# ---------------------------------------------------------------------------
# Sampling caps (keep the project runnable end-to-end on a laptop)
# ---------------------------------------------------------------------------
AMAZON_SAMPLE_SIZE = 20000     # Amazon reviews is huge upstream; sample down
TWITTER_SAMPLE_SIZE = None     # Twitter Airline is already small; use all
FINANCIAL_SAMPLE_SIZE = None   # Financial PhraseBank is already small; use all

# ---------------------------------------------------------------------------
# Synthetic fallback data (lets the pipeline run before real data is
# downloaded, e.g. for CI or a first local smoke test)
# ---------------------------------------------------------------------------
GENERATE_SYNTHETIC_IF_MISSING = True
SYNTHETIC_ROWS_PER_DOMAIN = 150


def ensure_directories() -> None:
    """Create every directory this project writes into, if missing."""
    for directory in (
        RAW_DIR, AMAZON_RAW_DIR, TWITTER_RAW_DIR, FINANCIAL_RAW_DIR,
        PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, OUTPUTS_DIR,
        FIGURES_DIR, REPORTS_DIR, METRICS_DIR, LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
