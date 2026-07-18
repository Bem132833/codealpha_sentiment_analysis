"""
feature_engineering.py
------------------------
Converts cleaned text into numerical features using TF-IDF, and produces
a stratified train/test split. TF-IDF (rather than a neural embedding) is
a deliberate choice here: it is fast to train, easy to interpret (you can
inspect which n-grams drive each class), and performs competitively with
deep learning on datasets of this size (tens of thousands of rows).
"""

from typing import Tuple

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import config
from src.utils import get_logger, save_pickle, timeit

logger = get_logger(__name__)


@timeit
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on label so class balance is preserved in both sets."""
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df["label"],
    )
    logger.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


@timeit
def fit_vectorizer(train_texts: pd.Series) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    vectorizer.fit(train_texts)
    logger.info(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,}")
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: pd.Series) -> csr_matrix:
    return vectorizer.transform(texts)


def fit_label_encoder(labels: pd.Series) -> LabelEncoder:
    encoder = LabelEncoder()
    # Fit on the fixed label set (not just what's present) so encoding is
    # stable even if a class is briefly absent from a small split.
    encoder.fit(list(config.LABELS))
    return encoder


@timeit
def build_features(df: pd.DataFrame):
    """
    Full feature engineering pipeline: split -> fit vectorizer on train ->
    transform both splits -> encode labels. Persists the vectorizer and
    label encoder to models/ so predict.py can reuse them at inference time.
    """
    train_df, test_df = split_data(df)

    vectorizer = fit_vectorizer(train_df["clean_text"])
    X_train = transform_texts(vectorizer, train_df["clean_text"])
    X_test = transform_texts(vectorizer, test_df["clean_text"])

    encoder = fit_label_encoder(df["label"])
    y_train = encoder.transform(train_df["label"])
    y_test = encoder.transform(test_df["label"])

    save_pickle(vectorizer, config.VECTORIZER_PATH)
    save_pickle(encoder, config.LABEL_ENCODER_PATH)
    logger.info(f"Saved vectorizer to {config.VECTORIZER_PATH}")
    logger.info(f"Saved label encoder to {config.LABEL_ENCODER_PATH}")

    return X_train, X_test, y_train, y_test, train_df, test_df, vectorizer, encoder


if __name__ == "__main__":
    if not config.PROCESSED_FILE.exists():
        logger.error(
            f"{config.PROCESSED_FILE} not found. Run data_loader.py then "
            f"preprocessing.py first."
        )
    else:
        data = pd.read_csv(config.PROCESSED_FILE)
        build_features(data)
