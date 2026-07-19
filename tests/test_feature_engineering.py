"""
test_feature_engineering.py
------------------------------
Unit tests for src/feature_engineering.py. Run with: pytest tests/ -v
"""

import pandas as pd
import pytest

from src import config
from src.feature_engineering import (
    fit_label_encoder,
    fit_vectorizer,
    split_data,
    transform_texts,
)


@pytest.fixture
def sample_df():
    rows = []
    texts_by_label = {
        "positive": ["great product love it", "amazing quality highly recommend",
                     "fantastic experience will buy again", "excellent service very happy"],
        "negative": ["terrible product broke fast", "awful experience never again",
                     "poor quality very disappointed", "worst purchase ever made"],
        "neutral": ["item arrived on schedule", "product comes in three sizes",
                    "delivery took the expected time", "standard packaging as described"],
    }
    for label, texts in texts_by_label.items():
        for t in texts:
            rows.append({"text": t, "clean_text": t, "label": label, "domain": "test_domain"})
    return pd.DataFrame(rows)


class TestSplitData:
    def test_split_preserves_all_rows(self, sample_df):
        train_df, test_df = split_data(sample_df)
        assert len(train_df) + len(test_df) == len(sample_df)

    def test_split_is_stratified(self, sample_df):
        train_df, test_df = split_data(sample_df)
        # every class present in the full set should appear in train
        assert set(train_df["label"].unique()) == set(sample_df["label"].unique())

    def test_split_ratio_approximately_correct(self, sample_df):
        train_df, test_df = split_data(sample_df)
        expected_test_size = round(len(sample_df) * config.TEST_SIZE)
        assert abs(len(test_df) - expected_test_size) <= 2


class TestVectorizer:
    def test_fit_vectorizer_builds_vocabulary(self, sample_df):
        vectorizer = fit_vectorizer(sample_df["clean_text"])
        assert len(vectorizer.vocabulary_) > 0

    def test_transform_produces_correct_shape(self, sample_df):
        vectorizer = fit_vectorizer(sample_df["clean_text"])
        matrix = transform_texts(vectorizer, sample_df["clean_text"])
        assert matrix.shape[0] == len(sample_df)
        assert matrix.shape[1] == len(vectorizer.vocabulary_)

    def test_transform_on_unseen_text_does_not_error(self, sample_df):
        vectorizer = fit_vectorizer(sample_df["clean_text"])
        matrix = transform_texts(vectorizer, pd.Series(["completely new unseen sentence"]))
        assert matrix.shape[0] == 1


class TestLabelEncoder:
    def test_encoder_covers_all_project_labels(self, sample_df):
        encoder = fit_label_encoder(sample_df["label"])
        assert set(encoder.classes_) == set(config.LABELS)

    def test_encode_decode_round_trip(self, sample_df):
        encoder = fit_label_encoder(sample_df["label"])
        encoded = encoder.transform(["positive", "negative", "neutral"])
        decoded = encoder.inverse_transform(encoded)
        assert list(decoded) == ["positive", "negative", "neutral"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
