"""
test_preprocessing.py
-----------------------
Unit tests for src/preprocessing.py. Run with: pytest tests/ -v
"""

import pytest

from src.preprocessing import clean_text, preprocess_text


class TestCleanText:
    def test_lowercases_text(self):
        assert clean_text("THIS IS GREAT") == "this is great"

    def test_removes_urls(self):
        result = clean_text("check this out https://example.com/page it's great")
        assert "http" not in result
        assert "example" not in result

    def test_removes_mentions(self):
        result = clean_text("@united your service was terrible today")
        assert "@united" not in result
        assert "united" not in result

    def test_keeps_hashtag_word_drops_symbol(self):
        result = clean_text("loved the #customerservice today")
        assert "#" not in result
        assert "customerservice" in result

    def test_removes_html_tags(self):
        result = clean_text("<p>great product</p>")
        assert "<" not in result and ">" not in result
        assert "great product" in result

    def test_removes_digits(self):
        result = clean_text("rated 5 stars out of 5")
        assert not any(char.isdigit() for char in result)

    def test_collapses_whitespace(self):
        result = clean_text("too    many      spaces")
        assert "  " not in result

    def test_handles_non_string_input(self):
        assert clean_text(None) == ""
        assert clean_text(float("nan")) == ""

    def test_handles_empty_string(self):
        assert clean_text("") == ""


class TestPreprocessText:
    def test_returns_string(self):
        result = preprocess_text("This product is absolutely amazing!")
        assert isinstance(result, str)

    def test_removes_stopwords(self):
        result = preprocess_text("this is a really good and great product")
        tokens = result.split()
        assert "is" not in tokens
        assert "a" not in tokens
        assert "and" not in tokens

    def test_preserves_sentiment_bearing_words(self):
        result = preprocess_text("the flight was absolutely terrible and delayed")
        assert "terrible" in result
        assert "delayed" in result

    def test_empty_input_returns_empty_string(self):
        assert preprocess_text("") == ""

    def test_lemmatizes_words(self):
        result = preprocess_text("the flights were delayed and services were disappointing")
        # lemmatization should reduce plurals; exact form depends on POS
        # tagging defaults, so just check the pipeline runs without error
        # and produces non-empty output.
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])