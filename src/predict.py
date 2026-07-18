"""
predict.py
----------
Loads the persisted model, vectorizer, and label encoder to run inference
on new text. Supports a single string (CLI arg), an interactive prompt,
or a batch CSV file (writes predictions to outputs/predictions.csv).

Usage:
    python -m src.predict --text "the flight was delayed and staff were rude"
    python -m src.predict --csv path/to/new_reviews.csv --text-col review
    python -m src.predict            # interactive mode
"""

import argparse

import pandas as pd

from src import config
from src.preprocessing import preprocess_text
from src.utils import get_logger, load_pickle

logger = get_logger(__name__)


class SentimentPredictor:
    def __init__(self):
        for path, label in (
            (config.MODEL_PATH, "model"),
            (config.VECTORIZER_PATH, "vectorizer"),
            (config.LABEL_ENCODER_PATH, "label encoder"),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"{label.capitalize()} not found at {path}. "
                    f"Run `python main.py` (or train.py) first."
                )

        self.model = load_pickle(config.MODEL_PATH)
        self.vectorizer = load_pickle(config.VECTORIZER_PATH)
        self.encoder = load_pickle(config.LABEL_ENCODER_PATH)

    def predict(self, text: str) -> dict:
        cleaned = preprocess_text(text)
        features = self.vectorizer.transform([cleaned])
        pred_id = self.model.predict(features)[0]
        label = self.encoder.inverse_transform([pred_id])[0]

        result = {"text": text, "clean_text": cleaned, "predicted_label": label}

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)[0]
            result["confidence_scores"] = dict(zip(self.encoder.classes_, scores.tolist()))
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            result["confidence_scores"] = dict(zip(self.encoder.classes_, probs.tolist()))

        return result

    def predict_batch(self, texts: pd.Series) -> pd.DataFrame:
        return pd.DataFrame([self.predict(t) for t in texts])


def run_cli():
    parser = argparse.ArgumentParser(description="Predict sentiment for new text")
    parser.add_argument("--text", type=str, help="A single piece of text to classify")
    parser.add_argument("--csv", type=str, help="Path to a CSV file of texts to classify")
    parser.add_argument("--text-col", type=str, default="text", help="Column name containing text (for --csv)")
    args = parser.parse_args()

    predictor = SentimentPredictor()

    if args.text:
        result = predictor.predict(args.text)
        print(f"\nText: {result['text']}")
        print(f"Predicted sentiment: {result['predicted_label'].upper()}")
        if "confidence_scores" in result:
            print("Scores:", {k: round(v, 3) for k, v in result["confidence_scores"].items()})

    elif args.csv:
        df = pd.read_csv(args.csv)
        if args.text_col not in df.columns:
            logger.error(f"Column '{args.text_col}' not found in {args.csv}")
            return
        predictions = predictor.predict_batch(df[args.text_col])
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(config.PREDICTIONS_FILE, index=False)
        logger.info(f"Saved {len(predictions)} predictions to {config.PREDICTIONS_FILE}")

    else:
        print("Interactive mode — type a sentence to classify (or 'quit' to exit)")
        while True:
            text = input("\n> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predictor.predict(text)
            print(f"Predicted sentiment: {result['predicted_label'].upper()}")


if __name__ == "__main__":
    run_cli()
