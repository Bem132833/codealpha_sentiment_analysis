"""
train.py
--------
Trains every model in config.MODEL_CANDIDATES on the same TF-IDF features,
scores each on the held-out test set, and persists the one with the
highest macro-F1 as the final sentiment_model.pkl.

Macro-F1 (not accuracy) is used to pick the winner because the neutral
class is typically underrepresented and harder to predict; macro-F1
weights all three classes equally instead of letting the majority class
dominate the score.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from src import config
from src.feature_engineering import build_features
from src.utils import get_logger, save_pickle, timeit

logger = get_logger(__name__)

_MODEL_BUILDERS = {
    "logistic_regression": lambda params: LogisticRegression(**params),
    "linear_svm": lambda params: LinearSVC(**params),
}


@timeit
def train_candidates(X_train, y_train, X_test, y_test) -> dict:
    """Train every candidate model and return their fitted instances + scores."""
    results = {}
    for name, params in config.MODEL_CANDIDATES.items():
        logger.info(f"Training {name} with params: {params}")
        model = _MODEL_BUILDERS[name](params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        macro_f1 = f1_score(y_test, preds, average="macro")
        logger.info(f"{name} macro-F1 on test set: {macro_f1:.4f}")

        results[name] = {"model": model, "macro_f1": macro_f1}

    return results


@timeit
def select_and_save_best(results: dict):
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_model = results[best_name]["model"]
    best_score = results[best_name]["macro_f1"]

    save_pickle(best_model, config.MODEL_PATH)
    logger.info(
        f"Best model: {best_name} (macro-F1={best_score:.4f}) -> saved to {config.MODEL_PATH}"
    )
    return best_name, best_model, best_score


def run() -> dict:
    if not config.PROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"{config.PROCESSED_FILE} not found. Run data_loader.py then "
            f"preprocessing.py first."
        )

    data = pd.read_csv(config.PROCESSED_FILE)
    X_train, X_test, y_train, y_test, train_df, test_df, vectorizer, encoder = build_features(data)

    results = train_candidates(X_train, y_train, X_test, y_test)
    best_name, best_model, best_score = select_and_save_best(results)

    return {
        "best_model_name": best_name,
        "best_macro_f1": best_score,
        "all_results": {k: v["macro_f1"] for k, v in results.items()},
        "X_test": X_test,
        "y_test": y_test,
        "test_df": test_df,
        "encoder": encoder,
    }


if __name__ == "__main__":
    run()
