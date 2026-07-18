"""
evaluate.py
-----------
Computes and persists evaluation metrics for the trained model:
overall classification report, confusion matrix, and a per-domain
breakdown (since a model trained on all three domains combined can still
perform very differently on tweets vs. news vs. reviews).
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src import config
from src.utils import get_logger, save_json, timeit

logger = get_logger(__name__)


@timeit
def evaluate_model(model, X_test, y_test, encoder, test_df: pd.DataFrame) -> dict:
    preds = model.predict(X_test)
    label_names = list(encoder.classes_)

    report = classification_report(
        y_test, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, preds).tolist()
    overall_accuracy = accuracy_score(y_test, preds)
    overall_macro_f1 = f1_score(y_test, preds, average="macro")

    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")
    logger.info(f"Overall macro-F1: {overall_macro_f1:.4f}")
    logger.info("\n" + classification_report(y_test, preds, target_names=label_names, zero_division=0))

    # Per-domain breakdown
    per_domain = {}
    test_df = test_df.reset_index(drop=True)
    for domain in config.DOMAINS:
        mask = (test_df["domain"] == domain).values
        if mask.sum() == 0:
            continue
        domain_f1 = f1_score(y_test[mask], preds[mask], average="macro")
        domain_acc = accuracy_score(y_test[mask], preds[mask])
        per_domain[domain] = {
            "n_samples": int(mask.sum()),
            "accuracy": round(float(domain_acc), 4),
            "macro_f1": round(float(domain_f1), 4),
        }
        logger.info(f"  [{domain}] n={mask.sum()} | acc={domain_acc:.4f} | macro-F1={domain_f1:.4f}")

    metrics = {
        "overall_accuracy": round(float(overall_accuracy), 4),
        "overall_macro_f1": round(float(overall_macro_f1), 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "label_order": label_names,
        "per_domain": per_domain,
    }

    save_json(metrics, config.METRICS_FILE)
    logger.info(f"Saved metrics to {config.METRICS_FILE}")
    return metrics


if __name__ == "__main__":
    from src.train import run as train_run

    train_results = train_run()

    from src.utils import load_pickle
    model = load_pickle(config.MODEL_PATH)

    evaluate_model(
        model,
        train_results["X_test"],
        train_results["y_test"],
        train_results["encoder"],
        train_results["test_df"],
    )
