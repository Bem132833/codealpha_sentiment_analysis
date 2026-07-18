"""
visualization.py
------------------
Generates every figure used in the final report: class distribution,
confusion matrix, cross-domain generalization heatmap, and top predictive
n-grams per class. All figures are saved to outputs/figures/ as PNGs so
they can be embedded directly in the README or a report.
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display required

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.utils import get_logger, load_json

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def _savefig(fig, filename: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    order = list(config.LABELS)
    sns.countplot(data=df, x="label", order=order, hue="label",
                   palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title("Overall class distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")

    domain_counts = df.groupby(["domain", "label"]).size().unstack(fill_value=0)
    domain_counts = domain_counts.reindex(columns=order, fill_value=0)
    domain_counts.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("Class distribution by domain")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()
    _savefig(fig, "class_distribution.png")


def plot_text_length_distribution(df: pd.DataFrame, text_col: str = "text") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lengths = df[text_col].astype(str).str.split().apply(len)
    for domain in config.DOMAINS:
        mask = df["domain"] == domain
        if mask.any():
            sns.kdeplot(lengths[mask], label=domain, fill=True, alpha=0.25, ax=ax)
    ax.set_title("Text length distribution by domain (word count)")
    ax.set_xlabel("Word count")
    ax.set_xlim(left=0)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "text_length_distribution.png")


def plot_confusion_matrix(metrics: dict) -> None:
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["label_order"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    _savefig(fig, "confusion_matrix.png")


def plot_per_domain_performance(metrics: dict) -> None:
    per_domain = metrics.get("per_domain", {})
    if not per_domain:
        return

    domains = list(per_domain.keys())
    accs = [per_domain[d]["accuracy"] for d in domains]
    f1s = [per_domain[d]["macro_f1"] for d in domains]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model performance by domain (combined model)")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "per_domain_performance.png")


def plot_cross_domain_heatmap(cross_domain_metrics: dict) -> None:
    matrix = cross_domain_metrics["matrix"]
    domains = cross_domain_metrics["domains_evaluated"]

    data = pd.DataFrame(matrix).reindex(index=domains, columns=domains)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        xticklabels=domains, yticklabels=domains, ax=ax,
        cbar_kws={"label": "Macro-F1"},
    )
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Trained on")
    ax.set_title("Cross-domain generalization (macro-F1)")
    fig.tight_layout()
    _savefig(fig, "cross_domain_heatmap.png")


def plot_top_features(vectorizer, model, top_n: int = 15) -> None:
    """Bar chart of the strongest TF-IDF n-grams per class (linear models only)."""
    if not hasattr(model, "coef_"):
        logger.warning("Model has no coef_ attribute; skipping top-features plot.")
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    n_classes = model.coef_.shape[0]

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    for i in range(n_classes):
        coefs = model.coef_[i]
        top_idx = np.argsort(coefs)[-top_n:]
        axes[i].barh(feature_names[top_idx], coefs[top_idx], color="#55A868")
        label = config.LABELS[i] if i < len(config.LABELS) else f"class_{i}"
        axes[i].set_title(f"Top n-grams: {label}")

    fig.tight_layout()
    _savefig(fig, "top_features_per_class.png")


def generate_all_figures() -> None:
    """Convenience entry point: build every figure from the persisted artifacts."""
    from src.utils import load_pickle

    if config.PROCESSED_FILE.exists():
        df = pd.read_csv(config.PROCESSED_FILE)
        plot_class_distribution(df)
        plot_text_length_distribution(df)

    if config.METRICS_FILE.exists():
        metrics = load_json(config.METRICS_FILE)
        plot_confusion_matrix(metrics)
        plot_per_domain_performance(metrics)

    if config.CROSS_DOMAIN_METRICS_FILE.exists():
        cross_metrics = load_json(config.CROSS_DOMAIN_METRICS_FILE)
        plot_cross_domain_heatmap(cross_metrics)

    if config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists():
        model = load_pickle(config.MODEL_PATH)
        vectorizer = load_pickle(config.VECTORIZER_PATH)
        plot_top_features(vectorizer, model)


if __name__ == "__main__":
    generate_all_figures()
