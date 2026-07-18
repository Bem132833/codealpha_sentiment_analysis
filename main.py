"""
main.py
-------
Single entry point that runs the full sentiment analysis pipeline
end-to-end:

  1. Load & unify the three domain datasets (or generate synthetic
     fallback data if raw files haven't been downloaded yet)
  2. Preprocess text (clean, tokenize, remove stopwords, lemmatize)
  3. Engineer TF-IDF features and split into train/test
  4. Train candidate models and select the best by macro-F1
  5. Evaluate the final model (overall + per-domain)
  6. Run the cross-domain generalization analysis
  7. Generate all figures for the report

Usage:
    python main.py                  # run everything
    python main.py --skip-figures   # skip plotting (faster iteration)
    python main.py --skip-cross-domain
"""

import argparse
import sys
import time

import pandas as pd

from src import config
from src.data_loader import load_all_domains
from src.preprocessing import preprocess_dataframe
from src.train import run as run_training
from src.evaluate import evaluate_model
from src.cross_domain_eval import run_cross_domain_matrix
from src.visualization import generate_all_figures
from src.utils import get_logger, load_pickle

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(description="Run the full sentiment analysis pipeline")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--skip-cross-domain", action="store_true", help="Skip cross-domain analysis")
    args = parser.parse_args()

    pipeline_start = time.time()
    config.ensure_directories()

    logger.info("=" * 70)
    logger.info("STEP 1/6 — Loading and unifying datasets")
    logger.info("=" * 70)
    raw_df = load_all_domains()

    logger.info("=" * 70)
    logger.info("STEP 2/6 — Preprocessing text")
    logger.info("=" * 70)
    try:
        processed_df = preprocess_dataframe(raw_df)
    except LookupError:
        logger.error(
            "NLTK data missing. Run: python -m src.preprocessing --download-nltk"
        )
        sys.exit(1)

    processed_df.to_csv(config.PROCESSED_FILE, index=False)
    logger.info(f"Saved processed dataset ({len(processed_df):,} rows) to {config.PROCESSED_FILE}")

    logger.info("=" * 70)
    logger.info("STEP 3/6 — Feature engineering & training")
    logger.info("=" * 70)
    train_results = run_training()
    logger.info(
        f"Best model: {train_results['best_model_name']} "
        f"(macro-F1={train_results['best_macro_f1']:.4f})"
    )
    logger.info(f"All candidate scores: {train_results['all_results']}")

    logger.info("=" * 70)
    logger.info("STEP 4/6 — Evaluating final model")
    logger.info("=" * 70)
    model = load_pickle(config.MODEL_PATH)
    evaluate_model(
        model,
        train_results["X_test"],
        train_results["y_test"],
        train_results["encoder"],
        train_results["test_df"],
    )

    if not args.skip_cross_domain:
        logger.info("=" * 70)
        logger.info("STEP 5/6 — Cross-domain generalization analysis")
        logger.info("=" * 70)
        run_cross_domain_matrix(processed_df)
    else:
        logger.info("Skipping cross-domain analysis (--skip-cross-domain)")

    if not args.skip_figures:
        logger.info("=" * 70)
        logger.info("STEP 6/6 — Generating figures")
        logger.info("=" * 70)
        generate_all_figures()
    else:
        logger.info("Skipping figure generation (--skip-figures)")

    elapsed = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Model:              {config.MODEL_PATH}")
    logger.info(f"  Metrics:            {config.METRICS_FILE}")
    logger.info(f"  Cross-domain:       {config.CROSS_DOMAIN_METRICS_FILE}")
    logger.info(f"  Figures:            {config.FIGURES_DIR}")
    logger.info("=" * 70)
    logger.info("Try a prediction with: python -m src.predict --text \"your text here\"")


if __name__ == "__main__":
    main()