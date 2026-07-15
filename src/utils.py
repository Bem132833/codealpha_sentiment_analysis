"""
utils.py
--------
Small shared helpers used across the pipeline: logging setup, pickle
save/load wrappers, and a timing decorator. Keeping these in one place
avoids every module reinventing its own logger or its own pickle code.
"""

import functools
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Callable

from src import config


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both console and outputs/logs/pipeline.log."""
    config.ensure_directories()
    logger = logging.getLogger(name)

    if logger.handlers:  # avoid duplicate handlers if called more than once
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def timeit(func: Callable) -> Callable:
    """Decorator that logs how long a function took to run."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} finished in {elapsed:.2f}s")
        return result
    return wrapper


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
