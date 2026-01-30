"""
Shared utilities: DB connection, paths, config.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda _: None  # optional: use defaults if not installed

from sqlalchemy import create_engine

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DB_PATH = os.getenv("DB_PATH", "data/reviews.db")
SAMPLE_DB_PATH = os.getenv("SAMPLE_DB_PATH", "data/reviews_sample.db")
REVIEW_JSON_PATH = os.getenv("REVIEW_JSON_PATH", "data/review.json")


def get_db_path(sample: bool = False) -> Path:
    """Return absolute path to SQLite DB."""
    p = SAMPLE_DB_PATH if sample else DB_PATH
    return PROJECT_ROOT / p


def get_engine(sample: bool = False):
    """Create SQLAlchemy engine for main or sample DB."""
    path = get_db_path(sample=sample)
    return create_engine(f"sqlite:///{path}", future=True)


def get_review_json_path() -> Path:
    """Return absolute path to review.json."""
    return PROJECT_ROOT / REVIEW_JSON_PATH
