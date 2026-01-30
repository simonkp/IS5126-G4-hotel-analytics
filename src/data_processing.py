"""
Data processing: full ETL (JSONL → SQLite), sample DB build, latest-5y logic.
Run full ETL: python -m src.data_processing --full-etl
Run sample DB only: python -m src.data_processing
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from dateutil import parser as date_parser
from sqlalchemy import create_engine, text
from tqdm import tqdm

from src.utils import get_db_path, get_review_json_path, PROJECT_ROOT, REVIEW_JSON_PATH, DB_PATH

# Assignment: latest 5 years, 50K-80K+ reviews after filtering
LATEST_YEARS = 5
MIN_REVIEWS = 50_000
MAX_REVIEWS = 80_000  # optional cap; "80K+" means at least 50K, can have more
SAMPLE_SIZE = 5_000   # for reviews_sample.db (TA testing)
BATCH_SIZE = 10_000

AUTHORS_DDL = """
CREATE TABLE authors (
    id TEXT PRIMARY KEY,
    username TEXT,
    location TEXT,
    num_cities INTEGER,
    num_helpful_votes INTEGER,
    num_reviews INTEGER,
    num_type_reviews INTEGER
);
"""
REVIEWS_DDL = """
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    offering_id INTEGER,
    author_id TEXT,
    title TEXT,
    text TEXT,
    date TEXT,
    date_stayed TEXT,
    num_helpful_votes INTEGER,
    via_mobile INTEGER,
    rating_overall REAL,
    rating_service REAL,
    rating_cleanliness REAL,
    rating_value REAL,
    rating_location REAL,
    rating_sleep_quality REAL,
    rating_rooms REAL,
    FOREIGN KEY (author_id) REFERENCES authors(id)
);
"""
REVIEWS_INDEXES = [
    "CREATE INDEX idx_reviews_offering ON reviews(offering_id);",
    "CREATE INDEX idx_reviews_author ON reviews(author_id);",
    "CREATE INDEX idx_reviews_rating_overall ON reviews(rating_overall);",
]


def _row_from_record(rec: dict) -> tuple[dict | None, dict | None]:
    """Extract (author_row, review_row). Returns (None, None) if record invalid."""
    try:
        rid = rec.get("id")
        if rid is None:
            return None, None
        author = rec.get("author") or {}
        ratings = rec.get("ratings") or {}
        author_id = author.get("id") or ""
        author_row = {
            "id": author_id,
            "username": author.get("username"),
            "location": author.get("location"),
            "num_cities": author.get("num_cities"),
            "num_helpful_votes": author.get("num_helpful_votes"),
            "num_reviews": author.get("num_reviews"),
            "num_type_reviews": author.get("num_type_reviews"),
        }
        review_row = {
            "id": rid,
            "offering_id": rec.get("offering_id"),
            "author_id": author_id,
            "title": rec.get("title"),
            "text": rec.get("text"),
            "date": rec.get("date"),
            "date_stayed": rec.get("date_stayed"),
            "num_helpful_votes": rec.get("num_helpful_votes"),
            "via_mobile": 1 if rec.get("via_mobile") else 0,
            "rating_overall": ratings.get("overall"),
            "rating_service": ratings.get("service"),
            "rating_cleanliness": ratings.get("cleanliness"),
            "rating_value": ratings.get("value"),
            "rating_location": ratings.get("location"),
            "rating_sleep_quality": ratings.get("sleep_quality"),
            "rating_rooms": ratings.get("rooms"),
        }
        return author_row, review_row
    except Exception:
        return None, None


def run_full_etl(
    json_path: Path | None = None,
    db_path: Path | None = None,
    dry_run: int | None = None,
) -> tuple[int, int, int]:
    """
    Stream JSONL into SQLite. Drops and recreates tables.
    Returns (lines_read, total_inserted, errors).
    """
    json_path = json_path or (PROJECT_ROOT / REVIEW_JSON_PATH)
    db_path = db_path or get_db_path(sample=False)
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Place review.json there or set REVIEW_JSON_PATH.")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)

    if dry_run is None:
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS reviews;"))
            conn.execute(text("DROP TABLE IF EXISTS authors;"))
            conn.execute(text(AUTHORS_DDL))
            conn.execute(text(REVIEWS_DDL))
            for stmt in REVIEWS_INDEXES:
                conn.execute(text(stmt))
            conn.commit()

    author_cols = ["id", "username", "location", "num_cities", "num_helpful_votes", "num_reviews", "num_type_reviews"]
    review_cols = [
        "id", "offering_id", "author_id", "title", "text", "date", "date_stayed",
        "num_helpful_votes", "via_mobile",
        "rating_overall", "rating_service", "rating_cleanliness", "rating_value",
        "rating_location", "rating_sleep_quality", "rating_rooms",
    ]
    author_sql = f"INSERT OR IGNORE INTO authors ({', '.join(author_cols)}) VALUES ({', '.join('?' * len(author_cols))})"
    review_sql = f"INSERT INTO reviews ({', '.join(review_cols)}) VALUES ({', '.join('?' * len(review_cols))})"

    authors_batch: list[dict] = []
    reviews_batch: list[dict] = []
    lines_read = 0
    errors = 0
    total_inserted = 0

    def flush() -> None:
        nonlocal total_inserted
        if dry_run is not None:
            total_inserted += len(reviews_batch)
            authors_batch.clear()
            reviews_batch.clear()
            return
        raw = engine.raw_connection()
        try:
            cur = raw.cursor()
            if authors_batch:
                seen: set[str] = set()
                unique_authors = [a for a in authors_batch if a["id"] and a["id"] not in seen and not seen.add(a["id"])]
                if unique_authors:
                    cur.executemany(author_sql, [[a[c] for c in author_cols] for a in unique_authors])
            if reviews_batch:
                cur.executemany(review_sql, [[r[c] for c in review_cols] for r in reviews_batch])
            raw.commit()
        finally:
            raw.close()
        total_inserted += len(reviews_batch)
        authors_batch.clear()
        reviews_batch.clear()

    with open(json_path, "r", encoding="utf-8", errors="replace") as f:
        if dry_run is not None:
            def limit():
                for _ in range(dry_run):
                    line = f.readline()
                    if not line:
                        break
                    yield line
            iterator = limit()
        else:
            iterator = iter(f)
        pbar = tqdm(iterator, desc="Lines", unit=" lines")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            lines_read += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            author_row, review_row = _row_from_record(rec)
            if author_row is None or review_row is None:
                errors += 1
                continue
            authors_batch.append(author_row)
            reviews_batch.append(review_row)
            if len(reviews_batch) >= BATCH_SIZE:
                flush()
            pbar.set_postfix(inserted=total_inserted + len(reviews_batch), err=errors)
    flush()
    return lines_read, total_inserted, errors


def parse_review_year(date_str: str | None) -> int | None:
    """Parse date string (e.g. 'December 17, 2012') to year. Returns None if invalid."""
    if not date_str or not str(date_str).strip():
        return None
    try:
        return date_parser.parse(str(date_str).strip()).year
    except Exception:
        return None


def get_latest_years_cutoff(engine) -> int:
    """Return the latest year present in DB; cutoff = that year - (LATEST_YEARS-1)."""
    import pandas as pd
    df = pd.read_sql("SELECT DISTINCT date FROM reviews WHERE date IS NOT NULL AND date != '' LIMIT 50000", engine)
    df["year"] = df["date"].apply(parse_review_year)
    df = df.dropna(subset=["year"])
    if df.empty:
        return 2010  # fallback
    max_year = int(df["year"].max())
    return max_year - (LATEST_YEARS - 1)


def create_filtered_view_or_table(engine, cutoff_year: int):
    """Create a view 'reviews_latest_5y' with review_year column for filtering."""
    from sqlalchemy import text
    with engine.connect() as conn:
        # SQLite doesn't have a simple date parse; we do filtering in Python when building sample
        # For direct SQL we could use a view that keeps all and add review_year in a materialized step
        conn.execute(text("DROP VIEW IF EXISTS reviews_latest_5y"))
        conn.commit()
    # View with year extracted via substring: "December 17, 2012" -> 2012
    # SQLite: we can use SUBSTR(date, -4) for 4-digit year at end
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE VIEW IF NOT EXISTS reviews_latest_5y AS
            SELECT *, CAST(SUBSTR(TRIM(date), -4) AS INTEGER) AS review_year
            FROM reviews
            WHERE date IS NOT NULL AND TRIM(date) != ''
              AND SUBSTR(TRIM(date), -4) GLOB '[0-9][0-9][0-9][0-9]'
        """))
        conn.commit()


def build_sample_db(sample_size: int = SAMPLE_SIZE, from_full_db: bool = True) -> Path:
    """
    Create data/reviews_sample.db with at least sample_size reviews (for TAs).
    If from_full_db: copy from full reviews.db (filter latest 5 years in SQL if possible).
    """
    sample_path = get_db_path(sample=True)
    full_path = get_db_path(sample=False)
    if from_full_db and not full_path.exists():
        raise FileNotFoundError(f"Full DB not found: {full_path}. Run: python -m src.data_processing --full-etl first.")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path.exists():
        sample_path.unlink()
    conn_full = sqlite3.connect(str(full_path))
    conn_sample = sqlite3.connect(str(sample_path))
    # Copy schema
    for row in conn_full.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name IN ('authors','reviews') ORDER BY name"):
        if row[0]:
            conn_sample.execute(row[0])
    for row in conn_full.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='reviews'"):
        if row[0]:
            conn_sample.execute(row[0])
    # Prefer latest 5 years: year from end of date string (e.g. "December 17, 2012" -> 2012)
    cursor = conn_full.execute("""
        SELECT id, offering_id, author_id, title, text, date, date_stayed, num_helpful_votes, via_mobile,
               rating_overall, rating_service, rating_cleanliness, rating_value, rating_location,
               rating_sleep_quality, rating_rooms
        FROM reviews
        WHERE date IS NOT NULL AND TRIM(date) != ''
          AND LENGTH(TRIM(date)) >= 4
        ORDER BY CAST(SUBSTR(TRIM(date), -4) AS INTEGER) DESC, id
        LIMIT ?
    """, (max(sample_size, 60_000),))
    reviews = cursor.fetchall()
    if len(reviews) < sample_size:
        cursor = conn_full.execute(
            "SELECT id, offering_id, author_id, title, text, date, date_stayed, num_helpful_votes, via_mobile, rating_overall, rating_service, rating_cleanliness, rating_value, rating_location, rating_sleep_quality, rating_rooms FROM reviews LIMIT ?",
            (sample_size,),
        )
        reviews = cursor.fetchall()
    reviews = reviews[:sample_size] if len(reviews) > sample_size else reviews
    author_ids = set(r[2] for r in reviews)
    # Copy authors
    placeholders = ",".join("?" * len(author_ids))
    authors = conn_full.execute(f"SELECT id, username, location, num_cities, num_helpful_votes, num_reviews, num_type_reviews FROM authors WHERE id IN ({placeholders})", list(author_ids)).fetchall()
    conn_sample.executemany(
        "INSERT INTO authors (id, username, location, num_cities, num_helpful_votes, num_reviews, num_type_reviews) VALUES (?,?,?,?,?,?,?)",
        authors,
    )
    conn_sample.executemany(
        """INSERT INTO reviews (id, offering_id, author_id, title, text, date, date_stayed, num_helpful_votes, via_mobile,
           rating_overall, rating_service, rating_cleanliness, rating_value, rating_location, rating_sleep_quality, rating_rooms)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        reviews,
    )
    conn_sample.commit()
    conn_full.close()
    conn_sample.close()
    return sample_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Data processing: full ETL or sample DB.")
    parser.add_argument("--full-etl", action="store_true", help="Stream review.json into SQLite (reviews.db).")
    parser.add_argument("--dry-run", type=int, metavar="N", help="With --full-etl: only read first N lines.")
    args = parser.parse_args()
    if args.full_etl:
        try:
            lines_read, total_inserted, errors = run_full_etl(dry_run=args.dry_run)
            print(f"Lines read: {lines_read}, inserted: {total_inserted}, errors: {errors}")
            return 0
        except Exception as e:
            print(e, file=sys.stderr)
            return 1
    try:
        path = build_sample_db()
        with sqlite3.connect(str(path)) as c:
            n = c.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        print(f"Created {path} with {n} reviews.")
        return 0
    except Exception as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
