#!/usr/bin/env python3
"""
Build a year-stratified, hotel-diversified curated SQLite DB from a full reviews DB.

- Input:  data/reviews_full.db  (tables: reviews, authors)
- Output: data/reviews.db       (tables: reviews(with date_iso), authors)

Run this script from the project root:
    python3 src/build_curated_db.py
"""

import os
import sqlite3
from datetime import datetime
from collections import defaultdict
from dateutil import parser as date_parser

SRC_DB = "data/reviews_full.db"
OUT_DB = "data/reviews.db"

TARGET_N = 80_000
YEARS = ["2008", "2009", "2010", "2011", "2012"]
CAP_PER_HOTEL_PER_YEAR = 30
CHUNK = 5_000
FLOOR_PER_YEAR = 2_000  # ensures representation even if a year is sparse

def parse_to_iso(date_str: str) -> str:
    dt = date_parser.parse(date_str, fuzzy=True, default=datetime(1900, 1, 1))
    return dt.date().isoformat()

def main():
    if not os.path.exists(SRC_DB):
        raise FileNotFoundError(f"Source DB not found: {SRC_DB}")

    if os.path.exists(OUT_DB):
        os.remove(OUT_DB)

    src = sqlite3.connect(SRC_DB)
    src.row_factory = sqlite3.Row

    # --- Pass 1: parse date -> date_iso and bucket by year ---
    print("Parsing dates and assigning year buckets...")
    meta_by_year = {y: [] for y in YEARS}  # year -> list[(review_id, offering_id, date_iso)]
    bad = 0
    total = 0

    for row in src.execute("SELECT id, offering_id, date FROM reviews"):
        total += 1
        try:
            date_iso = parse_to_iso(row["date"])
            y = date_iso[:4]
            if y in meta_by_year:
                meta_by_year[y].append((int(row["id"]), int(row["offering_id"]), date_iso))
        except Exception:
            bad += 1

    print(f"Rows scanned: {total:,}")
    print(f"Unparsable:   {bad:,} ({bad/total:.2%})")

    year_counts = {y: len(meta_by_year[y]) for y in YEARS}
    print("Candidate counts by year (2008-2012):")
    for y in YEARS:
        print(f"  {y}: {year_counts[y]:,}")

    total_candidates = sum(year_counts.values())
    if total_candidates == 0:
        raise RuntimeError("No candidates found in the target year window.")

    # --- Allocate TARGET_N across years proportionally with floors ---
    remaining = TARGET_N - FLOOR_PER_YEAR * len(YEARS)
    if remaining < 0:
        raise RuntimeError("TARGET_N too small for FLOOR_PER_YEAR setting.")

    alloc_year = {}
    for y in YEARS:
        prop = year_counts[y] / total_candidates
        alloc_year[y] = FLOOR_PER_YEAR + int(round(prop * remaining))

    # adjust to exact TARGET_N
    delta = TARGET_N - sum(alloc_year.values())
    if delta != 0:
        ys_sorted = sorted(YEARS, key=lambda y: year_counts[y], reverse=True)
        i = 0
        while delta != 0:
            y = ys_sorted[i % len(ys_sorted)]
            if delta > 0:
                alloc_year[y] += 1
                delta -= 1
            else:
                if alloc_year[y] > 1:
                    alloc_year[y] -= 1
                    delta += 1
            i += 1

    print("\nPlanned sample by year:")
    for y in YEARS:
        print(f"  {y}: {alloc_year[y]:,}")
    print(f"Total planned: {sum(alloc_year.values()):,}")

    # --- Pass 2: within each year, cap per hotel then select deterministically ---
    print("\nSelecting samples per year (hotel-stratified, deterministic)...")
    sample = []  # list[(review_id, date_iso)]

    for y in YEARS:
        rows = meta_by_year[y]

        by_hotel = defaultdict(list)
        for rid, oid, date_iso in rows:
            by_hotel[oid].append((rid, date_iso))

        capped_pool = []
        for oid, lst in by_hotel.items():
            # sort ascending; take most recent CAP
            lst_sorted = sorted(lst, key=lambda t: (t[1], t[0]))
            capped_pool.extend(lst_sorted[-CAP_PER_HOTEL_PER_YEAR:])

        # pick alloc_year[y] from capped pool: prefer recency but deterministic
        capped_sorted = sorted(capped_pool, key=lambda t: (t[1], t[0]))  # asc; take from end
        chosen = capped_sorted[-alloc_year[y]:]
        sample.extend(chosen)

    # de-dupe in case of any accidental overlap (shouldn't happen, but safe)
    sample = list({rid: date_iso for rid, date_iso in sample}.items())
    sample.sort(key=lambda t: t[0])

    if len(sample) != TARGET_N:
        print(f"WARNING: sampled {len(sample):,} reviews (expected {TARGET_N:,}).")

    print(f"Final sampled reviews total: {len(sample):,}")
    date_iso_by_id = dict(sample)
    sampled_ids = [rid for rid, _ in sample]

    # --- Create output schema (authors + reviews + date_iso) ---
    out = sqlite3.connect(OUT_DB)
    out.row_factory = sqlite3.Row

    authors_schema = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='authors';"
    ).fetchone()
    if not authors_schema or not authors_schema[0]:
        raise RuntimeError("Could not read authors table schema from source DB.")

    reviews_cols = src.execute("PRAGMA table_info(reviews);").fetchall()
    review_colnames = [c["name"] for c in reviews_cols]

    out.execute(authors_schema[0])

    col_defs = []
    for c in reviews_cols:
        name = c["name"]
        ctype = c["type"] if c["type"] else "TEXT"
        if c["pk"] == 1:
            col_defs.append(f"{name} {ctype} PRIMARY KEY")
        else:
            nn = "NOT NULL" if c["notnull"] else ""
            col_defs.append(f"{name} {ctype} {nn}".strip())

    col_defs.append("date_iso TEXT")
    out.execute(f"CREATE TABLE reviews ({', '.join(col_defs)});")
    out.commit()

    # performance pragmas
    out.execute("PRAGMA synchronous = OFF;")
    out.execute("PRAGMA journal_mode = MEMORY;")

    # --- Insert sampled reviews in chunks ---
    cols_sql = ",".join(review_colnames)
    placeholders = ",".join(["?"] * (len(review_colnames) + 1))
    insert_sql = f"INSERT INTO reviews ({cols_sql}, date_iso) VALUES ({placeholders})"

    print("\nInserting sampled reviews...")
    inserted = 0

    for i in range(0, len(sampled_ids), CHUNK):
        chunk_ids = sampled_ids[i : i + CHUNK]
        qmarks = ",".join(["?"] * len(chunk_ids))
        rows = src.execute(
            f"SELECT {cols_sql} FROM reviews WHERE id IN ({qmarks})", chunk_ids
        ).fetchall()

        payload = []
        for r in rows:
            rid = int(r["id"])
            payload.append(tuple(r[c] for c in review_colnames) + (date_iso_by_id[rid],))

        out.executemany(insert_sql, payload)
        inserted += len(payload)

    out.commit()
    print(f"Inserted reviews: {inserted:,}")

    # --- Insert referenced authors FROM SOURCE DB (correct) ---
    print("\nInserting referenced authors...")
    out.execute(f"ATTACH DATABASE '{SRC_DB}' AS src;")
    out.execute(
        """
        INSERT INTO authors
        SELECT a.*
        FROM src.authors a
        JOIN (
          SELECT DISTINCT trim(author_id) AS author_id
          FROM reviews
          WHERE author_id IS NOT NULL AND trim(author_id) != ''
        ) r
        ON a.id = r.author_id;
        """
    )
    out.commit()
    out.execute("DETACH DATABASE src;")
    out.commit()

    # --- Indexes ---
    print("\nCreating indexes...")
    out.execute("CREATE INDEX IF NOT EXISTS idx_reviews_offering_id ON reviews(offering_id);")
    out.execute("CREATE INDEX IF NOT EXISTS idx_reviews_author_id ON reviews(author_id);")
    out.execute("CREATE INDEX IF NOT EXISTS idx_reviews_rating_overall ON reviews(rating_overall);")
    out.execute("CREATE INDEX IF NOT EXISTS idx_reviews_date_iso ON reviews(date_iso);")
    out.commit()

    # --- Final stats ---
    n_reviews = out.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    n_authors = out.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
    n_hotels = out.execute("SELECT COUNT(DISTINCT offering_id) FROM reviews").fetchone()[0]
    dmin, dmax = out.execute("SELECT MIN(date_iso), MAX(date_iso) FROM reviews").fetchone()

    print("\nCurated DB stats:")
    print(f"  reviews: {n_reviews:,}")
    print(f"  authors: {n_authors:,}")
    print(f"  hotels:  {n_hotels:,}")
    print(f"  date_iso range: {dmin} to {dmax}")

    print("\nYear distribution in curated DB:")
    for y, n in out.execute(
        "SELECT substr(date_iso,1,4), COUNT(*) FROM reviews GROUP BY 1 ORDER BY 1;"
    ).fetchall():
        print(f"  {y}: {n:,}")

    src.close()
    out.close()
    print(f"\n✅ Created curated DB: {OUT_DB}")

if __name__ == "__main__":
    main()
