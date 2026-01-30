"""
Streamlit dashboard for hotel analytics.
Assignment: 3-5 core features for hotel managers (satisfaction drivers, improvement, benchmarking, trends).
Run: streamlit run app/streamlit_app.py (from project root)
"""
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.utils import get_engine

st.set_page_config(page_title="Hotel Analytics", layout="wide")
st.title("Hotel Analytics Dashboard")
st.caption("IS5126 Applied Analytics – HospitalityTech Solutions")

# Use sample DB if full DB not present (for TAs / quick demo)
from src.utils import get_db_path, get_engine
full_path = get_db_path(sample=False)
sample_path = get_db_path(sample=True)
if full_path.exists():
    engine = get_engine(sample=False)
elif sample_path.exists():
    engine = get_engine(sample=True)
else:
    st.error("No database found. Run: 1) python -m src.data_processing --full-etl 2) python -m src.data_processing")
    st.stop()


# --- Feature 1: Overview & satisfaction drivers ---
st.header("1. Overview & satisfaction drivers")
col1, col2, col3, col4 = st.columns(4)
with col1:
    n_reviews = pd.read_sql("SELECT COUNT(*) AS n FROM reviews", engine).iloc[0]["n"]
    st.metric("Total reviews", f"{n_reviews:,}")
with col2:
    n_hotels = pd.read_sql("SELECT COUNT(DISTINCT offering_id) AS n FROM reviews", engine).iloc[0]["n"]
    st.metric("Hotels", f"{n_hotels:,}")
with col3:
    avg_rating = pd.read_sql("SELECT AVG(rating_overall) AS avg FROM reviews", engine).iloc[0]["avg"]
    st.metric("Avg overall rating", f"{avg_rating:.2f}" if avg_rating else "N/A")
with col4:
    avg_helpful = pd.read_sql("SELECT AVG(num_helpful_votes) AS avg FROM reviews", engine).iloc[0]["avg"]
    st.metric("Avg helpful votes", f"{avg_helpful:.1f}" if avg_helpful else "N/A")

st.subheader("Aspect ratings (drivers of satisfaction)")
aspects = ["rating_service", "rating_cleanliness", "rating_value", "rating_location", "rating_sleep_quality", "rating_rooms"]
aspect_labels = [a.replace("rating_", "").replace("_", " ").title() for a in aspects]
avgs = pd.read_sql(
    f"SELECT {', '.join('AVG(' + a + ') AS ' + a for a in aspects)} FROM reviews",
    engine,
).iloc[0]
st.bar_chart({l: [avgs[a]] for l, a in zip(aspect_labels, aspects)})


# --- Feature 2: Hotel comparison (benchmarking) ---
st.header("2. Hotel comparison")
top_n = st.slider("Top N hotels by review count", 5, 30, 10)
agg = pd.read_sql(f"""
    SELECT offering_id, COUNT(*) AS n_reviews, ROUND(AVG(rating_overall), 2) AS avg_rating,
           ROUND(AVG(rating_cleanliness), 2) AS avg_cleanliness
    FROM reviews
    GROUP BY offering_id
    ORDER BY n_reviews DESC
    LIMIT {int(top_n)}
""", engine)
st.dataframe(agg, use_container_width=True)


# --- Feature 3: Rating distribution ---
st.header("3. Rating distribution")
dist = pd.read_sql(
    "SELECT rating_overall AS rating, COUNT(*) AS count FROM reviews GROUP BY rating_overall ORDER BY rating",
    engine,
)
st.bar_chart(dist.set_index("rating"))


# --- Feature 4: Time trend (review volume by year) ---
st.header("4. Review volume by year")
try:
    trend = pd.read_sql("""
        SELECT CAST(SUBSTR(TRIM(date), -4) AS INTEGER) AS year, COUNT(*) AS count
        FROM reviews WHERE date IS NOT NULL AND LENGTH(TRIM(date)) >= 4
        GROUP BY year ORDER BY year
    """, engine)
    st.line_chart(trend.set_index("year"))
except Exception:
    st.info("Date parsing not available in this DB. Use full ETL for time trends.")


# --- Feature 5: Improvement opportunities (low-rated aspects) ---
st.header("5. Improvement opportunities")
st.caption("Hotels with lowest avg cleanliness vs overall (potential focus)")
low_clean = pd.read_sql("""
    SELECT offering_id, COUNT(*) AS n_reviews,
           ROUND(AVG(rating_overall), 2) AS avg_overall,
           ROUND(AVG(rating_cleanliness), 2) AS avg_cleanliness,
           ROUND(AVG(rating_cleanliness) - AVG(rating_overall), 2) AS gap
    FROM reviews
    GROUP BY offering_id
    HAVING n_reviews >= 50
    ORDER BY gap ASC
    LIMIT 15
""", engine)
st.dataframe(low_clean, use_container_width=True)

st.divider()
st.caption("Dashboard uses sample or full DB. See README for setup.")
