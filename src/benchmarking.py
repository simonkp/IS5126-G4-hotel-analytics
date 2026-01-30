"""
Competitive benchmarking: identify comparable hotel groups and compare performance.
Assignment: "Who are my real competitors? How do we systematically identify truly
comparable properties? What are similar hotels doing better? Where to focus improvement?"
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_engine


def get_reviews_df(sample: bool = False) -> pd.DataFrame:
    """Load reviews from DB (optionally sample DB for quick runs)."""
    engine = get_engine(sample=sample)
    return pd.read_sql(
        "SELECT id, offering_id, author_id, rating_overall, rating_service, rating_cleanliness, "
        "rating_value, rating_location, rating_sleep_quality, rating_rooms, date, num_helpful_votes FROM reviews",
        engine,
    )


def comparable_groups_by_volume_and_rating(
    df: pd.DataFrame,
    n_reviews_bins: list[tuple[int, int]] | None = None,
    rating_tier_bins: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """
    Group hotels by review volume and average rating tier (comparable peers).
    Returns offering_id with group labels and summary stats.
    """
    agg = df.groupby("offering_id").agg(
        n_reviews=("id", "count"),
        avg_rating=("rating_overall", "mean"),
        avg_service=("rating_service", "mean"),
        avg_cleanliness=("rating_cleanliness", "mean"),
        avg_value=("rating_value", "mean"),
    ).reset_index()
    if n_reviews_bins is None:
        n_reviews_bins = [(0, 100), (100, 500), (500, 2000), (2000, 10000), (10000, 1_000_000)]
    if rating_tier_bins is None:
        rating_tier_bins = [(0, 2.5), (2.5, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.1)]
    def volume_group(n):
        for i, (lo, hi) in enumerate(n_reviews_bins):
            if lo <= n < hi:
                return i
        return len(n_reviews_bins)
    def rating_group(r):
        for i, (lo, hi) in enumerate(rating_tier_bins):
            if lo <= r < hi:
                return i
        return len(rating_tier_bins)
    agg["volume_group"] = agg["n_reviews"].apply(volume_group)
    agg["rating_tier"] = agg["avg_rating"].apply(rating_group)
    agg["peer_group"] = agg["volume_group"].astype(str) + "_" + agg["rating_tier"].astype(str)
    return agg


def best_practices_within_peers(peer_agg: pd.DataFrame, metric: str = "avg_cleanliness") -> pd.DataFrame:
    """Within each peer_group, rank hotels by metric (e.g. cleanliness) for best-practice identification."""
    peer_agg = peer_agg.copy()
    peer_agg["rank_in_peer"] = peer_agg.groupby("peer_group")[metric].rank(ascending=False, method="min")
    return peer_agg.sort_values(["peer_group", "rank_in_peer"])


def recommendations_for_underperformers(
    peer_agg: pd.DataFrame,
    peer_group: str,
    metric: str = "avg_cleanliness",
    bottom_pct: float = 0.25,
) -> pd.DataFrame:
    """Hotels in the bottom bottom_pct of metric within peer_group, with peer median for gap analysis."""
    g = peer_agg[peer_agg["peer_group"] == peer_group].copy()
    if g.empty:
        return g
    g["peer_median"] = g[metric].median()
    g["gap"] = g["peer_median"] - g[metric]
    threshold = g[metric].quantile(bottom_pct)
    return g[g[metric] <= threshold].sort_values("gap", ascending=False)


if __name__ == "__main__":
    # Demo: run with sample DB
    df = get_reviews_df(sample=True)
    print("Reviews shape:", df.shape)
    peers = comparable_groups_by_volume_and_rating(df)
    print("Peer groups:", peers["peer_group"].nunique())
    print(peers.groupby("peer_group").size().head(10))
