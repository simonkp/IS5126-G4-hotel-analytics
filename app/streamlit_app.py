"""
Hotel Analytics Dashboard — IS5126 Assignment 1
Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import text
from src.utils import get_engine, get_db_path
from src.benchmarking import (
    get_reviews_df,
    extract_hotel_features,
    filter_low_signal_hotels,
    create_comparable_groups,
    generate_actionable_recommendations,
)

# ── colour palette ──────────────────────────────────────────────────────────
ACCENT      = "#6C5CE7"   # primary purple
ACCENT_LITE = "#A29BFE"   # lighter purple
GOOD        = "#00B894"   # green
WARN        = "#FDCB6E"   # amber
BAD         = "#D63031"   # red
BG_CARD     = "#F8F9FD"   # light card background
PALETTE     = px.colors.qualitative.Set2

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Analytics · IS5126 G4",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS tweaks ───────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fd 0%, #eef1fb 100%);
        border: 1px solid #e4e8f1;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(108,92,231,.08);
    }
    [data-testid="stMetric"] label { font-weight: 600; }
    .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] { background: #f4f3ff; }
</style>
""", unsafe_allow_html=True)

# ── database connection ─────────────────────────────────────────────────────
full_path   = get_db_path(sample=False)
sample_path = get_db_path(sample=True)

if full_path.exists():
    engine  = get_engine(sample=False)
    db_info = "Full Database"
    using_sample = False
else:
    engine  = get_engine(sample=True)
    db_info = "Sample Database (5 K reviews)"
    using_sample = True

# ── cached helpers ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_review_count():
    return pd.read_sql("SELECT COUNT(*) AS n FROM reviews", engine).iloc[0]["n"]

@st.cache_data(show_spinner=False)
def load_hotel_count():
    return pd.read_sql("SELECT COUNT(DISTINCT offering_id) AS n FROM reviews", engine).iloc[0]["n"]

@st.cache_data(show_spinner=False)
def load_avg_rating():
    return pd.read_sql("SELECT AVG(rating_overall) AS avg FROM reviews", engine).iloc[0]["avg"]

@st.cache_data(show_spinner=False)
def load_date_range():
    return pd.read_sql("""
        SELECT
            CAST(SUBSTR(TRIM(MIN(date)), -4) AS INTEGER) AS min_year,
            CAST(SUBSTR(TRIM(MAX(date)), -4) AS INTEGER) AS max_year
        FROM reviews WHERE date IS NOT NULL
    """, engine).iloc[0]

@st.cache_data(show_spinner="Running ML clustering… this may take a moment")
def run_benchmarking(_sample: bool):
    """Run the full benchmarking pipeline (cached for the session)."""
    df       = get_reviews_df(sample=_sample)
    features = extract_hotel_features(df, verbose=False)
    features = filter_low_signal_hotels(features, verbose=False)
    features, sil_score, profiles = create_comparable_groups(features, n_clusters=7, verbose=False)
    return df, features, sil_score, profiles


# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/hotel-building.png", width=64)
    st.markdown("## 🏨 Hotel Analytics")
    st.caption(f"IS5126 · Group 4 · {db_info}")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📊  Overview",
            "🏨  Hotel Explorer",
            "🎯  Competitive Benchmarking",
            "📈  Performance Trends",
            "🔍  Review Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Data: TripAdvisor hotel reviews (2008‑2012)")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page.startswith("📊"):
    st.title("📊 Dataset Overview")
    st.caption("Key metrics and satisfaction drivers across the entire dataset")

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Reviews", f"{load_review_count():,}")
    with c2:
        st.metric("Hotels", f"{load_hotel_count():,}")
    with c3:
        st.metric("Avg Rating", f"{load_avg_rating():.2f} / 5.0")
    with c4:
        dr = load_date_range()
        st.metric("Time Span", f"{dr['min_year']} – {dr['max_year']}")

    st.divider()

    # ── satisfaction drivers ──
    st.subheader("🎯 Satisfaction Drivers")
    aspects      = ["rating_service", "rating_cleanliness", "rating_value",
                     "rating_location", "rating_rooms", "rating_sleep_quality"]
    aspect_names = [a.replace("rating_", "").replace("_", " ").title() for a in aspects]

    avgs = pd.read_sql(
        f"SELECT {', '.join(f'AVG({a}) AS {a}' for a in aspects)} FROM reviews", engine
    ).iloc[0]

    colors = [GOOD if v >= 4.0 else (WARN if v >= 3.5 else BAD) for v in avgs.values]

    fig = go.Figure(go.Bar(
        x=aspect_names, y=avgs.values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in avgs.values],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 5], title="Average Rating"),
        height=400, showlegend=False,
        margin=dict(t=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── two‑column: distribution + top hotels ──
    left, right = st.columns(2)

    with left:
        st.subheader("Rating Distribution")
        dist = pd.read_sql(
            "SELECT rating_overall AS rating, COUNT(*) AS count FROM reviews GROUP BY rating ORDER BY rating",
            engine,
        )
        fig = px.bar(
            dist, x="rating", y="count",
            color="count",
            color_continuous_scale=["#dfe6e9", ACCENT],
            labels={"rating": "Overall Rating", "count": "Reviews"},
        )
        fig.update_layout(height=370, showlegend=False, coloraxis_showscale=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top 10 Hotels by Volume")
        top = pd.read_sql("""
            SELECT offering_id AS Hotel,
                   COUNT(*)           AS Reviews,
                   ROUND(AVG(rating_overall), 2) AS "Avg Rating"
            FROM reviews
            GROUP BY offering_id
            ORDER BY Reviews DESC
            LIMIT 10
        """, engine)
        st.dataframe(top, use_container_width=True, hide_index=True)

    # ── correlation heatmap ──
    st.subheader("📐 Aspect Rating Correlations")
    corr_data = pd.read_sql(f"SELECT {', '.join(aspects)} FROM reviews", engine)
    corr = corr_data.corr()
    corr.index   = aspect_names
    corr.columns = aspect_names
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#dfe6e9", ACCENT_LITE, ACCENT],
        zmin=0, zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=420, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — HOTEL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("🏨"):
    st.title("🏨 Hotel Explorer")
    st.caption("Deep‑dive into an individual hotel's performance versus its cluster peers")

    # Hotel selector
    hotels_list = pd.read_sql("""
        SELECT DISTINCT offering_id, COUNT(*) AS review_count
        FROM reviews GROUP BY offering_id
        HAVING COUNT(*) >= 5
        ORDER BY review_count DESC LIMIT 200
    """, engine)

    selected = st.selectbox(
        "Select a hotel",
        hotels_list["offering_id"].tolist(),
        format_func=lambda x: f"Hotel {x}  ({hotels_list[hotels_list['offering_id']==x]['review_count'].iloc[0]} reviews)",
    )

    if selected:
        # Fetch hotel stats
        hotel_stats = pd.read_sql(
            text("""
                SELECT COUNT(*) AS reviews,
                       AVG(rating_overall)      AS overall,
                       AVG(rating_service)       AS service,
                       AVG(rating_cleanliness)   AS cleanliness,
                       AVG(rating_value)         AS value,
                       AVG(rating_rooms)         AS rooms,
                       AVG(rating_location)      AS location,
                       AVG(rating_sleep_quality) AS sleep_quality
                FROM reviews WHERE offering_id = :id
            """),
            engine, params={"id": selected},
        ).iloc[0]

        # ── KPIs ──
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Overall Rating", f"{hotel_stats['overall']:.2f} / 5.0")
        with c2:
            st.metric("Total Reviews", f"{int(hotel_stats['reviews']):,}")
        with c3:
            all_avgs = pd.read_sql(
                "SELECT AVG(rating_overall) AS avg FROM reviews GROUP BY offering_id", engine
            )
            pct = (all_avgs["avg"] < hotel_stats["overall"]).sum() / len(all_avgs) * 100
            st.metric("Percentile Rank", f"{pct:.0f}th")

        st.divider()
        col_radar, col_bar = st.columns(2)

        # ── radar: hotel vs cluster peers ──
        with col_radar:
            st.subheader("Radar — Hotel vs Cluster Peers")
            with st.spinner("Computing clusters…"):
                _, bench_feats, _, bench_profiles = run_benchmarking(using_sample)

            categories = ["Service", "Cleanliness", "Value", "Rooms", "Location", "Sleep Quality"]
            hotel_vals = [hotel_stats[c.lower().replace(" ", "_")] for c in categories]

            # Cluster peer average (fallback to industry)
            if selected in bench_feats["offering_id"].values:
                cluster_id = bench_feats.loc[bench_feats["offering_id"] == selected, "cluster"].iloc[0]
                peers = bench_feats[bench_feats["cluster"] == cluster_id]
                peer_label = f"Cluster {cluster_id} Avg"
                aspect_map = {"Service": "avg_service", "Cleanliness": "avg_cleanliness",
                              "Value": "avg_value", "Rooms": "avg_rooms",
                              "Location": "avg_location", "Sleep Quality": "avg_sleep"}
                peer_vals = [peers[aspect_map[c]].mean() for c in categories]
            else:
                industry = pd.read_sql("""
                    SELECT AVG(rating_service) AS service, AVG(rating_cleanliness) AS cleanliness,
                           AVG(rating_value) AS value, AVG(rating_rooms) AS rooms,
                           AVG(rating_location) AS location, AVG(rating_sleep_quality) AS sleep_quality
                    FROM reviews
                """, engine).iloc[0]
                peer_vals = [industry[c.lower().replace(" ", "_")] for c in categories]
                peer_label = "Industry Avg"

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=hotel_vals + [hotel_vals[0]],
                theta=categories + [categories[0]],
                fill="toself", name=f"Hotel {selected}",
                line_color=ACCENT, fillcolor="rgba(108,92,231,.15)",
            ))
            fig.add_trace(go.Scatterpolar(
                r=peer_vals + [peer_vals[0]],
                theta=categories + [categories[0]],
                fill="toself", name=peer_label,
                line_color="#636e72", fillcolor="rgba(99,110,114,.10)",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                height=420, margin=dict(t=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── aspect bar chart ──
        with col_bar:
            st.subheader("Aspect Breakdown")
            aspect_df = pd.DataFrame({
                "Aspect": categories,
                "Rating": hotel_vals,
            })
            colors_bar = [GOOD if v >= 4 else (WARN if v >= 3.5 else BAD) for v in hotel_vals]
            fig = go.Figure(go.Bar(
                x=aspect_df["Aspect"], y=aspect_df["Rating"],
                marker_color=colors_bar,
                text=[f"{v:.2f}" for v in hotel_vals],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 5], title="Rating"),
                height=420, showlegend=False, margin=dict(t=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── monthly rating trend ──
        st.subheader("📈 Rating Trend Over Time")
        trend = pd.read_sql(
            text("""
                SELECT
                    CAST(SUBSTR(TRIM(date), -4) AS INTEGER) AS year,
                    AVG(rating_overall) AS avg_rating,
                    COUNT(*) AS n
                FROM reviews
                WHERE offering_id = :id AND date IS NOT NULL AND LENGTH(TRIM(date)) >= 4
                GROUP BY year ORDER BY year
            """),
            engine, params={"id": selected},
        )

        if len(trend) > 1:
            fig = px.line(
                trend, x="year", y="avg_rating",
                markers=True,
                labels={"year": "Year", "avg_rating": "Avg Rating"},
            )
            fig.update_traces(line_color=ACCENT, line_width=3, marker_size=9)
            fig.update_layout(yaxis=dict(range=[1, 5]), height=320, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough yearly data points to chart a trend for this hotel.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPETITIVE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("🎯"):
    st.title("🎯 Competitive Benchmarking")
    st.caption("ML‑powered hotel segmentation using K‑means clustering on text‑mined features")

    with st.spinner("Running benchmarking pipeline…"):
        df_all, features, sil_score, profiles = run_benchmarking(using_sample)

    # ── cluster overview ──
    st.subheader("Cluster Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Segments Found", len(profiles))
    with c2:
        st.metric("Silhouette Score", f"{sil_score:.3f}")
    with c3:
        st.metric("Hotels Clustered", f"{len(features):,}")

    # Profile table
    price_map = {0: "Budget", 1: "Mid‑range", 2: "Upscale", 3: "Luxury"}
    profile_rows = []
    for cid, p in sorted(profiles.items()):
        profile_rows.append({
            "Cluster": cid,
            "Hotels": p["n_hotels"],
            "Avg Rating": round(p["avg_rating"], 2),
            "Price Tier": price_map.get(p.get("price_tier", 1), "—"),
            "Location": p.get("common_location", "—").title(),
            "Type": p.get("common_type", "—").title(),
            "Amenities": ", ".join(p.get("common_amenities", [])) or "—",
        })
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # ── cluster distribution scatter ──
    st.subheader("Cluster Distribution")
    scatter_df = features[["offering_id", "avg_rating", "n_reviews", "cluster"]].copy()
    scatter_df["cluster_label"] = scatter_df["cluster"].apply(lambda c: f"Cluster {c}")
    fig = px.scatter(
        scatter_df, x="n_reviews", y="avg_rating",
        color="cluster_label",
        size="n_reviews",
        color_discrete_sequence=PALETTE,
        labels={"n_reviews": "Review Count", "avg_rating": "Avg Rating", "cluster_label": "Segment"},
        hover_data={"offering_id": True},
    )
    fig.update_layout(height=480, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── per‑hotel recommendations ──
    st.divider()
    st.subheader("🔎 Hotel‑Level Gap Analysis & Recommendations")
    hotel_options = features.sort_values("n_reviews", ascending=False)["offering_id"].tolist()
    sel_hotel = st.selectbox(
        "Pick a hotel to analyse",
        hotel_options[:200],
        format_func=lambda x: f"Hotel {x}  (Cluster {features.loc[features['offering_id']==x, 'cluster'].iloc[0]})",
    )

    if sel_hotel:
        recs = generate_actionable_recommendations(sel_hotel, features, df_all, top_n=5)

        if recs:
            for i, r in enumerate(recs, 1):
                with st.container():
                    priority_icon = "🔴" if r["gap"] > 0.5 else "🟡"
                    st.markdown(f"#### {priority_icon} {i}. Improve **{r['aspect'].title()}**")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Your Score", f"{r['current_score']:.2f}")
                    m2.metric("Peer Median", f"{r['peer_median']:.2f}")
                    m3.metric("Gap", f"-{r['gap']:.2f}", delta=f"-{r['gap']:.2f}", delta_color="inverse")
                    m4.metric("Est. ROI", f"{r['roi_estimate']:.0f}%")

                    if r.get("best_practices"):
                        st.markdown("**Best practices from top performers:**")
                        for bp in r["best_practices"]:
                            st.markdown(f"- {bp}")
                    st.divider()
        else:
            st.success("🎉 This hotel performs at or above its cluster peers across all aspects!")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — PERFORMANCE TRENDS
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("📈"):
    st.title("📈 Performance Trends")
    st.caption("Year‑over‑year rating and volume trends")

    # ── dual axis chart ──
    st.subheader("Review Volume & Rating Trend")
    yearly = pd.read_sql("""
        SELECT
            CAST(SUBSTR(TRIM(date), -4) AS INTEGER) AS year,
            AVG(rating_overall) AS avg_rating,
            COUNT(*) AS num_reviews
        FROM reviews
        WHERE date IS NOT NULL AND LENGTH(TRIM(date)) >= 4
        GROUP BY year ORDER BY year
    """, engine)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["num_reviews"],
        name="Review Volume", yaxis="y2",
        marker_color=ACCENT_LITE, opacity=0.45,
    ))
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["avg_rating"],
        mode="lines+markers", name="Avg Rating",
        line=dict(color=ACCENT, width=3), marker=dict(size=10),
    ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis=dict(title="Average Rating", range=[3, 5]),
        yaxis2=dict(title="Review Count", overlaying="y", side="right"),
        hovermode="x unified", height=480, margin=dict(t=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── aspect trends over time ──
    st.subheader("Aspect Trends Over Time")
    aspects_trend = ["rating_service", "rating_cleanliness", "rating_value",
                     "rating_rooms", "rating_location", "rating_sleep_quality"]
    agg_expr = ", ".join(f"AVG({a}) AS {a.replace('rating_', '')}" for a in aspects_trend)
    aspect_yearly = pd.read_sql(f"""
        SELECT CAST(SUBSTR(TRIM(date), -4) AS INTEGER) AS year, {agg_expr}
        FROM reviews
        WHERE date IS NOT NULL AND LENGTH(TRIM(date)) >= 4
        GROUP BY year ORDER BY year
    """, engine)

    aspect_melted = aspect_yearly.melt(
        id_vars="year", var_name="Aspect", value_name="Avg Rating",
    )
    aspect_melted["Aspect"] = aspect_melted["Aspect"].str.replace("_", " ").str.title()

    fig = px.line(
        aspect_melted, x="year", y="Avg Rating", color="Aspect",
        markers=True, color_discrete_sequence=PALETTE,
        labels={"year": "Year"},
    )
    fig.update_layout(height=420, margin=dict(t=10),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.22))
    st.plotly_chart(fig, use_container_width=True)

    # ── top / bottom performers ──
    left, right = st.columns(2)
    with left:
        st.subheader("🏆 Top Performers")
        top = pd.read_sql("""
            SELECT offering_id AS Hotel, COUNT(*) AS Reviews,
                   ROUND(AVG(rating_overall), 2) AS Rating
            FROM reviews GROUP BY offering_id
            HAVING Reviews >= 20
            ORDER BY Rating DESC, Reviews DESC LIMIT 10
        """, engine)
        st.dataframe(top, use_container_width=True, hide_index=True)

    with right:
        st.subheader("⚠️ Underperformers")
        bottom = pd.read_sql("""
            SELECT offering_id AS Hotel, COUNT(*) AS Reviews,
                   ROUND(AVG(rating_overall), 2) AS Rating
            FROM reviews GROUP BY offering_id
            HAVING Reviews >= 20
            ORDER BY Rating ASC, Reviews DESC LIMIT 10
        """, engine)
        st.dataframe(bottom, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — REVIEW INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page.startswith("🔍"):
    st.title("🔍 Review Insights")
    st.caption("Text, helpfulness, and reviewer analytics")

    left, right = st.columns(2)

    # ── review length distribution ──
    with left:
        st.subheader("Review Length Distribution")
        lengths = pd.read_sql(
            "SELECT LENGTH(text) AS len FROM reviews WHERE text IS NOT NULL", engine
        )
        fig = px.histogram(
            lengths, x="len", nbins=50,
            color_discrete_sequence=[ACCENT],
            labels={"len": "Character Count", "count": "Reviews"},
        )
        fig.update_layout(height=380, margin=dict(t=10), showlegend=False,
                          yaxis_title="Reviews")
        st.plotly_chart(fig, use_container_width=True)

    # ── mobile vs desktop ──
    with right:
        st.subheader("Mobile vs Desktop Reviews")
        mobile = pd.read_sql("""
            SELECT
                CASE WHEN via_mobile = 1 THEN 'Mobile' ELSE 'Desktop' END AS source,
                COUNT(*) AS count,
                ROUND(AVG(rating_overall), 2) AS avg_rating
            FROM reviews GROUP BY source
        """, engine)
        fig = px.pie(
            mobile, values="count", names="source",
            color_discrete_sequence=[ACCENT, ACCENT_LITE],
            hole=0.45,
        )
        fig.update_traces(textinfo="label+percent", textfont_size=14)
        fig.update_layout(height=380, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Show avg ratings beside
        for _, row in mobile.iterrows():
            st.caption(f"**{row['source']}**: avg rating {row['avg_rating']}")

    # ── most helpful reviews ──
    st.subheader("📌 Most Helpful Reviews")
    helpful = pd.read_sql("""
        SELECT
            r.offering_id AS Hotel,
            r.title        AS Title,
            r.num_helpful_votes AS "Helpful Votes",
            r.rating_overall    AS Rating,
            SUBSTR(r.text, 1, 200) || '…' AS Snippet
        FROM reviews r
        WHERE r.num_helpful_votes > 0
        ORDER BY r.num_helpful_votes DESC
        LIMIT 15
    """, engine)
    st.dataframe(helpful, use_container_width=True, hide_index=True)

    # ── reviewer geography ──
    st.subheader("🌍 Top Reviewer Locations")
    geo = pd.read_sql("""
        SELECT a.location AS Location, COUNT(*) AS Reviewers
        FROM authors a
        WHERE a.location IS NOT NULL AND a.location != ''
        GROUP BY a.location
        ORDER BY Reviewers DESC
        LIMIT 20
    """, engine)

    fig = px.bar(
        geo, x="Reviewers", y="Location", orientation="h",
        color="Reviewers",
        color_continuous_scale=["#dfe6e9", ACCENT],
    )
    fig.update_layout(
        height=max(360, len(geo) * 28),
        margin=dict(t=10), yaxis=dict(autorange="reversed"),
        showlegend=False, coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("Hotel Analytics Dashboard · IS5126 Assignment 1 · Group 4 · Data: 2008–2012")