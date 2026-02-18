"""
Enhanced Hotel Analytics Dashboard with competitive benchmarking.
Run: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import text
from src.utils import get_engine, get_db_path

st.set_page_config(page_title="Hotel Analytics", layout="wide")

# Load database
full_path = get_db_path(sample=False)
sample_path = get_db_path(sample=True)

if full_path.exists():
    engine = get_engine(sample=False)
    db_info = "Full Database (79K reviews)"
else:
    engine = get_engine(sample=True)
    db_info = "Sample Database (5K reviews)"

# Header
st.title("🏨 Hotel Analytics Dashboard")
st.caption(f"IS5126 Applied Analytics | {db_info}")

# Sidebar navigation
page = st.sidebar.radio("📊 Navigation", [
    "Overview",
    "Competitive Analysis",
    "Performance Trends"
])

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================

if page == "Overview":
    st.header("Dataset Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_reviews = pd.read_sql("SELECT COUNT(*) AS n FROM reviews", engine).iloc[0]["n"]
        st.metric("Total Reviews", f"{n_reviews:,}")
    
    with col2:
        n_hotels = pd.read_sql("SELECT COUNT(DISTINCT offering_id) AS n FROM reviews", engine).iloc[0]["n"]
        st.metric("Hotels", f"{n_hotels:,}")
    
    with col3:
        avg_rating = pd.read_sql("SELECT AVG(rating_overall) AS avg FROM reviews", engine).iloc[0]["avg"]
        st.metric("Avg Rating", f"{avg_rating:.2f}/5.0")
    
    with col4:
        date_range = pd.read_sql("""
            SELECT 
                CAST(SUBSTR(TRIM(MIN(date)), -4) AS INTEGER) as min_year,
                CAST(SUBSTR(TRIM(MAX(date)), -4) AS INTEGER) as max_year
            FROM reviews WHERE date IS NOT NULL
        """, engine).iloc[0]
        st.metric("Years", f"{date_range['min_year']}-{date_range['max_year']}")
    
    # Satisfaction drivers
    st.subheader("Key Satisfaction Drivers")
    
    aspects = ['rating_service', 'rating_cleanliness', 'rating_value', 
               'rating_location', 'rating_rooms', 'rating_sleep_quality']
    aspect_labels = [a.replace('rating_', '').replace('_', ' ').title() for a in aspects]
    
    aspect_data = pd.read_sql(
        f"SELECT {', '.join(f'AVG({a}) as {a}' for a in aspects)} FROM reviews",
        engine
    ).iloc[0]
    
    fig = go.Figure(data=[
        go.Bar(
            x=aspect_labels,
            y=aspect_data.values,
            marker_color='steelblue',
            text=[f'{v:.2f}' for v in aspect_data.values],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        yaxis_title="Average Rating",
        yaxis=dict(range=[0, 5]),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        dist = pd.read_sql(
            "SELECT rating_overall as rating, COUNT(*) as count FROM reviews GROUP BY rating ORDER BY rating",
            engine
        )
        
        fig = go.Figure(data=[
            go.Bar(x=dist['rating'], y=dist['count'], marker_color='steelblue')
        ])
        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Count",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Hotels")
        top = pd.read_sql("""
            SELECT 
                offering_id as Hotel,
                COUNT(*) as Reviews,
                ROUND(AVG(rating_overall), 2) as Rating
            FROM reviews
            GROUP BY offering_id
            ORDER BY Reviews DESC
            LIMIT 10
        """, engine)
        st.dataframe(top, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 2: COMPETITIVE ANALYSIS
# ============================================================================

elif page == "Competitive Analysis":
    st.header("Competitive Benchmarking")
    
    # Hotel selector
    hotels = pd.read_sql("""
        SELECT DISTINCT offering_id, COUNT(*) as review_count
        FROM reviews
        GROUP BY offering_id
        HAVING COUNT(*) >= 10
        ORDER BY review_count DESC
        LIMIT 100
    """, engine)
    
    selected = st.selectbox(
        "Select hotel to analyze:",
        hotels['offering_id'].tolist(),
        format_func=lambda x: f"Hotel {x} ({hotels[hotels['offering_id']==x]['review_count'].iloc[0]} reviews)"
    )
    
    if selected:
        # Hotel stats
        hotel_stats = pd.read_sql(
            text("""
                SELECT 
                    COUNT(*) as reviews,
                    AVG(rating_overall) as overall,
                    AVG(rating_service) as service,
                    AVG(rating_cleanliness) as cleanliness,
                    AVG(rating_value) as value,
                    AVG(rating_rooms) as rooms,
                    AVG(rating_location) as location
                FROM reviews
                WHERE offering_id = :id
            """),
            engine,
            params={"id": selected},
        ).iloc[0]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Rating", f"{hotel_stats['overall']:.2f}/5.0")
        with col2:
            st.metric("Total Reviews", f"{int(hotel_stats['reviews']):,}")
        with col3:
            # Calculate percentile
            all_ratings = pd.read_sql(
                "SELECT AVG(rating_overall) as avg FROM reviews GROUP BY offering_id",
                engine
            )
            percentile = (all_ratings['avg'] < hotel_stats['overall']).sum() / len(all_ratings) * 100
            st.metric("Percentile Rank", f"{percentile:.0f}%")
        
        # Radar chart
        st.subheader("Performance vs Industry Average")
        
        # Industry averages
        industry = pd.read_sql("""
            SELECT 
                AVG(rating_service) as service,
                AVG(rating_cleanliness) as cleanliness,
                AVG(rating_value) as value,
                AVG(rating_rooms) as rooms,
                AVG(rating_location) as location
            FROM reviews
        """, engine).iloc[0]
        
        categories = ['Service', 'Cleanliness', 'Value', 'Rooms', 'Location']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[hotel_stats['service'], hotel_stats['cleanliness'], hotel_stats['value'],
               hotel_stats['rooms'], hotel_stats['location']],
            theta=categories,
            fill='toself',
            name=f'Hotel {selected}',
            line_color='#1f77b4'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[industry['service'], industry['cleanliness'], industry['value'],
               industry['rooms'], industry['location']],
            theta=categories,
            fill='toself',
            name='Industry Avg',
            line_color='#ff7f0e',
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gap analysis
        st.subheader("Improvement Opportunities")
        
        gaps = []
        for aspect in ['service', 'cleanliness', 'value', 'rooms', 'location']:
            gap = industry[aspect] - hotel_stats[aspect]
            if gap > 0.2:
                gaps.append({
                    'Aspect': aspect.title(),
                    'Your Score': round(hotel_stats[aspect], 2),
                    'Industry Avg': round(industry[aspect], 2),
                    'Gap': round(gap, 2),
                    'Priority': '🔴 High' if gap > 0.5 else '🟡 Medium'
                })
        
        if gaps:
            gaps_df = pd.DataFrame(gaps).sort_values('Gap', ascending=False)
            st.dataframe(gaps_df, use_container_width=True, hide_index=True)
            
            st.info(f"💡 **Focus Area:** {gaps_df.iloc[0]['Aspect']} (gap: {gaps_df.iloc[0]['Gap']:.2f} points)")
        else:
            st.success("✅ Performing above industry average across all aspects!")

# ============================================================================
# PAGE 3: PERFORMANCE TRENDS
# ============================================================================

elif page == "Performance Trends":
    st.header("Performance Trends & Rankings")
    
    # Year-over-year
    st.subheader("Review Volume & Rating Trend")
    
    yearly = pd.read_sql("""
        SELECT 
            CAST(SUBSTR(TRIM(date), -4) AS INTEGER) as year,
            AVG(rating_overall) as avg_rating,
            COUNT(*) as num_reviews
        FROM reviews
        WHERE date IS NOT NULL AND LENGTH(TRIM(date)) >= 4
        GROUP BY year
        ORDER BY year
    """, engine)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['avg_rating'],
        mode='lines+markers',
        name='Avg Rating',
        yaxis='y',
        line=dict(color='steelblue', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Bar(
        x=yearly['year'],
        y=yearly['num_reviews'],
        name='Review Volume',
        yaxis='y2',
        opacity=0.3,
        marker_color='orange'
    ))
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis=dict(title="Average Rating", range=[3, 5]),
        yaxis2=dict(title="Review Count", overlaying='y', side='right'),
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top/Bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performers")
        top = pd.read_sql("""
            SELECT 
                offering_id as Hotel,
                COUNT(*) as Reviews,
                ROUND(AVG(rating_overall), 2) as Rating
            FROM reviews
            GROUP BY offering_id
            HAVING Reviews >= 20
            ORDER BY Rating DESC, Reviews DESC
            LIMIT 10
        """, engine)
        st.dataframe(top, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("⚠️ Underperformers")
        bottom = pd.read_sql("""
            SELECT 
                offering_id as Hotel,
                COUNT(*) as Reviews,
                ROUND(AVG(rating_overall), 2) as Rating
            FROM reviews
            GROUP BY offering_id
            HAVING Reviews >= 20
            ORDER BY Rating ASC, Reviews DESC
            LIMIT 10
        """, engine)
        st.dataframe(bottom, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.caption("Hotel Analytics Dashboard | IS5126 Assignment 1 | Data: 2008-2012")