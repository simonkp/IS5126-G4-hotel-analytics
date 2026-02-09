"""
Competitive benchmarking: identify comparable hotel groups and compare performance.
Assignment: "Who are my real competitors? How do we systematically identify truly
comparable properties? What are similar hotels doing better? Where to focus improvement?"
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import re
from typing import Dict, List, Tuple
from src.utils import get_engine


def get_reviews_df(sample: bool = False) -> pd.DataFrame:
    """Load reviews from DB (optionally sample DB for quick runs)."""
    engine = get_engine(sample=sample)
    return pd.read_sql(
        "SELECT id, offering_id, author_id, rating_overall, rating_service, rating_cleanliness, "
        "rating_value, rating_location, rating_sleep_quality, rating_rooms, date, num_helpful_votes, text FROM reviews",
        engine,
    )

def extract_hotel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract comprehensive hotel features for clustering.
    
    Features extracted:
    1. Rating metrics (mean, std)
    2. Review volume
    3. Text-derived features (amenities, location type, price tier)
    4. Reviewer engagement (helpful votes)
    """
    print("Extracting hotel-level features...")
    
    # Aggregate rating features
    agg = df.groupby("offering_id").agg(
        n_reviews=("id", "count"),
        avg_rating=("rating_overall", "mean"),
        std_rating=("rating_overall", "std"),
        avg_service=("rating_service", "mean"),
        avg_cleanliness=("rating_cleanliness", "mean"),
        avg_value=("rating_value", "mean"),
        avg_location=("rating_location", "mean"),
        avg_rooms=("rating_rooms", "mean"),
        avg_sleep=("rating_sleep_quality", "mean"),
        avg_helpful_votes=("num_helpful_votes", "mean"),
    ).reset_index()
    
    # Extract text features
    print("Analyzing review text for hotel characteristics...")
    text_features_list = []
    
    for hotel_id in df['offering_id'].unique():
        hotel_reviews = df[df['offering_id'] == hotel_id]
        features = extract_text_features_for_hotel(hotel_reviews)
        features['offering_id'] = hotel_id
        text_features_list.append(features)
    
    text_features = pd.DataFrame(text_features_list)
    
    # Merge
    features = agg.merge(text_features, on='offering_id', how='left')
    
    # Fill NaN
    features = features.fillna(0)
    
    return features


def extract_text_features_for_hotel(hotel_reviews: pd.DataFrame) -> Dict:
    """
    Extract features from review text for a single hotel.
    
    Returns dict with:
    - price_tier: 0=budget, 1=mid-range, 2=upscale, 3=luxury
    - location_type: beach, downtown, suburban, airport
    - hotel_type: resort, business, boutique
    - amenities: pool, spa, gym, restaurant scores
    """
    # Combine all text
    all_text = ' '.join(hotel_reviews['text'].dropna().astype(str)).lower()
    
    # Price tier (0-3)
    luxury_terms = ['luxury', 'luxurious', 'upscale', 'premium', 'high-end', 'five-star', '5-star', 'deluxe']
    upscale_terms = ['upscale', 'elegant', 'sophisticated', 'refined']
    budget_terms = ['budget', 'affordable', 'cheap', 'economical', 'value', 'inexpensive']
    
    luxury_score = sum(1 for term in luxury_terms if term in all_text)
    upscale_score = sum(1 for term in upscale_terms if term in all_text)
    budget_score = sum(1 for term in budget_terms if term in all_text)
    
    if luxury_score >= 3:
        price_tier = 3
    elif upscale_score >= 2 or luxury_score >= 1:
        price_tier = 2
    elif budget_score >= 2:
        price_tier = 0
    else:
        price_tier = 1
    
    # Location type (binary flags)
    is_beach = int(any(term in all_text for term in ['beach', 'ocean', 'oceanfront', 'beachfront', 'seaside']))
    is_downtown = int(any(term in all_text for term in ['downtown', 'city center', 'midtown', 'central location']))
    is_suburban = int(any(term in all_text for term in ['suburban', 'outskirts', 'quiet area']))
    is_airport = int(any(term in all_text for term in ['airport', 'near airport']))
    
    # Hotel type
    is_resort = int('resort' in all_text)
    is_business = int(any(term in all_text for term in ['business hotel', 'business center', 'conference', 'convention']))
    is_boutique = int('boutique' in all_text)
    
    # Amenities (count mentions)
    has_pool = sum(1 for _ in re.finditer(r'\bpool\b', all_text))
    has_spa = sum(1 for _ in re.finditer(r'\bspa\b', all_text))
    has_gym = sum(1 for _ in re.finditer(r'\bgym\b|\bfitness\b', all_text))
    has_restaurant = sum(1 for _ in re.finditer(r'\brestaurant\b', all_text))
    has_bar = sum(1 for _ in re.finditer(r'\bbar\b|\blounge\b', all_text))
    has_parking = sum(1 for _ in re.finditer(r'\bparking\b', all_text))
    
    # Normalize amenity scores (mentions per review)
    n_reviews = len(hotel_reviews)
    
    return {
        'price_tier': price_tier,
        'is_beach': is_beach,
        'is_downtown': is_downtown,
        'is_suburban': is_suburban,
        'is_airport': is_airport,
        'is_resort': is_resort,
        'is_business': is_business,
        'is_boutique': is_boutique,
        'pool_score': has_pool / max(n_reviews, 1),
        'spa_score': has_spa / max(n_reviews, 1),
        'gym_score': has_gym / max(n_reviews, 1),
        'restaurant_score': has_restaurant / max(n_reviews, 1),
        'bar_score': has_bar / max(n_reviews, 1),
        'parking_score': has_parking / max(n_reviews, 1),
    }


def create_comparable_groups(
    features: pd.DataFrame,
    n_clusters: int = 6,
) -> Tuple[pd.DataFrame, float, Dict]:
    """
    Create comparable hotel groups using K-means clustering.
    
    Returns:
    - features with 'cluster' column
    - silhouette_score
    - cluster_profiles dict
    Strategy: 
    - Use only ONE rating metric (avg_rating) instead of all aspects
    - Weight hotel characteristics (location, type, amenities) more heavily
    - This creates clusters based on HOTEL TYPE, not just quality
    """
    print(f"Creating {n_clusters} comparable groups...")
    
    # Select features for clustering
    clustering_features = [
        'avg_rating',
        'n_reviews',
        'price_tier',
        'is_beach',
        'is_downtown',
        'is_resort',  
        'is_business',
        'pool_score',
        'spa_score',
        'gym_score',
    ]
    
    # Prepare data
    X = features[clustering_features].copy()
    
    # Log transform review count to reduce skew
    X['n_reviews'] = np.log1p(X['n_reviews'])
    
    # IMPORTANT: Weight non-rating features more
    # By duplicating them, we make clustering consider them more
    X_weighted = X.copy()
    
    # Triple-weight the differentiating features
    for col in ['price_tier', 'is_beach', 'is_downtown', 'is_resort', 'is_business']:
        X_weighted[f'{col}_2'] = X[col]
        X_weighted[f'{col}_3'] = X[col]
    
    # Double-weight amenities
    for col in ['pool_score', 'spa_score', 'gym_score']:
        X_weighted[f'{col}_2'] = X[col]
    
    # Standardize (critical for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_weighted)
    
    # Find optimal number of clusters (try n_clusters ± 2)
    best_score = -1
    best_n = n_clusters
    best_labels = None
    
    for n in range(max(4, n_clusters-2), min(12, n_clusters+4)):
        kmeans_temp = KMeans(n_clusters=n, random_state=42, n_init=30)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        
        # Only consider if clusters are reasonably sized
        counts = pd.Series(labels_temp).value_counts()
        if counts.min() < 5:  # Skip if any cluster has <5 hotels
            continue
            
        score = silhouette_score(X_scaled, labels_temp)
        
        if score > best_score:
            best_score = score
            best_n = n
            best_labels = labels_temp
    
    print(f"Optimal n_clusters: {best_n} (silhouette: {best_score:.3f})")
    
    # Use best clustering
    features['cluster'] = best_labels
    sil_score = best_score
    
    # Create cluster profiles
    cluster_profiles = {}
    for cluster_id in range(best_n):
        cluster_hotels = features[features['cluster'] == cluster_id]
        
        if len(cluster_hotels) == 0:
            continue
        
        profile = {
            'n_hotels': len(cluster_hotels),
            'avg_rating': cluster_hotels['avg_rating'].mean(),
            'avg_reviews': cluster_hotels['n_reviews'].mean(),
            'price_tier': cluster_hotels['price_tier'].mode().iloc[0] if len(cluster_hotels) > 0 else 1,
            'common_location': _get_common_location(cluster_hotels),
            'common_type': _get_common_type(cluster_hotels),
            'common_amenities': _get_common_amenities(cluster_hotels),
        }
        
        cluster_profiles[cluster_id] = profile
    
    return features, sil_score, cluster_profiles


def _get_common_location(cluster_df: pd.DataFrame) -> str:
    """Determine most common location type in cluster."""
    location_cols = ['is_beach', 'is_downtown', 'is_suburban', 'is_airport']
    location_sums = cluster_df[location_cols].sum()
    
    if location_sums.max() == 0:
        return "various"
    
    return location_sums.idxmax().replace('is_', '')


def _get_common_type(cluster_df: pd.DataFrame) -> str:
    """Determine most common hotel type in cluster."""
    type_cols = ['is_resort', 'is_business', 'is_boutique']
    type_sums = cluster_df[type_cols].sum()
    
    if type_sums.max() == 0:
        return "general"
    
    return type_sums.idxmax().replace('is_', '')


def _get_common_amenities(cluster_df: pd.DataFrame) -> List[str]:
    """List common amenities (present in >50% of cluster)."""
    amenity_cols = ['pool_score', 'spa_score', 'gym_score', 'restaurant_score', 'bar_score']
    amenities = []
    
    for col in amenity_cols:
        # If more than 50% of hotels mention this amenity
        if (cluster_df[col] > 0).sum() / len(cluster_df) > 0.5:
            amenities.append(col.replace('_score', ''))
    
    return amenities


def generate_actionable_recommendations(
    hotel_id: int,
    features: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = 3,
) -> List[Dict]:
    """
    Generate specific, actionable recommendations with ROI.
    
    Returns list of dicts with:
    - aspect: which aspect to improve
    - current_score: hotel's current score
    - peer_median: peer group median
    - peer_top_quartile: top 25% performance
    - gap: difference from peer median
    - best_practices: list of what top performers do
    - estimated_impact: expected rating improvement
    - roi_estimate: estimated ROI%
    """
    if hotel_id not in features['offering_id'].values:
        return []
    
    hotel = features[features['offering_id'] == hotel_id].iloc[0]
    cluster_id = hotel['cluster']
    
    # Get peer hotels (same cluster, excluding target)
    peers = features[(features['cluster'] == cluster_id) & (features['offering_id'] != hotel_id)]
    
    if len(peers) < 3:
        return []
    
    recommendations = []
    aspects = ['service', 'cleanliness', 'value', 'rooms', 'sleep']
    
    for aspect in aspects:
        col = f'avg_{aspect}'
        
        if col not in hotel.index:
            continue
        
        current_score = hotel[col]
        peer_median = peers[col].median()
        peer_top_quartile = peers[col].quantile(0.75)
        gap = peer_median - current_score
        
        # Only recommend if gap > 0.3 (significant)
        if gap > 0.3:
            # Find top performers in this aspect
            top_performers = peers.nlargest(5, col)
            
            # Analyze best practices
            best_practices = _analyze_best_practices(
                top_performers['offering_id'].tolist(),
                aspect,
                df
            )
            
            # Estimate impact and ROI
            estimated_impact = min(gap * 0.7, 1.0)  # Conservative: can close 70% of gap
            roi = _calculate_roi(estimated_impact, aspect)
            
            recommendations.append({
                'aspect': aspect,
                'current_score': round(current_score, 2),
                'peer_median': round(peer_median, 2),
                'peer_top_quartile': round(peer_top_quartile, 2),
                'gap': round(gap, 2),
                'best_practices': best_practices,
                'estimated_impact': round(estimated_impact, 2),
                'roi_estimate': round(roi, 1),
                'priority': _calculate_priority(gap, roi),
            })
    
    # Sort by priority (gap * ROI)
    recommendations.sort(key=lambda x: x['priority'], reverse=True)
    
    return recommendations[:top_n]


def _analyze_best_practices(
    hotel_ids: List[int],
    aspect: str,
    df: pd.DataFrame,
) -> List[str]:
    """
    Analyze what top performers do differently.
    Uses text analysis of reviews.
    """
    # Get reviews for top performers
    top_reviews = df[df['offering_id'].isin(hotel_ids)]['text'].dropna()
    all_text = ' '.join(top_reviews.astype(str)).lower()
    
    practices = []
    
    # Aspect-specific analysis
    if aspect == 'cleanliness':
        if 'daily housekeeping' in all_text or 'daily cleaning' in all_text:
            practices.append("Daily housekeeping service mentioned frequently")
        if 'spotless' in all_text or 'immaculate' in all_text:
            practices.append("Exceptional cleanliness standards ('spotless', 'immaculate')")
        if 'modern bathroom' in all_text or 'renovated bathroom' in all_text:
            practices.append("Recently renovated, modern bathrooms")
        
    elif aspect == 'service':
        if 'friendly staff' in all_text or 'helpful staff' in all_text:
            freq = (all_text.count('friendly staff') + all_text.count('helpful staff')) / len(top_reviews)
            if freq > 0.2:
                practices.append(f"Consistently friendly staff (mentioned in {freq*100:.0f}% of reviews)")
        if 'remembered' in all_text or 'personal' in all_text:
            practices.append("Personalized service - staff remember guest names/preferences")
        if '24 hour' in all_text or '24/7' in all_text:
            practices.append("24/7 service availability")
    
    elif aspect == 'value':
        if 'worth' in all_text or 'good value' in all_text:
            practices.append("Guests consistently mention good value for money")
        if 'included' in all_text or 'complimentary' in all_text:
            practices.append("Complimentary amenities add perceived value")
    
    elif aspect == 'rooms':
        if 'spacious' in all_text:
            practices.append("Spacious rooms frequently mentioned")
        if 'comfortable bed' in all_text or 'great bed' in all_text:
            practices.append("High-quality, comfortable beds")
        if 'modern' in all_text or 'updated' in all_text:
            practices.append("Modern, updated room furnishings")
    
    elif aspect == 'sleep':
        if 'quiet' in all_text:
            practices.append("Effective noise control - quiet rooms")
        if 'blackout' in all_text:
            practices.append("Blackout curtains for better sleep")
        if 'comfortable bed' in all_text:
            practices.append("High-quality mattresses and bedding")
    
    # If no specific practices found, provide generic
    if not practices:
        practices.append(f"Top performers excel in {aspect} - analyze their detailed reviews")
    
    return practices


def _calculate_roi(rating_impact: float, aspect: str) -> float:
    """
    Calculate ROI estimate based on rating improvement.
    
    Industry benchmarks:
    - 0.1 rating increase → 2-3% booking increase
    - Different aspects have different improvement costs
    """
    # Booking increase % (conservative)
    booking_increase = rating_impact * 20  # 0.5 impact → 10% increase
    
    # Estimated costs by aspect (USD)
    costs = {
        'cleanliness': 8_000,      # Enhanced cleaning protocols, training
        'service': 20_000,         # Staff training programs
        'rooms': 75_000,           # Room renovations (partial)
        'value': 5_000,            # Process improvements, added amenities
        'sleep': 15_000,           # Soundproofing, better mattresses
    }
    
    cost = costs.get(aspect, 15_000)
    
    # Assume average hotel annual revenue: $1.5M
    avg_annual_revenue = 1_500_000
    revenue_increase = avg_annual_revenue * (booking_increase / 100)
    
    # Net benefit over 1 year
    net_benefit = revenue_increase - cost
    roi = (net_benefit / cost) * 100
    
    return roi


def _calculate_priority(gap: float, roi: float) -> float:
    """Calculate priority score (gap * ROI weight)."""
    # Prioritize: large gaps with positive ROI
    if roi < 0:
        return 0
    return gap * (1 + roi / 100)



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
    features = extract_hotel_features(df)
    print(f"Extracted features for {len(features)} hotels")
    
    # Create groups
    features, sil_score, profiles = create_comparable_groups(features, n_clusters=8)
    print(f"\nCluster Profiles:")
    for cluster_id, profile in profiles.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Hotels: {profile['n_hotels']}")
        print(f"  Avg Rating: {profile['avg_rating']:.2f}")
        print(f"  Price Tier: {['Budget', 'Mid-range', 'Upscale', 'Luxury'][profile['price_tier']]}")
        print(f"  Location: {profile['common_location']}")
        print(f"  Type: {profile['common_type']}")
        print(f"  Amenities: {', '.join(profile['common_amenities']) if profile['common_amenities'] else 'None'}")
    
    # Test recommendations
    sample_hotel = features['offering_id'].iloc[0]
    print(f"\nGenerating recommendations for hotel {sample_hotel}:")
    recs = generate_actionable_recommendations(sample_hotel, features, df)
    
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. Improve {rec['aspect'].upper()}")
        print(f"   Current: {rec['current_score']:.2f} | Peer Median: {rec['peer_median']:.2f}")
        print(f"   Gap: {rec['gap']:.2f} | Est. Impact: +{rec['estimated_impact']:.2f}")
        print(f"   ROI: {rec['roi_estimate']:.0f}%")
        print(f"   Best Practices:")
        for practice in rec['best_practices']:
            print(f"     - {practice}")
