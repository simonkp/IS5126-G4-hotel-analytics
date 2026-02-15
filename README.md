# IS5126 G4 – Hotel Analytics Platform

**Course:** IS5126 Hands-on with Applied Analytics  
**Team:** Group 4  
**Assignment:** Data Foundation & Exploratory Analytics (Assignment 1)

---

## 🎯 Executive Summary

**Problem:** Hotel managers lack competitive intelligence tools to identify true competitors and actionable improvement opportunities beyond generic rating comparisons.

**Solution:** ML-powered analytics platform featuring:
- ✅ **7 distinct hotel segments** identified via K-means clustering (silhouette: 0.302)
- ✅ **Text mining** extracts hotel characteristics (location, amenities, price tier) from 80K reviews
- ✅ **ROI-based recommendations** (typical: 500-2,800% ROI)
- ✅ **Interactive dashboard** for performance tracking

**Key Innovation:** Multi-dimensional clustering groups hotels by ACTUAL similarity (beach resorts vs beach resorts), not just rating tiers.

**Result:** 35% variance reduction within clusters = hotels grouped meaningfully, validated statistically and by business logic.

---

## 📊 Technical Highlights

### What Makes This Different

| Feature | Basic Approach | **Our Enhanced Approach** |
|---------|----------------|---------------------------|
| **Grouping Method** | 2D bins (volume × rating) | K-means on 6+ dimensions |
| **Hotel Features** | None | Text-mined: location, type, amenities, price |
| **Validation** | Visual only | Silhouette (0.302) + variance reduction (35%) |
| **Recommendations** | Generic ("improve cleanliness") | Specific + ROI + best practices |
| **Optimization** | Fixed bins | Auto-selects K (tested 5-12) |

### Innovation: Text Feature Extraction

From reviews like *"This luxury beachfront resort has a pool and spa"*, we extract:
- `price_tier = 3` (luxury)
- `is_beach = 1` (beachfront)
- `pool_score = 0.8` (pool mentioned)
- `spa_score = 0.7` (spa mentioned)

**Result:** Beach resorts cluster with beach resorts, NOT with downtown hotels of similar rating.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- `review.json` dataset (~1.1GB) in `data/` folder

### Setup (2 minutes)
```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ETL (filters to 2008-2012, targets ~80K reviews)
python -m src.data_processing --full-etl --target-reviews 80000

# 4. Create sample DB (5K reviews for TAs)
python -m src.data_processing
```

### Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

**Dashboard Features:**
- 📊 Overview with satisfaction drivers
- 🎯 Competitive benchmarking (radar charts, gap analysis)
- 📈 Performance trends (year-over-year)

---

## 📂 Repository Structure
```
IS5126-G4-hotel-analytics/
├── data/
│   ├── reviews.db              # Full DB (79,853 reviews, 2008-2012)
│   ├── reviews_sample.db       # Sample for TAs (5,000 reviews)
│   └── data_schema.sql         # Schema documentation
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_competitive_benchmarking.ipynb  ⭐ Our main innovation
│   └── 04_performance_profiling.ipynb
├── src/
│   ├── data_processing.py      # ETL with date filtering & sampling
│   ├── benchmarking.py         # K-means clustering + recommendations
│   └── utils.py                # DB connections, paths
├── app/
│   └── streamlit_app.py        # Interactive dashboard
├── profiling/
│   ├── query_results.txt       # Query timing & EXPLAIN QUERY PLAN
│   └── code_profiling.txt      # cProfile output
└── reports/
    └── assignment1_report.pdf  # Technical report (8-10 pages)
```

---

## 📈 Results Summary

### Data Quality
- ✅ **79,853 reviews** (within 50K-80K requirement)
- ✅ **2008-2012** (exactly 5 years)
- ✅ **>99% completeness** for all rating fields
- ✅ **3,374 hotels** (avg 24 reviews/hotel)

### Exploratory Analysis
- **Top satisfaction driver:** Rooms (r=0.80, p<0.001, highly significant)
- **Statistical validation:** All correlations significant (p<0.001)
- **Effect size:** Large (Cohen's d=1.45) for service quality impact

### Competitive Benchmarking

**7 Hotel Segments:**
1. Upscale Beach Resorts (265 hotels, 11%) - 100% beach, high amenities
2. Downtown General Properties (690 hotels, 29%) - Mixed business/leisure
3. Budget/Value Hotels (108 hotels, 5%) - Cost-focused positioning
4. Mid-Tier Urban Hotels (571 hotels, 24%)
5. Mid-Range Business Hotels (374 hotels, 16%)
6. Boutique Downtown Properties (170 hotels, 7%)
7. Suburban Business Hotels (202 hotels, 8%)

**Clustering Quality:**
- Silhouette Score: **0.302** (good for business data)
- Variance Reduction: **35.1%** (meaningful grouping)
- Validation: Statistical + manual business review

**Recommendation Engine:**
- **70%** of hotels have ≥1 recommendation
- **Average:** 1.2 recommendations per hotel
- **High-ROI:** 45% of recommendations >500% ROI
- **Typical payback:** <3 weeks

### Performance
- **Query speed:** All operations <30ms
- **Scalability:** Linear scaling to 80K+ reviews
- **Optimization:** Proper index usage confirmed

---

## 🎓 For Teaching Assistants

### Quick Test (No review.json needed)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Sample DB (5,000 reviews) is included in repository.

---

## 🔬 Methodology

### Data Foundation
- **ETL:** Streaming JSONL parser with temporal filtering
- **Schema:** Normalized (2 tables: authors, reviews)
- **Indexes:** 3 strategic indexes (offering_id, author_id, rating_overall)
- **Filtering:** Latest 5 years with deterministic sampling (seed=42)

### Exploratory Analysis
- Pearson correlation with significance testing (scipy.stats)
- Comparative analysis (t-tests, Cohen's d effect sizes)
- Business-focused insights (not just statistics)

### Competitive Benchmarking

**Feature Engineering:**
- Rating aggregations (mean, std across 6 aspects)
- Text mining: regex-based extraction of hotel characteristics
- Review volume: log-transformed for normalization

**Clustering:**
- Algorithm: K-means
- Features: 6 dimensions (after removing low-variance gym_score)
- Optimization: Tested K=5-12, selected K=7 (best silhouette)
- Validation: Silhouette score + variance reduction + manual review

**Recommendations:**
- Compare hotels to cluster peers (not all hotels)
- Identify gaps >0.3 rating points
- Extract best practices from top 25% performers
- Calculate ROI using industry benchmarks

### Performance Profiling
- Query: EXPLAIN QUERY PLAN + timing (5-run averages)
- Code: cProfile on benchmarking module
- Documentation: Outputs in `profiling/` directory

---

## 👥 Team Contributions

[Add your team member contributions here]

---

**Key Differentiator:** We extract hotel characteristics from text to identify TRUE competitors, not just group by ratings.