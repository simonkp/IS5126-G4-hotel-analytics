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

**Dashboard Features (5 pages):**
- 📊 Overview — KPIs, satisfaction drivers, correlation heatmap
- 🏨 Hotel Explorer — per-hotel radar chart vs cluster peers
- 🎯 Competitive Benchmarking — ML clusters, gap analysis, ROI recommendations
- 📈 Performance Trends — year-over-year, aspect trends, top/bottom tables
- 🔍 Review Insights — review length, helpfulness, reviewer geography

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
- ✅ **Referential integrity confirmed** (0 orphaned author records)

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

### Performance (Quantified Improvements)
- **Query profiling:** Baseline (no indexes) vs with indexes; improvement % per query
- **Key results:** Avg rating by hotel **96.9%** faster, Filter by offering_id **99.8%** faster, Complex aggregation **96.2%** faster (covering index)
- **Indexes:** 4 (offering_id, author_id, rating_overall, covering index for aggregations)
- **Code profiling:** cProfile on benchmarking workflow; outputs in `profiling/`

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
- **Indexes:** 4 (offering_id, author_id, rating_overall, covering index for GROUP BY aggregations)
- **Filtering:** Latest 5 years with deterministic sampling (seed=42)
- **Data Validation:** Great Expectations (GX) with 6-dimension quality framework
  - Completeness, Uniqueness, Validity, Consistency, Timeliness, Accuracy
  - NIH missing data thresholds applied to all rating fields
  - Pipeline gate: analysis only proceeds if all GX checks pass

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
- **Query:** Baseline (no indexes) vs with indexes; 5-run averages; quantified improvement % per query; EXPLAIN QUERY PLAN (with indexes)
- **Code:** cProfile on benchmarking workflow (runctx); top functions by cumulative time
- **Outputs:** `profiling/query_results.txt`, `profiling/code_profiling.txt`

---

## 🏗️ System Architecture & Dashboard

### Architecture Overview

The platform follows a **three-layer architecture** — Data, Analytics, and Presentation — with clear separation of concerns.

**Data Layer.** Raw review data (~1.1 GB JSONL) is ingested through a streaming ETL pipeline (`src/data_processing.py`) that performs temporal filtering (latest 5 years: 2008–2012) and deterministic sampling (seed=42). The output is a normalized SQLite database with two tables (`reviews`, `authors`) and four indexes, including a covering index for GROUP BY aggregation queries. A 5,000-review sample database ships with the repository for reproducibility.

**Analytics Layer.** Three modules consume data from SQLite. The exploratory analysis (Notebook 02) computes Pearson correlations and statistical tests. The benchmarking engine (`src/benchmarking.py`) performs text-based feature extraction, K-Means clustering, and ROI-based recommendation generation. The performance profiler (Notebook 04) benchmarks query execution with and without indexes.

**Presentation Layer.** A Streamlit dashboard (`app/streamlit_app.py`) serves as the user-facing interface, querying the database via SQLAlchemy and consuming clustering outputs from the benchmarking engine. All expensive computations are cached using `@st.cache_data` to ensure responsive interaction after first load.

### Dashboard Features & Business Rationale

| # | Feature | Business Question | Key Features | How It Helps |
|---|---------|-------------------|--------------|--------------|
| 1 | **📊 Overview** | *"What's our review landscape?"* | KPI cards, satisfaction driver rankings, aspect correlation heatmap | Identifies which aspects drive overall satisfaction (Rooms: r=0.80) |
| 2 | **🏨 Hotel Explorer** | *"How is MY hotel doing?"* | Radar chart vs cluster peers, aspect breakdown, yearly trend | Shows gaps versus **similar** hotels, not all hotels |
| 3 | **🎯 Benchmarking** | *"Who are my competitors? Where to invest?"* | ML cluster profiles, scatter plot, gap analysis with ROI estimates | Groups hotels by actual similarity; provides actionable recs with payback |
| 4 | **📈 Trends** | *"Getting better or worse?"* | Dual-axis volume/rating chart, aspect trends, top/bottom tables | Reveals temporal patterns for proactive intervention |
| 5 | **🔍 Insights** | *"What can reviews tell us?"* | Review length, helpfulness, mobile/desktop, reviewer geography | Informs marketing and review response strategy |

### Technology Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Database** | SQLite (79K reviews, 4 indexes) |
| **ORM** | SQLAlchemy 2.0 |
| **ML** | scikit-learn (K-Means, StandardScaler, silhouette_score) |
| **Visualisation** | Plotly (interactive), matplotlib/seaborn (notebooks) |
| **Dashboard** | Streamlit (5 pages, cached queries) |
| **Data Quality** | Great Expectations (6-dimension validation) |
| **Version Control** | Git / GitHub |


---

## 👥 Team Contributions

All team members contributed equally to the project's success through collaborative effort and division of responsibilities.

### Joint Contributions

- **Manjunath Warad & Yadagiri Spurthi**  
  Worked together on data preprocessing, ETL pipeline implementation, database schema design, indexing, and data quality validation.

- **Rayaan Nabi Ahmed Quraishi & Aryan Jain**  
  Collaborated on feature engineering, text-based characteristic extraction, K-means clustering implementation, model validation (silhouette and variance reduction), and development of the recommendation logic.

- **Simon Kalayil Philip & the Team**  
  Led exploratory data analysis, statistical testing, correlation analysis, visualization design, and contributed to dashboard feature development and business insight interpretation.

---

**Key Differentiator:** We extract hotel characteristics from text to identify TRUE competitors, not just group by ratings.