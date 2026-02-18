# IS5126-G4-Assignment1 Report

# 1. Data Foundation Summary

### 1.1 Dataset construction and cleanup

We constructed a curated SQLite database to support efficient analytics and reproducible querying at meaningful scale. Review dates in the raw dataset were stored as natural-language strings (e.g., “July 9, 2012”), which are not reliably filterable using SQLite’s native date functions. We therefore normalized dates into an ISO-formatted field date\_iso (YYYY-MM-DD) to enable correct time-window filtering, grouping, and indexing.

### 1.2 Sampling strategy (rubric-scale, representative, reproducible)

To meet the rubric scale requirement while keeping analysis tractable, we sampled 80,000 reviews from the most recent 5-year window available in the dataset (2008–2012). The underlying dataset is temporally imbalanced (later years contain substantially more reviews), and a recency-only sampling approach can amplify this skew. To mitigate bias and preserve time representativeness, we applied year-stratified sampling across 2008–2012. We additionally enforced a per-hotel-per-year cap (30 reviews) to preserve hotel diversity and prevent dominance by high-volume properties. Sampling was deterministic to ensure full reproducibility.

### 1.3 Data model / schema overview

The curated SQLite database follows a simple relational design with two core entities: reviews and authors. The reviews table stores one row per review and includes foreign key–like field author_id (nullable) that links to authors.id. Reviews are also associated with hotels via offering_id (hotel identifier), enabling hotel-level benchmarking and aggregation. Key measures include rating_overall and subratings (service, cleanliness, value, location, rooms, sleep quality), along with metadata such as via_mobile, num_helpful_votes, and normalized date_iso for time-based analysis.

### 1.4 Indexing strategy

To support interactive analytics and repeated aggregation queries, we created indexes on the columns most frequently used for filtering and grouping. idx_reviews_date_iso accelerates time-window queries (e.g., year/month trends) by avoiding full-table scans when filtering by date. idx_reviews_offering_id and idx_reviews_author_id speed up hotel- and author-level aggregations and joins (e.g., computing hotel KPIs or linking reviews to authors), while idx_reviews_rating_overall supports fast segmentation and benchmarking queries that filter/stratify by rating bands (e.g., low ≤2 vs high ≥4).

### 1.5 Final curated dataset profile

* **Reviews:** 80,000
* **Hotels:** 3,422
* **Authors (linked):** 68,241
* **Reviews with missing/blank author_id:** 3,411 (4.26%)
* **Date range (date_iso):** 2008-11-01 to 2012-12-20

**Year distribution**
* 2008: 7,458
* 2009: 9,603
* 2010: 12,993
* 2011: 20,638
* 2012: 29,308

### 1.6 Data quality checks

**Rating bounds and vote sanity checks passed:**

* All rating fields fall within **\[1, 5\]**

* No negative helpful votes were found

**Text completeness is high:**

* Missing title: 0

* Missing text: 0

* Missing rating\_overall: 0

### 1.7 Missingness (subratings)

Several subratings contain missing values, with `rating_sleep_quality` notably sparse. Missingness in the curated dataset is:

* `rating_service`: 6,953 (**8.69%**)
* `rating_cleanliness`: 6,798 (**8.50%**)
* `rating_value`: 7,138 (**8.92%**)
* `rating_location`: 6,898 (**8.62%**)
* `rating_rooms`: 7,123 (**8.90%**)
* `rating_sleep_quality`: 26,924 (**33.66%**)

For analysis, we use **pairwise deletion** (available-case analysis per metric) and explicitly report denominators (N). We avoid imputing rating values unless explicitly justified.


### 1.8 Hotel-level distribution

Reviews per hotel exhibit a long-tail distribution:

* **min \= 1**, **max \= 150**, **avg \= 23.38** reviews per hotel

This long-tail supports the use of a per-hotel-per-year sampling cap to maintain diversity and reduce over-representation of high-volume hotels.


# 2. Exploratory Data Analysis (EDA)

### 2.1 Overall rating distribution (sentiment shape)

Overall ratings are strongly positive and skewed toward high scores. Across 80,000 reviews, **74.62%** of reviews are **4–5★** (4★: 33.26%, 5★: 41.36%). Low ratings are comparatively rare: **11.11%** of reviews are **1–2★** (1★: 4.99%, 2★: 6.12%), with the remaining **14.27%** at 3★.

**Implication:** downstream analyses (e.g., classification of dissatisfaction) should account for class imbalance (few low-rating reviews relative to high-rating reviews).

### 2.2 Mobile vs non-mobile reviews

Mobile-submitted reviews have a slightly lower mean overall rating (**3.977**) compared to non-mobile reviews (**4.000**), but the gap is small (**Δ = 0.023**). Given the large sample size, this difference appears **practically negligible**.

### 2.3 Subrating drivers of overall satisfaction (gap analysis)

To understand what differentiates satisfied vs dissatisfied stays, we compared mean subratings across overall rating bands:

* **Low (≤2★):** service 2.041, cleanliness 2.394, value 1.854, location 3.408, sleep 2.109, rooms 1.978  
* **Mid (3★):** service 3.316, cleanliness 3.557, value 3.140, location 4.010, sleep 3.274, rooms 3.042  
* **High (≥4★):** service 4.592, cleanliness 4.653, value 4.420, location 4.651, sleep 4.524, rooms 4.436  

The largest separations between low and high ratings occur for **value**, **service**, **rooms**, and **sleep quality**, while **location** remains relatively high even among low overall ratings. This suggests dissatisfaction is driven more by experience/quality factors than by geography.

### 2.4 Time trend (2008–2012)

Average overall ratings increased from **3.790 (2008)** to **4.053 (2011)**, followed by a slight dip to **4.010 (2012)**. This pattern may reflect changes in the hotel mix reviewed over time, reviewer behavior, or service quality trends; subsequent analyses should control for hotel composition when attributing time effects.

| Year | Reviews (n) | Avg overall rating |
|---|---:|---:|
| 2008 | 7,458 | 3.790 |
| 2009 | 9,603 | 3.983 |
| 2010 | 12,993 | 4.019 |
| 2011 | 20,638 | 4.053 |
| 2012 | 29,308 | 4.010 |

### 2.5 Hotel-level dispersion and top/bottom segments

Hotel performance varies significantly even after enforcing a minimum sample size (≥20 reviews). Among the highest-rated hotels, the top segment achieves near-perfect averages (e.g., `offering_id=1149434` has **avg_overall = 5.000** across **33** reviews). In contrast, the lowest-rated segment includes hotels with very low averages (e.g., `offering_id=224229` has **avg_overall = 1.958** across **24** reviews; `offering_id=93356` has **avg_overall = 2.000** across **39** reviews).

**Implication:** benchmarking is meaningful in this dataset: there is a clear separation between top and bottom properties, enabling peer comparison and targeted improvement strategies. Notably, the bottom list includes a high-volume hotel (`offering_id=93421`, **n=130**) with **avg_overall=2.592**, suggesting persistent underperformance rather than sampling noise.

### 2.6 Hotel experience consistency (variance / volatility)

Beyond averages, hotels differ in the **consistency** of guest experiences. Using per-hotel variance of `rating_overall` (computed as `E[x²] − (E[x])²`) for hotels with ≥20 reviews, several hotels show very high variability (e.g., `offering_id=119735` has **avg_overall=3.174** with **var_overall=2.6654**). High variance indicates polarized guest experiences and may reflect inconsistent service delivery, uneven room quality, or variable operational execution.

**Implication:** benchmarking should consider both *level* (mean rating) and *stability* (variance). Hotels with moderate averages but high variance represent “inconsistency risk” and may benefit from standardization interventions.

### 2.7 “Pain point profiles” of low-performing hotels (subrating patterns)

Low-performing hotels exhibit distinct subrating patterns that reveal likely drivers of dissatisfaction. For example, `offering_id=224229` (avg_overall **1.958**) has extremely low **service (1.778)**, **cleanliness (1.833)**, **rooms (1.611)**, and **sleep quality (1.556)**, while **location remains relatively higher (3.444)**. Similarly, `offering_id=93356` (avg_overall **2.000**) shows weak **rooms (1.667)** despite strong **location (4.061)**.

**Implication:** poor performance is concentrated in controllable operational dimensions (rooms, cleanliness, service, sleep quality, perceived value) rather than geography. This supports dimension-specific recommendations in the competitive benchmarking stage.