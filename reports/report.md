# IS5126-G4-Assignment1 Report

# Data Foundation Summary

### 1.1 Dataset construction and cleanup

We constructed a curated SQLite database to support efficient analytics and reproducible querying at meaningful scale. Review dates in the raw dataset were stored as natural-language strings (e.g., “July 9, 2012”), which are not reliably filterable using SQLite’s native date functions. We therefore normalized dates into an ISO-formatted field date\_iso (YYYY-MM-DD) to enable correct time-window filtering, grouping, and indexing.

### 1.2 Sampling strategy (rubric-scale, representative, reproducible)

To meet the rubric scale requirement while keeping analysis tractable, we sampled 80,000 reviews from the most recent 5-year window available in the dataset (2008–2012). The underlying dataset is temporally imbalanced (later years contain substantially more reviews), and a recency-only sampling approach can amplify this skew. To mitigate bias and preserve time representativeness, we applied year-stratified sampling across 2008–2012. We additionally enforced a per-hotel-per-year cap (30 reviews) to preserve hotel diversity and prevent dominance by high-volume properties. Sampling was deterministic to ensure full reproducibility.

### 1.3 Final curated dataset profile

* **Reviews:** 80,000

* **Hotels:** 3,422

* **Authors (linked):** 68,300

* **Reviews with missing/blank author\_id:** 3,366 (4.21%)

* **Date range (date\_iso):** 2008-11-01 to 2012-12-20

**Year distribution**

* 2008: 7,458

* 2009: 9,603

* 2010: 12,993

* 2011: 20,638

* 2012: 29,308

### 1.4 Data quality checks

**Rating bounds and vote sanity checks passed:**

* All rating fields fall within **\[1, 5\]**

* No negative helpful votes were found

**Text completeness is high:**

* Missing title: 0

* Missing text: 0

* Missing rating\_overall: 0

### 1.5 Missingness (subratings)

Several subratings contain missing values, with rating\_sleep\_quality notably sparse. Missingness in the curated dataset is:

* rating\_service: 6,915 (**8.64%**)

* rating\_cleanliness: 6,762 (**8.45%**)

* rating\_value: 7,105 (**8.88%**)

* rating\_location: 6,866 (**8.58%**)

* rating\_rooms: 7,093 (**8.87%**)

* rating\_sleep\_quality: 26,921 (**33.65%**)

For analysis, we use **pairwise deletion** (available-case analysis per metric) and explicitly report denominators (N). We avoid imputing rating values unless explicitly justified.

### 1.6 Hotel-level distribution

Reviews per hotel exhibit a long-tail distribution:

* **min \= 1**, **max \= 150**, **avg \= 23.38** reviews per hotel

This long-tail supports the use of a per-hotel-per-year sampling cap to maintain diversity and reduce over-representation of high-volume hotels.

