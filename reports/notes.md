### EDA evidence (data/reviews.db, 80k curated)

### Overall rating distribution (counts + %)

**Command**
```bash
sqlite3 -header -column data/reviews.db "
SELECT rating_overall,
       COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM reviews), 2) AS pct
FROM reviews
GROUP BY rating_overall
ORDER BY rating_overall;"

rating_overall  n      pct
-------------  -----  -----
1.0            3991   4.99
2.0            4898   6.12
3.0            11414  14.27
4.0            26611  33.26
5.0            33086  41.36

# Mobile vs non-mobile (mean rating)
sqlite3 -header -column data/reviews.db "
SELECT via_mobile,
       COUNT(*) AS n,
       ROUND(AVG(rating_overall), 3) AS avg_overall
FROM reviews
GROUP BY via_mobile
ORDER BY via_mobile;"

via_mobile  n      avg_overall
----------  -----  ----------
0           74812  4.000
1           5188   3.977

# Driver gaps: subratings by overall band (low/mid/high)
sqlite3 -header -column data/reviews.db "
SELECT
  CASE
    WHEN rating_overall <= 2 THEN 'low(<=2)'
    WHEN rating_overall = 3 THEN 'mid(3)'
    ELSE 'high(>=4)'
  END AS band,
  COUNT(*) AS n,
  ROUND(AVG(rating_service), 3) AS svc,
  ROUND(AVG(rating_cleanliness), 3) AS clean,
  ROUND(AVG(rating_value), 3) AS val,
  ROUND(AVG(rating_location), 3) AS loc,
  ROUND(AVG(rating_sleep_quality), 3) AS sleep,
  ROUND(AVG(rating_rooms), 3) AS rooms
FROM reviews
GROUP BY band
ORDER BY band;"

band       n      svc    clean  val    loc    sleep  rooms
---------  -----  -----  -----  -----  -----  -----  -----
high(>=4)  59697  4.592  4.653  4.420  4.651  4.524  4.436
low(<=2)   8889   2.041  2.394  1.854  3.408  2.109  1.978
mid(3)     11414  3.316  3.557  3.140  4.010  3.274  3.042

# Time trend (avg rating by year)
sqlite3 -header -column data/reviews.db "
SELECT substr(date_iso,1,4) AS year,
       COUNT(*) AS n,
       ROUND(AVG(rating_overall), 3) AS avg_overall
FROM reviews
GROUP BY year
ORDER BY year;"

year  n      avg_overall
----  -----  ----------
2008  7458   3.790
2009  9603   3.983
2010  12993  4.019
2011  20638  4.053
2012  29308  4.010


## Data quality checks
# Missingness in subratings (counts + %)
sqlite3 data/reviews.db "
SELECT
  COUNT(*) AS total,
  SUM(CASE WHEN rating_service IS NULL THEN 1 ELSE 0 END) AS miss_service,
  ROUND(100.0 * SUM(CASE WHEN rating_service IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_service,
  SUM(CASE WHEN rating_cleanliness IS NULL THEN 1 ELSE 0 END) AS miss_cleanliness,
  ROUND(100.0 * SUM(CASE WHEN rating_cleanliness IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_cleanliness,
  SUM(CASE WHEN rating_value IS NULL THEN 1 ELSE 0 END) AS miss_value,
  ROUND(100.0 * SUM(CASE WHEN rating_value IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_value,
  SUM(CASE WHEN rating_location IS NULL THEN 1 ELSE 0 END) AS miss_location,
  ROUND(100.0 * SUM(CASE WHEN rating_location IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_location,
  SUM(CASE WHEN rating_rooms IS NULL THEN 1 ELSE 0 END) AS miss_rooms,
  ROUND(100.0 * SUM(CASE WHEN rating_rooms IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_rooms,
  SUM(CASE WHEN rating_sleep_quality IS NULL THEN 1 ELSE 0 END) AS miss_sleep_quality,
  ROUND(100.0 * SUM(CASE WHEN rating_sleep_quality IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_sleep_quality
FROM reviews;"

80000|6953|8.69|6798|8.5|7138|8.92|6898|8.62|7123|8.9|26924|33.66

# Missing/blank author_id
sqlite3 -header -column data/reviews.db "
SELECT
  COUNT(*) AS total_reviews,
  SUM(CASE WHEN author_id IS NULL OR trim(author_id)='' THEN 1 ELSE 0 END) AS missing_author_id,
  ROUND(100.0 * SUM(CASE WHEN author_id IS NULL OR trim(author_id)='' THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_missing
FROM reviews;"

total_reviews  missing_author_id  pct_missing
-------------  -----------------  -----------
80000          3411               4.26    

# Top 10 hotels by avg rating (min 20 reviews)
sqlite3 -header -column data/reviews.db "
SELECT offering_id,
       COUNT(*) AS n_reviews,
       ROUND(AVG(rating_overall), 3) AS avg_overall
FROM reviews
GROUP BY offering_id
HAVING COUNT(*) >= 20
ORDER BY avg_overall DESC, n_reviews DESC
LIMIT 10;"

offering_id  n_reviews  avg_overall
-----------  ---------  -----------
1149434      33         5.0
781627       38         4.947
80092        29         4.931
111408       27         4.926
114938       55         4.891
578190       34         4.882
2322597      22         4.864
225105       39         4.846
89575        57         4.842
564652       25         4.84

# Bottom 10 hotels by avg rating (min 20 reviews)
sqlite3 -header -column data/reviews.db "
SELECT offering_id,
       COUNT(*) AS n_reviews,
       ROUND(AVG(rating_overall), 3) AS avg_overall
FROM reviews
GROUP BY offering_id
HAVING COUNT(*) >= 20
ORDER BY avg_overall ASC, n_reviews DESC
LIMIT 10;"

offering_id  n_reviews  avg_overall
-----------  ---------  -----------
224229       24         1.958
93356        39         2.0
73889        20         2.3
1155229      25         2.44
122007       49         2.571
93421        130        2.592
1223830      53         2.623
80799        27         2.667
119650       24         2.667
217148       28         2.679

# Highest-variance hotels (rating volatility) (min 20 reviews)
sqlite3 -header -column data/reviews.db "
WITH hotel AS (
  SELECT offering_id,
         COUNT(*) AS n_reviews,
         AVG(rating_overall) AS mean_overall,
         AVG(rating_overall * rating_overall) AS mean_sq
  FROM reviews
  GROUP BY offering_id
  HAVING COUNT(*) >= 20
)
SELECT offering_id,
       n_reviews,
       ROUND(mean_overall, 3) AS avg_overall,
       ROUND(mean_sq - mean_overall*mean_overall, 4) AS var_overall
FROM hotel
ORDER BY var_overall DESC
LIMIT 10;"

offering_id  n_reviews  avg_overall  var_overall
-----------  ---------  -----------  -----------
119735       23         3.174        2.6654
219622       25         3.36         2.4704
119388       21         3.714        2.3946
95253        21         3.667        2.3175
80950        28         2.893        2.3099
102519       37         3.378        2.2352
77753        27         2.741        2.118
87627        35         3.114        2.1012
270508       20         3.25         2.0875
81212        42         3.357        2.0867


# Subrating profile for worst hotels (min 20 reviews)
sqlite3 -header -column data/reviews.db "
WITH hotel AS (
  SELECT offering_id,
         COUNT(*) AS n_reviews,
         AVG(rating_overall) AS avg_overall
  FROM reviews
  GROUP BY offering_id
  HAVING COUNT(*) >= 20
),
worst AS (
  SELECT offering_id
  FROM hotel
  ORDER BY avg_overall ASC
  LIMIT 10
)
SELECT r.offering_id,
       COUNT(*) AS n_reviews,
       ROUND(AVG(r.rating_overall), 3) AS avg_overall,
       ROUND(AVG(r.rating_service), 3) AS svc,
       ROUND(AVG(r.rating_cleanliness), 3) AS clean,
       ROUND(AVG(r.rating_value), 3) AS val,
       ROUND(AVG(r.rating_location), 3) AS loc,
       ROUND(AVG(r.rating_rooms), 3) AS rooms,
       ROUND(AVG(r.rating_sleep_quality), 3) AS sleep
FROM reviews r
JOIN worst w ON r.offering_id = w.offering_id
GROUP BY r.offering_id
ORDER BY avg_overall ASC;"

offering_id  n_reviews  avg_overall  svc    clean  val    loc    rooms  sleep
-----------  ---------  -----------  -----  -----  -----  -----  -----  -----
224229       24         1.958        1.778  1.833  2.611  3.444  1.611  1.556
93356        39         2.0          2.697  2.03   2.182  4.061  1.667  2.0
73889        20         2.3          2.75   2.842  2.75   2.789  2.579  2.684
1155229      25         2.44         2.818  2.591  2.909  3.955  2.273  2.875
122007       49         2.571        2.976  2.595  2.81   3.905  2.405  2.552
93421        130        2.592        2.581  2.093  3.324  4.63   2.167  3.129
1223830      53         2.623        2.958  3.0    3.122  3.612  2.894  2.679
80799        27         2.667        3.182  3.045  2.955  3.0    2.696  2.5
119650       24         2.667        2.476  3.143  2.8    3.476  2.81   2.667
217148       28         2.679        3.042  2.958  3.2    2.458  3.0    3.0


