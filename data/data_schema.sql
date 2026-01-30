-- Hotel reviews schema (SQLite)
-- Assignment: IS5126 Data Foundation

CREATE TABLE authors (
    id TEXT PRIMARY KEY,
    username TEXT,
    location TEXT,
    num_cities INTEGER,
    num_helpful_votes INTEGER,
    num_reviews INTEGER,
    num_type_reviews INTEGER
);

CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    offering_id INTEGER,
    author_id TEXT,
    title TEXT,
    text TEXT,
    date TEXT,
    date_stayed TEXT,
    num_helpful_votes INTEGER,
    via_mobile INTEGER,
    rating_overall REAL,
    rating_service REAL,
    rating_cleanliness REAL,
    rating_value REAL,
    rating_location REAL,
    rating_sleep_quality REAL,
    rating_rooms REAL,
    FOREIGN KEY (author_id) REFERENCES authors(id)
);

CREATE INDEX idx_reviews_offering ON reviews(offering_id);
CREATE INDEX idx_reviews_author ON reviews(author_id);
CREATE INDEX idx_reviews_rating_overall ON reviews(rating_overall);
