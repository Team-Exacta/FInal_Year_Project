"""Central configuration for the UGC Analysis project."""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CLEANED_DATA_DIR = os.path.join(OUTPUT_DIR, "cleaned_data")

# Data sources
REVIEW_COLLECTION_FILE = os.path.join(
    BASE_DIR, "Review_Collection", "Review_Collection", "aggregated_cleaned_reviews.json"
)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")   # raw Apify output lands here

# Scraper settings
PLACES_TO_SCRAPE_FILE = os.path.join(CONFIG_DIR, "places_to_scrape.json")
SCRAPE_PROGRESS_FILE = os.path.join(CONFIG_DIR, "scrape_progress.json")
MAX_REVIEWS_PER_PLACE = 200

# Preprocessing settings
LANGUAGE_CONFIDENCE_THRESHOLD = 0.7
MIN_TEXT_LENGTH_FOR_LANG_DETECT = 15
DUPLICATE_SIMILARITY_THRESHOLD = 0.95

# Review Analysis settings
ANALYSIS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "analysis")
ANALYSIS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Per-category contrastive-score thresholds (selected from the threshold sweep
# in run_eval_threshold_sweep.py — values where each category's F1 peaks).
# Each category has a different optimal cutoff because the score distribution
# is shifted by the per-category negative-anchor subtraction.
ANALYSIS_SCORE_THRESHOLDS = {
    "best_time":   0.34,
    "crowd_level": 0.34,
    "cost_level":  0.30,
}
# Scalar fallback / backward-compat default
ANALYSIS_SCORE_THRESHOLD = 0.34
ANALYSIS_BATCH_SIZE = 32
ANALYSIS_PROGRESS_FILE = os.path.join(ANALYSIS_OUTPUT_DIR, "_progress.json")

# Extraction / Normalization settings
EXTRACTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "extraction")
EXTRACTION_PROGRESS_FILE = os.path.join(EXTRACTION_OUTPUT_DIR, "_progress.json")
SPACY_MODEL = "en_core_web_sm"

# Ensure output directories exist
for dir_path in [CLEANED_DATA_DIR, DATASET_DIR, ANALYSIS_OUTPUT_DIR, EXTRACTION_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
