# UGC-Based Travel Recommendation System — Project Documentation

**Domain:** Sri Lanka Tourism  
**Goal:** Extract structured travel advice (best time to visit, crowd level, cost) from user-generated reviews and produce per-place recommendation profiles.  
**Scope:** Steps 1–6 complete. 180 places. 48,364 reviews.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pipeline Summary](#pipeline-summary)
3. [Step 1 — Data Collection](#step-1--data-collection)
4. [Step 2 — Preprocessing](#step-2--preprocessing)
5. [Step 3 — Suggestion Detection & Classification](#step-3--suggestion-detection--classification)
6. [Step 4 — Span Extraction](#step-4--span-extraction)
7. [Step 5 — Normalization](#step-5--normalization)
8. [Step 6 — POI Aggregation](#step-6--poi-aggregation)
9. [Evaluation Framework](#evaluation-framework)
10. [Output Files](#output-files)
11. [File Structure](#file-structure)
12. [How to Run](#how-to-run)
13. [Key Statistics](#key-statistics)

---

## System Overview

This system processes raw travel reviews from TripAdvisor and a Kaggle corpus to produce structured, machine-readable recommendation profiles for 180 Points of Interest (POIs) in Sri Lanka. Each profile answers three practical travel questions extracted from real visitor reviews:

- **When is the best time to visit?** (time of day, season, months)
- **How crowded is it?** (level 1–5 scale, peak periods)
- **How much does it cost?** (cost level, amount in LKR, fee type)

The pipeline is fully automated, resume-safe, and runs end-to-end from raw review data to structured JSON profiles.

---

## Pipeline Summary

```
Raw Reviews (TripAdvisor + Kaggle)
         │
         ▼
  Step 1: Data Collection         → 48,364 raw reviews, 180 POIs
         │
         ▼
  Step 2: Preprocessing           → cleaned text, language filtered, deduplicated
         │
         ▼
  Step 3: Suggestion Detection    → tagged reviews (best_time / crowd_level / cost_level)
         │                          using Sentence-BERT contrastive scoring
         ▼
  Step 4: Span Extraction         → key phrases extracted from tagged sentences
         │                          using SpaCy NER + Matcher + dependency parsing
         ▼
  Step 5: Normalization           → free-text phrases → structured values
         │                          (EARLY_MORNING, PACKED, HIGH, etc.)
         ▼
  Step 6: POI Aggregation         → one clean JSON profile per place
                                    with confidence scores
```

---

## Step 1 — Data Collection

**Status:** Complete  
**Code:** `scraper/apify_collector.py`, `run_scraper.py`

### Sources

| Source | Description |
|---|---|
| TripAdvisor (via Apify) | Cloud scraper — up to 200 reviews per POI |
| Kaggle Review_Collection | Pre-collected corpus of Sri Lanka tourism reviews |

### What was done

- Defined 180 Sri Lanka POIs across all provinces (temples, national parks, beaches, waterfalls, historical sites, etc.)
- Scraped reviews from TripAdvisor using the Apify cloud API
- Merged with the Kaggle corpus
- Each raw review contains: place name, review text, rating, travel date, published date, reviewer location

### Output

- `dataset/` — raw Apify scrape output (per place JSON)
- `Review_Collection/` — Kaggle corpus
- **48,364 total raw reviews** across 180 places

---

## Step 2 — Preprocessing

**Status:** Complete  
**Code:** `preprocessing/text_cleaner.py`, `preprocessing/language_filter.py`, `preprocessing/deduplicator.py`, `preprocessing/pipeline.py`, `run_preprocessing.py`

### What was done

**Text Cleaning** (`text_cleaner.py`)
- Removed HTML tags, URLs, emoji characters
- Normalized whitespace and punctuation
- Preserved original text alongside cleaned version (`text_clean`, `title_clean`)

**Language Filtering** (`language_filter.py`)
- Used `langdetect` library to detect language of each review
- Confidence threshold: 0.7
- Minimum text length for detection: 15 characters
- Kept only English reviews

**Deduplication** (`deduplicator.py`)
- Used TF-IDF vectorization + cosine similarity
- Similarity threshold: 0.95
- Removed near-duplicate reviews (copy-pasted or repeated submissions)

### Output

- `output/cleaned_data/{place_key}.json` — cleaned review list per place
- `output/cleaned_data/_all_reviews.csv` — flat CSV of all 48,364 cleaned reviews
- **180 place files** ready for analysis

---

## Step 3 — Suggestion Detection & Classification

**Status:** Complete  
**Code:** `analysis/detector.py`, `analysis/pipeline.py`, `run_analysis.py`

### Approach: Sentence-BERT Contrastive Scoring

This is the core ML classification step. The goal is to identify which reviews contain actionable travel advice in the three categories. This uses a zero-shot classification approach — no labeled training data required.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (22 MB, runs on CPU)

#### How it works

1. Each review is split into sentences
2. Every sentence is encoded into a 384-dimensional embedding vector
3. For each category, a set of **anchor sentences** (positive examples) and **negative anchors** (generic travel praise) are pre-encoded
4. A **contrastive score** is computed per sentence:

```
net_score = max_similarity(sentence, positive_anchors)
          - 0.5 × max_similarity(sentence, negative_anchors)
```

5. If `net_score >= 0.38`, the sentence is tagged for that category

#### Why contrastive scoring?

Simple cosine similarity against positive anchors gave too many false positives — generic sentences like "This is a beautiful place worth visiting" scored high for best_time because they share vocabulary. The negative anchors penalize generic travel praise and isolate category-specific content.

#### Anchor sentences (examples)

| Category | Positive anchor example | Negative anchor example |
|---|---|---|
| best_time | "The best time to visit is early morning before the crowds arrive" | "This is a beautiful and amazing place" |
| crowd_level | "It was extremely crowded with tourists during peak hours" | "Highly recommend visiting this wonderful attraction" |
| cost_level | "The entrance fee is Rs. 1500 for foreigners" | "The scenery is absolutely stunning and breathtaking" |

#### Threshold selection

| Threshold | best_time | crowd_level | cost_level | Notes |
|---|---|---|---|---|
| 0.45 | 320 | 267 | 1,104 | Too strict — misses valid reviews |
| 0.40 | ~700 | ~600 | ~1,800 | Moderate |
| **0.38** | **1,228** | **1,221** | **2,144** | Selected — good balance |
| 0.35 | ~2,500 | ~2,400 | ~3,800 | Too loose — too many false positives |

### Output per review

```json
{
  "analysis_tags": {
    "best_time": true,
    "crowd_level": false,
    "cost_level": false
  },
  "analysis_scores": {
    "best_time": 0.3863,
    "crowd_level": 0.2397,
    "cost_level": 0.1176
  },
  "analysis_sentences": {
    "best_time": ["We left early so that we could see the sunrise on the summit."],
    "crowd_level": [],
    "cost_level": []
  }
}
```

### Results

| Category | Tagged Reviews | Rate |
|---|---|---|
| best_time | 1,228 | 2.54% |
| crowd_level | 1,221 | 2.52% |
| cost_level | 2,144 | 4.43% |
| **Total** | **4,593** | **9.49%** |

### Output files

- `output/analysis/{place_key}.json` — reviews enriched with tags + sentences
- `output/analysis/_best_time.csv` — all tagged best_time sentences
- `output/analysis/_crowd_level.csv` — all tagged crowd_level sentences
- `output/analysis/_cost_level.csv` — all tagged cost_level sentences
- `output/analysis/_all_tagged.csv` — combined flat CSV
- `output/analysis/_stats.json` — run statistics

### CLI

```bash
python run_analysis.py                        # process all places (resume-safe)
python run_analysis.py --no-resume            # restart from scratch
python run_analysis.py --place Sigiriya_Lion_Rock
python run_analysis.py --threshold 0.38
```

---

## Step 4 — Span Extraction

**Status:** Complete  
**Code:** `extraction/extractor.py`

### Approach: SpaCy Rule-Based Extraction

Each tagged sentence is processed by SpaCy (`en_core_web_sm`) to extract the specific actionable phrase — not the whole sentence, just the key part.

**Model:** SpaCy `en_core_web_sm` (12 MB, ~10,000 sentences/sec on CPU)

### Extraction methods per category

#### best_time extraction
1. **PhraseMatcher** — matches known time phrases: "early morning", "dry season", "monsoon season", "weekends", "public holidays", etc.
2. **NER (TIME entities)** — SpaCy detects time expressions: "morning", "August", "9am"
3. **Dependency parsing** — finds the ROOT verb and its subtree for imperative sentences: "Go early in the morning"
4. **Avoid-object extraction** — finds what is being avoided: "avoid weekends", "skip peak hours"

#### crowd_level extraction
1. **PhraseMatcher** — matches crowd descriptors: "very crowded", "packed with tourists", "not too crowded", "had the place to ourselves"
2. **Adjective + modifier** — captures intensity: "extremely crowded", "not busy", "very quiet"
3. **Temporal context** — when does the crowd occur: "on weekends", "in the morning"

#### cost_level extraction
1. **NER (MONEY entities)** — SpaCy detects monetary amounts
2. **Custom MONEY_LKR Matcher** — captures LKR-specific patterns: "Rs 200", "1500 rupees", "LKR 5000"
3. **PhraseMatcher** — fee vocabulary: "entry fee", "entrance fee", "free admission", "no charge"
4. **Evaluation adjectives** — "cheap", "expensive", "reasonable", "worth it"

### Output per review (added fields)

```json
{
  "extracted_spans": {
    "best_time": {
      "time_spans": ["early morning", "dry season"],
      "action": "go",
      "avoid": ["weekends"]
    },
    "crowd_level": {
      "crowd_spans": ["very crowded", "packed"],
      "crowd_when": ["weekends"]
    },
    "cost_level": {
      "price_spans": ["Rs 1500", "entry fee"],
      "evaluation": "expensive",
      "fee_type": "ENTRY_FEE"
    }
  }
}
```

---

## Step 5 — Normalization

**Status:** Complete  
**Code:** `extraction/normalizer.py`, `extraction/pipeline.py`, `run_extraction.py`

### Approach: Regex-based mapping to fixed schema

Extracted free-text spans are mapped to standardized categorical values using regex pattern matching. This makes values from different reviews directly comparable.

### Normalization schemas

#### TIME_OF_DAY (5 levels)

| Label | Trigger patterns | Example phrases |
|---|---|---|
| EARLY_MORNING | "early morning", "sunrise", "dawn", "before 9", "6/7/8am", "go early" | "go early", "before sunrise", "before 9am" |
| MID_MORNING | "morning", "9/10/11am" | "in the morning", "around 10am" |
| AFTERNOON | "afternoon", "midday", "noon", "12–4pm" | "in the afternoon", "at noon" |
| EVENING | "evening", "late afternoon", "sunset", "5–7pm" | "at sunset", "in the evening" |
| NIGHT | "night", "after dark", "8pm+" | "at night", "after dark" |

#### SEASON (3 levels)

| Label | Trigger patterns |
|---|---|
| DRY_SEASON | "dry season", "dry months", December, January, February, March, April |
| MONSOON | "monsoon", "rainy season", May, June, July, August, September, October |
| SHOULDER | November, "shoulder season" |

#### DAY_TYPE

| Label | Trigger patterns |
|---|---|
| WEEKEND | "weekend", "weekends" |
| WEEKDAY | "weekday", "weekdays" |
| PUBLIC_HOLIDAY | "public holiday", "poya day", "national holiday" |

#### CROWD_LEVEL (1–5 scale)

| Level | Label | Trigger patterns |
|---|---|---|
| 5 | PACKED | "very crowded", "extremely crowded", "packed", "overrun", "impossible to enjoy" |
| 4 | BUSY | "crowded", "busy", "full of tourists", "long queue", "many people" |
| 3 | MODERATE | "moderate", "manageable", "not too crowded", "some tourists" |
| 2 | QUIET | "quiet", "calm", "not crowded", "not busy", "few visitors" |
| 1 | EMPTY | "no one", "deserted", "had it to ourselves", "no other people" |

> **Negation handling:** "not crowded" triggers BUSY (level 4) first, then negation detection bumps it down 2 levels to QUIET (level 2).

#### COST_LEVEL (5 levels)

| Label | Trigger | LKR range |
|---|---|---|
| FREE | "free", "no entry fee", "no charge" | 0 |
| LOW | "cheap", "inexpensive", amount < 500 | < 500 LKR |
| MODERATE | "reasonable", "affordable", "worth it" | 500–2,000 LKR |
| HIGH | "expensive", "pricey" | 2,000–5,000 LKR |
| VERY_HIGH | "very expensive", "overpriced", amount > 5000 | > 5,000 LKR |

> LKR amount extraction: `(?:rs\.?\s*|lkr\s*|rupees?\s*)(\d[\d,]*)` — supports "Rs 200", "Rs. 1,500", "1500 rupees", "LKR 5000"

### Output per review (added fields)

```json
{
  "normalized": {
    "best_time": {
      "time_of_day": "EARLY_MORNING",
      "season": "DRY_SEASON",
      "months": ["DECEMBER", "JANUARY"],
      "recommend_day": null,
      "avoid": ["WEEKEND"]
    },
    "crowd_level": {
      "crowd_level": 5,
      "crowd_label": "PACKED",
      "crowd_when": ["WEEKEND"]
    },
    "cost_level": {
      "cost_level": "HIGH",
      "amount_lkr": 5000,
      "evaluation": "expensive",
      "fee_type": "ENTRY_FEE"
    }
  }
}
```

### Output files

- `output/extraction/{place_key}.json` — all reviews enriched with extracted_spans + normalized
- `output/extraction/_all_spans.csv` — flat CSV of 4,593 extracted span rows
- `output/extraction/_progress.json` — resume checkpoint

### CLI

```bash
python run_extraction.py                        # all places (resume-safe)
python run_extraction.py --no-resume            # restart
python run_extraction.py --place Sigiriya_Lion_Rock
```

---

## Step 6 — POI Aggregation

**Status:** Complete  
**Code:** `aggregation/aggregator.py`, `run_aggregation.py`

### Approach: Frequency-based voting with confidence scoring

All normalized values across reviews for a place are aggregated into one profile using frequency counting (majority voting). A confidence score measures how much the reviews agree on the dominant value.

```
confidence = count_of_dominant_value / total_reviews_that_produced_a_value
```

### Profile structure

```json
{
  "place_name": "Sigiriya Lion Rock",
  "total_reviews": 538,
  "best_time": {
    "time_of_day": "EARLY_MORNING",
    "season": "DRY_SEASON",
    "months": ["MARCH", "APRIL", "DECEMBER", "JULY"],
    "avoid": [],
    "confidence": 0.74,
    "based_on": 45
  },
  "crowd": {
    "level": 5,
    "label": "PACKED",
    "avg_level": 4.7,
    "busiest_period": ["WEEKEND"],
    "confidence": 0.86,
    "based_on": 20
  },
  "cost": {
    "level": "HIGH",
    "median_lkr": 100,
    "fee_type": "ENTRY_FEE",
    "confidence": 0.60,
    "based_on": 31
  }
}
```

### Field descriptions

| Field | Description |
|---|---|
| `place_name` | Human-readable name (cleaned from file key) |
| `total_reviews` | Total reviews collected for this place |
| `best_time.time_of_day` | Dominant time of day label (most common across tagged reviews) |
| `best_time.season` | Dominant season |
| `best_time.months` | Months mentioned, ranked by frequency (up to 6) |
| `best_time.avoid` | Day types reviewers advise against |
| `best_time.confidence` | Fraction of time-tagged reviews agreeing on time_of_day |
| `best_time.based_on` | Number of reviews tagged as best_time |
| `crowd.level` | Peak crowd level seen (1–5) |
| `crowd.label` | Text label for peak level |
| `crowd.avg_level` | Average crowd level across tagged reviews |
| `crowd.busiest_period` | Day types when crowds are highest |
| `crowd.confidence` | Fraction of crowd-tagged reviews agreeing on the dominant level |
| `crowd.based_on` | Number of reviews tagged as crowd_level |
| `cost.level` | Primary cost level — uses `amount_level` when available, else `sentiment_level` |
| `cost.amount_level` | Cost level derived purely from numeric LKR amounts |
| `cost.sentiment_level` | Cost level derived purely from opinion words ("expensive", "cheap") |
| `cost.median_lkr` | Median of all extracted LKR amounts |
| `cost.fee_type` | Most common fee type (ENTRY_FEE / CONSERVATION_FEE / GUIDE_FEE) |
| `cost.confidence` | Fraction of cost-tagged reviews agreeing on the primary cost level |
| `cost.based_on` | Number of reviews tagged as cost_level |

> **Note on cost signals:** `amount_level` and `sentiment_level` can disagree. For example, a review saying "the Rs 100 parking fee is very expensive for what you get" yields `amount_level = LOW` (100 LKR < 500 threshold) but `sentiment_level = HIGH` (opinion word "expensive"). The `level` field uses the amount-based signal when a numeric value is present, since it is objective. If only opinion words are present, sentiment is used. Researchers should report both fields separately when the two signals disagree.|

### Sample profiles

**Horton Plains National Park** (570 reviews, high data quality):
```json
{
  "place_name": "Horton Plains National Park",
  "total_reviews": 570,
  "best_time": { "time_of_day": "EARLY_MORNING", "season": "DRY_SEASON", "months": ["MARCH","APRIL"], "confidence": 0.89, "based_on": 39 },
  "crowd":     { "level": 5, "label": "PACKED", "avg_level": 3.8, "busiest_period": ["WEEKEND"], "confidence": 0.40, "based_on": 40 },
  "cost":      { "level": "VERY_HIGH", "median_lkr": 7000, "fee_type": "ENTRY_FEE", "confidence": 0.50, "based_on": 55 }
}
```

**Ravana Ella Falls** (499 reviews):
```json
{
  "place_name": "Ravana Ella Falls",
  "total_reviews": 499,
  "best_time": { "time_of_day": "AFTERNOON", "season": "DRY_SEASON", "confidence": 1.0, "based_on": 41 },
  "crowd":     { "level": 5, "label": "PACKED", "avg_level": 4.5, "busiest_period": ["WEEKEND"], "confidence": 0.65, "based_on": 59 },
  "cost":      { "level": "LOW", "median_lkr": 30, "fee_type": "ENTRY_FEE", "confidence": 0.86, "based_on": 9 }
}
```

### Coverage

| Aspect | Places with profile |
|---|---|
| best_time | 144 / 180 |
| crowd | 119 / 180 |
| cost | 133 / 180 |
| All 3 aspects | 90 / 180 |

### Average confidence scores

| Aspect | Avg confidence |
|---|---|
| best_time | 0.64 |
| crowd | 0.75 |
| cost | 0.75 |

### Output files

- `output/aggregation/_poi_profiles.json` — all 180 place profiles

### CLI

```bash
python run_aggregation.py
```

---

## Evaluation Framework

**Code:** `evaluation/sampler.py`, `evaluation/scorer.py`, `run_eval_sample.py`, `run_eval_score.py`

The evaluation framework supports gold-label validation of the classification step (Step 3) to produce precision, recall, and F1 scores suitable for a research paper.

### Step 1 — Generate sample for manual labeling

```bash
python run_eval_sample.py --n 300
```

Produces `output/evaluation/gold_label_sample.csv` with 300 sentences:
- **150 tagged sentences** — what the system said yes to (checks for false positives)
- **150 untagged sentences** — what the system ignored (checks for false negatives)

Sample CSV format:

| place | sentence | system_best_time | system_crowd_level | system_cost_level | true_best_time | true_crowd_level | true_cost_level | notes |
|---|---|---|---|---|---|---|---|---|
| Sigiriya_Lion_Rock | "Go early in the morning..." | 1 | 0 | 0 | _(fill in)_ | _(fill in)_ | _(fill in)_ | |
| Ella_Rock | "The views are amazing..." | 0 | 0 | 0 | _(fill in)_ | _(fill in)_ | _(fill in)_ | |

### Step 2 — Fill in true labels

Open the CSV in Excel or Google Sheets. For each sentence, set:
- `true_best_time` = 1 if the sentence contains timing advice (when to go), else 0
- `true_crowd_level` = 1 if the sentence describes crowd conditions, else 0
- `true_cost_level` = 1 if the sentence mentions cost or fees, else 0

### Step 3 — Compute metrics

```bash
python run_eval_score.py
```

Output format:

```
Evaluating on 300 fully-labeled sentences.

Category             Prec    Rec     F1    Acc  Support
------------------------------------------------------------
best_time           0.821  0.754  0.786  0.923       65
crowd_level         0.803  0.781  0.792  0.917       59
cost_level          0.876  0.832  0.853  0.940       91
------------------------------------------------------------
Macro avg           0.833  0.789  0.810

=== best_time — Error Analysis ===
  False Positives (system said yes, human said no) [12 total]:
    - The views from the top are absolutely stunning at all times of day
  False Negatives (system missed, human said yes) [8 total]:
    - We got there right before it opened which was the smart move
```

### What the metrics mean for the research

| Metric | What it tells you |
|---|---|
| Precision | Of what the system tagged, how much was actually travel advice |
| Recall | Of all the real travel advice in the corpus, how much did the system find |
| F1 | Harmonic mean — the key single-number research metric |
| False positives | Generic or descriptive sentences wrongly tagged |
| False negatives | Advice phrased in ways the system's anchors don't cover |

The threshold of 0.38 can be justified by showing the F1 curve across thresholds (0.30–0.50) with the labeled set — the labeled set makes the threshold choice defensible rather than observation-based.

---

## Output Files

| File | Step | Description |
|---|---|---|
| `output/cleaned_data/{place}.json` | 2 | Cleaned reviews per place |
| `output/cleaned_data/_all_reviews.csv` | 2 | All 48,364 cleaned reviews flat |
| `output/analysis/{place}.json` | 3 | Reviews with analysis_tags + sentences |
| `output/analysis/_best_time.csv` | 3 | All tagged best_time sentences |
| `output/analysis/_crowd_level.csv` | 3 | All tagged crowd_level sentences |
| `output/analysis/_cost_level.csv` | 3 | All tagged cost_level sentences |
| `output/analysis/_all_tagged.csv` | 3 | All tagged sentences combined |
| `output/analysis/_stats.json` | 3 | Detection run statistics |
| `output/extraction/{place}.json` | 4+5 | Reviews with extracted_spans + normalized |
| `output/extraction/_all_spans.csv` | 4+5 | 4,593 extracted span rows flat CSV |
| `output/aggregation/_poi_profiles.json` | 6 | **Final output** — 180 place profiles |

---

## File Structure

```
UGC-analysis/
├── scraper/
│   └── apify_collector.py          # Step 1: TripAdvisor scraper via Apify
├── preprocessing/
│   ├── text_cleaner.py             # HTML/emoji/URL removal
│   ├── language_filter.py          # langdetect English filter
│   ├── deduplicator.py             # TF-IDF cosine deduplication
│   └── pipeline.py                 # Orchestration
├── analysis/
│   ├── detector.py                 # Sentence-BERT contrastive classifier
│   └── pipeline.py                 # Orchestration + progress tracking
├── extraction/
│   ├── extractor.py                # SpaCy span extractor
│   ├── normalizer.py               # Regex normalization to schema
│   └── pipeline.py                 # Orchestration + CSV export
├── aggregation/
│   └── aggregator.py               # Frequency voting + confidence scoring
├── config/
│   └── settings.py                 # Central config (paths, thresholds, model names)
├── output/
│   ├── cleaned_data/               # Step 2 output
│   ├── analysis/                   # Step 3 output
│   ├── extraction/                 # Step 4+5 output
│   └── aggregation/                # Step 6 output (final profiles)
├── run_scraper.py                  # CLI: Step 1
├── run_preprocessing.py            # CLI: Step 2
├── run_analysis.py                 # CLI: Step 3
├── run_extraction.py               # CLI: Step 4+5
├── run_aggregation.py              # CLI: Step 6
├── RESEARCH_PLAN.md
└── DOCUMENTATION.md                # This file
```

---

## How to Run

Run each step in order. Each step is resume-safe — if interrupted, re-running continues from where it stopped.

```bash
# Step 1 — Scrape reviews
python run_scraper.py

# Step 2 — Preprocess
python run_preprocessing.py

# Step 3 — Classify reviews by category
python run_analysis.py
# Options:
#   --no-resume          restart from scratch
#   --place PLACE_KEY    single place only
#   --threshold 0.38     override classification threshold

# Step 4+5 — Extract spans and normalize
python run_extraction.py
# Options:
#   --no-resume
#   --place PLACE_KEY

# Step 6 — Build final POI profiles
python run_aggregation.py
```

### Dependencies

```
sentence-transformers>=2.6.0
spacy>=3.7.0
torch>=2.1.0
transformers>=4.38.0
nltk>=3.8.1
langdetect
scikit-learn
```

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Key Statistics

| Metric | Value |
|---|---|
| Total POIs | 180 |
| Total reviews | 48,364 |
| Reviews tagged best_time | 1,228 (2.54%) |
| Reviews tagged crowd_level | 1,221 (2.52%) |
| Reviews tagged cost_level | 2,144 (4.43%) |
| Total tagged span rows | 4,593 |
| Places with best_time profile | 144 / 180 |
| Places with crowd profile | 119 / 180 |
| Places with cost profile | 133 / 180 |
| Places with all 3 aspects | 90 / 180 |
| Avg best_time confidence | 0.64 |
| Avg crowd confidence | 0.75 |
| Avg cost confidence | 0.75 |
| Classification model | sentence-transformers/all-MiniLM-L6-v2 |
| Classification threshold | 0.38 |
| NLP extraction model | SpaCy en_core_web_sm |
