# UGC-Based Travel Recommendation System — Research Plan
## Sri Lanka Tourism · Aspect-Aware Suggestion Pipeline

---

## Overview

Extract structured travel advice (best time, crowd level, cost) from user-generated reviews and aggregate them into per-place recommendation profiles.

---

## Pipeline

### Step 1 — Data Collection ✅ DONE
Collect reviews from TripAdvisor (Apify cloud scraper) and Kaggle Review_Collection corpus.

**Output:** 48,364 cleaned English reviews across 180 Sri Lanka POIs

```json
{
  "poi": "Sigiriya",
  "review": "Go early to avoid crowds. Tickets cheaper online.",
  "date": "2023-07"
}
```

---

### Step 2 — Preprocessing ✅ DONE
Remove HTML, emojis, URLs. Sentence splitting. Language filtering. Deduplication.

**Tools:** NLTK, langdetect, scikit-learn (TF-IDF dedup)

**Output:** `output/cleaned_data/` — per-place JSON + `_all_reviews.csv`

```
["go early to avoid crowds", "tickets cheaper online"]
```

---

### Step 3 — Suggestion Detection & Classification ✅ DONE
Tag reviews containing timing advice, crowd information, or cost/fee mentions.

**Method:** Sentence-BERT (`all-MiniLM-L6-v2`) with contrastive scoring against
category anchor sentences. Negative anchors subtract generic travel praise to
reduce false positives.

**Output:** `output/analysis/` — per-place JSON + category CSVs

| Category | Tagged reviews |
|---|---|
| `best_time` | 1,228 |
| `crowd_level` | 1,221 |
| `cost_level` | 2,144 |

```
"go early to avoid crowds"  → best_time ✓, crowd_level ✓
"tickets cheaper online"    → cost_level ✓
```

---

### Step 4 — Suggestion Span Extraction ✅ DONE
Extract the exact actionable instruction from each tagged sentence.

**Method:** SpaCy `en_core_web_sm` — PhraseMatcher (TIME/CROWD/COST phrases), NER (TIME/MONEY), dependency parsing (imperative verb subtrees, avoid-objects), custom MONEY_LKR Matcher

**Input:**
```
"You should go early in the morning to avoid the crowds"
```
**Output:**
```
"go early in the morning"
"avoid the crowds"
```

---

### Step 5 — Normalization ✅ DONE (Core Research Contribution 🔥)
Map free-text suggestions to structured, comparable values.

#### 5a — TIME Normalization
**Input:** `"go early"`, `"before 9am"`, `"in August"`, `"dry season"`

**Tools:** SUTime / HeidelTime (optional) + rule mapping

**Output:**
```json
{
  "time_of_day": "EARLY_MORNING",
  "before": "09:00",
  "months": ["DECEMBER", "JANUARY", "FEBRUARY", "MARCH", "APRIL"],
  "season": "DRY_SEASON"
}
```

**Scale / Categories:**
- Time of day: EARLY_MORNING · MID_MORNING · AFTERNOON · EVENING · NIGHT
- Season: DRY_SEASON · MONSOON · SHOULDER
- Day type: WEEKDAY · WEEKEND · PUBLIC_HOLIDAY

---

#### 5b — CROWD Normalization
**Input:** `"very crowded"`, `"no queue"`, `"packed"`, `"had the place to ourselves"`

**Output:**
```json
{
  "crowd_level": 4,
  "crowd_label": "BUSY"
}
```

**Scale:**
| Level | Label | Example phrases |
|---|---|---|
| 1 | EMPTY | "had it to ourselves", "no one there" |
| 2 | QUIET | "few visitors", "not crowded" |
| 3 | MODERATE | "some tourists", "manageable" |
| 4 | BUSY | "very crowded", "long queue" |
| 5 | PACKED | "overrun", "impossible to enjoy" |

---

#### 5c — COST Normalization
**Input:** `"cheap"`, `"discount"`, `"free"`, `"Rs. 1500"`, `"expensive"`

**Tools:** Duckling (money extraction) + rule mapping

**Output:**
```json
{
  "cost_level": "LOW",
  "amount_lkr": null,
  "rule": "FREE_ENTRY",
  "tip": "BUY_ONLINE"
}
```

**Scale:** FREE · LOW · MODERATE · HIGH · VERY_HIGH

---

### Step 6 — POI Aggregation ✅ DONE
Combine all normalized values across reviews into one structured profile per place.

**Method:** Frequency counting + majority voting + confidence scoring

**Output:**
```json
{
  "poi": "Sigiriya",
  "review_count": 538,
  "best_time": {
    "time_of_day": "EARLY_MORNING",
    "confidence": 0.82,
    "months": ["JANUARY", "FEBRUARY", "MARCH"],
    "avoid": "WEEKENDS"
  },
  "crowd": {
    "peak_level": 5,
    "peak_period": "10:00–14:00",
    "quiet_period": "EARLY_MORNING"
  },
  "cost": {
    "level": "HIGH",
    "amount_lkr": 5000,
    "tip": "BUY_ONLINE"
  }
}
```

---

### Step 7 — Recommendation Engine 🔲 TODO
Given a user query (POI name, travel date, preferences), return structured advice.

**Input:** `"I want to visit Sigiriya next Saturday morning"`

**Output:**
```
📍 Sigiriya — Visit Summary
─────────────────────────────
⏰ Best time:   Before 9am (early morning)
📅 Best months: December – April (dry season)
⚠️  Avoid:      Weekends, public holidays
👥 Crowds:      Peak 10am–2pm (level 5/5)
               Quiet before 9am (level 2/5)
💰 Cost:        ~5,000 LKR (foreigners)
💡 Tip:         Buy tickets online for discount
```

---

## File Structure

```
UGC-analysis/
├── dataset/                        # Raw Apify scrape output
├── Review_Collection/              # Kaggle corpus
├── output/
│   ├── cleaned_data/               # Step 2 output (48,364 reviews)
│   └── analysis/                   # Step 3 output (tagged reviews + CSVs)
├── preprocessing/                  # Step 2 code
├── analysis/                       # Step 3 code
├── scraper/                        # Step 1 code
├── config/settings.py
└── RESEARCH_PLAN.md                # This file
```

---

## Status

| Step | Description | Status |
|---|---|---|
| 1 | Data Collection | ✅ Done |
| 2 | Preprocessing | ✅ Done |
| 3 | Suggestion Detection & Classification | ✅ Done |
| 4 | Suggestion Span Extraction | ✅ Done |
| 5 | Normalization (TIME / CROWD / COST) | ✅ Done |
| 6 | POI Aggregation | ✅ Done |
| 7 | Recommendation Engine | 🔲 Todo |
