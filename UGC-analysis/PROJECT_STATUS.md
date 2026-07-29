# UGC Analysis — Project Status & Roadmap

**Domain:** Sri Lanka tourism · Aspect-Based Travel Review Suggestion Normalization
**Goal:** Extract planner-ready constraints (best time, crowd, cost) from tourist reviews and aggregate them into per-POI profiles.
**Snapshot date:** 2026-07-28

This document is the single-page answer to *"what has been done"* and *"what should be done next"*. For deep technical details, see [DOCUMENTATION.md](DOCUMENTATION.md). For the original step plan, see [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

---

## 1. At-a-glance status

| # | Stage | Module / Code | Status | Output |
|---|---|---|---|---|
| 1 | Data collection | [scraper/apify_collector.py](scraper/apify_collector.py), [run_scraper.py](run_scraper.py) | ✅ Done | `dataset/`, [Review_Collection/](Review_Collection/) |
| 2 | Preprocessing | [preprocessing/](preprocessing/), [run_preprocessing.py](run_preprocessing.py) | ✅ Done | `output/cleaned_data/` |
| 3 | Suggestion detection (SBERT) | [analysis/detector.py](analysis/detector.py), [run_analysis.py](run_analysis.py) | ✅ Done | `output/analysis/` |
| 4 | Span extraction (SpaCy) | [extraction/extractor.py](extraction/extractor.py) | ✅ Done | `output/extraction/{place}.json` |
| 5 | Normalization (regex schema) | [extraction/normalizer.py](extraction/normalizer.py), [run_extraction.py](run_extraction.py) | ✅ Done | `output/extraction/_all_spans.csv` |
| 6 | POI aggregation (majority vote) | [aggregation/aggregator.py](aggregation/aggregator.py), [run_aggregation.py](run_aggregation.py) | ✅ Done | **[output/aggregation/_poi_profiles.json](output/aggregation/_poi_profiles.json)** *(final deliverable)* |
| 7 | Evaluation framework | [evaluation/](evaluation/), [labeling_ui.py](labeling_ui.py), `run_eval_*.py` | ✅ Done (sample = 600) | [output/evaluation/eval_results.json](output/evaluation/eval_results.json) |
| 8 | Recommendation engine / query interface | — | 🔲 **Not started** | — |
| 9 | Paper write-up | [interim_report_module1_filled_v2.docx](interim_report_module1_filled_v2.docx) (interim only) | 🟡 In progress | — |

**Headline numbers:** 177 unique POIs · 48,364 cleaned English reviews · 4,593 tagged sentences · macro-F1 = 0.76 on a 600-sentence held-out gold sample (per-category thresholds, tuned via threshold sweep).

---

## 2. What was done — detailed

### Step 1 · Data Collection ✅

- 177 unique Sri Lanka POIs across all 9 provinces (temples, parks, beaches, waterfalls, historical sites, museums, etc.).
- Reviews scraped from TripAdvisor via the **Apify** cloud API, capped at 200 per POI.
- Merged with the **Kaggle Review_Collection** corpus already in the repo.
- Output schema includes: `title`, `rating`, `travelDate`, `publishedDate`, `text`, `url`, `user`, `placeInfo` (id, name, webUrl).
- Final list of POIs and per-place review counts is in [config/places_to_scrape.json](config/places_to_scrape.json).

### Step 2 · Preprocessing ✅

| Sub-step | Method | Threshold |
|---|---|---|
| Text cleaning ([text_cleaner.py](preprocessing/text_cleaner.py)) | strip HTML / URLs / emojis, normalize whitespace; keep both `text` and `text_clean` | — |
| Language filtering ([language_filter.py](preprocessing/language_filter.py)) | `langdetect` on cleaned text | confidence ≥ 0.7, len ≥ 15 chars |
| Deduplication ([deduplicator.py](preprocessing/deduplicator.py)) | TF-IDF + cosine similarity | sim ≥ 0.95 → drop |

**Result:** 48,364 deduped English reviews stored as one JSON per POI under `output/cleaned_data/`.

### Step 3 · Suggestion Detection (SBERT contrastive) ✅

- Model: `sentence-transformers/all-MiniLM-L6-v2` (~22 MB, CPU-friendly).
- For each sentence: `net_score = max_sim(sent, positive_anchors) − 0.5 · max_sim(sent, negative_anchors)`, where the negative term now takes the max over **both** the global and the per-category negative anchors (previously the per-category negatives were computed but never applied).
- A sentence is tagged for a category if `net_score ≥ threshold`, using **per-category thresholds** chosen from the F1-peak of the threshold sweep: best_time 0.34 · crowd_level 0.34 · cost_level 0.30 ([run_eval_threshold_sweep.py](run_eval_threshold_sweep.py)).
- Anchor banks: hand-curated positive + generic negative + per-category negative anchors in [analysis/detector.py](analysis/detector.py); crowd/cost positives expanded from a 600-sample false-negative analysis.
- **Zero-shot — no model training was performed**; SBERT is used pretrained.

**Tag rates (out of 48,364 reviews):**

| Category | Tagged | Rate |
|---|---|---|
| best_time | 1,228 | 2.54 % |
| crowd_level | 1,221 | 2.52 % |
| cost_level | 2,144 | 4.43 % |
| **Any** | **4,593** | **9.49 %** |

### Step 4 · Span Extraction (SpaCy) ✅

Per-category extractors in [extraction/extractor.py](extraction/extractor.py):

- **best_time** — `PhraseMatcher` over a curated time vocabulary, SpaCy `TIME` NER, dependency-tree imperative subtrees, "avoid X" object extraction.
- **crowd_level** — `PhraseMatcher` over crowd descriptors, `ADJ + ADVMOD/NEG` patterns ("very crowded", "not busy"), temporal context.
- **cost_level** — SpaCy `MONEY` NER, custom `MONEY_LKR` matcher (`Rs 200`, `LKR 5,000`, `1500 rupees`), fee vocabulary, evaluation adjectives.

Rule-based throughout — no per-task training.

### Step 5 · Normalization (regex → fixed schema) ✅

[extraction/normalizer.py](extraction/normalizer.py) maps free-text spans to comparable categorical values:

| Field | Values |
|---|---|
| `time_of_day` | EARLY_MORNING · MID_MORNING · AFTERNOON · EVENING · NIGHT |
| `season` | DRY_SEASON · MONSOON · SHOULDER |
| `day_type` | WEEKDAY · WEEKEND · PUBLIC_HOLIDAY |
| `crowd_level` | 1 EMPTY · 2 QUIET · 3 MODERATE · 4 BUSY · 5 PACKED (with negation flip) |
| `cost_level` | FREE · LOW (<500 LKR) · MODERATE (500–1,500) · HIGH (1,500–5,000) · VERY_HIGH (>5,000) — **cutoffs data-driven** from the 33rd/67th/92nd percentiles of 708 extracted fee amounts ([cost_threshold_analysis.json](output/evaluation/cost_threshold_analysis.json)) |

Cost normalization keeps **two parallel signals** so the disagreement is preserved:
- `amount_level` — derived from numeric LKR/USD only (objective)
- `sentiment_level` — derived from opinion words only (subjective)
- `level` defaults to `amount_level` when present.

### Step 6 · POI Aggregation ✅

[aggregation/aggregator.py](aggregation/aggregator.py) collapses all per-review normalized values into one profile per place using **majority voting + agreement-ratio confidence** (`top_count / total_with_value`).

Final profile shape:

```json
{
  "place_name": "Sigiriya Lion Rock",
  "total_reviews": 538,
  "best_time": { "time_of_day": "...", "season": "...", "months": [...], "avoid": [...], "confidence": 0.74, "based_on": 45 },
  "crowd":     { "level": 5, "label": "PACKED", "avg_level": 4.7, "busiest_period": [...], "confidence": 0.86, "based_on": 20 },
  "cost":      { "level": "HIGH", "amount_level": "...", "sentiment_level": "...", "median_lkr": 100, "fee_type": "ENTRY_FEE", "confidence": 0.60, "based_on": 31 }
}
```

Final file: **[output/aggregation/_poi_profiles.json](output/aggregation/_poi_profiles.json)** (177 unique POIs / 180 profile keys).

**Coverage of profiles (out of 180):**

| Aspect | Places with profile | Avg confidence |
|---|---|---|
| best_time | 144 / 180 | 0.64 |
| crowd | 119 / 180 | 0.75 |
| cost | 133 / 180 | 0.75 |
| **All 3** | **90 / 180** | — |

### Step 7 · Evaluation framework ✅

- 600-sentence gold sample: Round 1 = 300 stratified (random tagged/untagged) via [evaluation/sampler.py](evaluation/sampler.py); Round 2 = +300 **borderline-zone** sentences (scores near the decision threshold, highest information gain) via [run_eval_sample_v2.py](run_eval_sample_v2.py).
- Manual labels collected via the Tk-based [labeling_ui.py](labeling_ui.py).
- Scoring in [evaluation/scorer.py](evaluation/scorer.py); final scores in [output/evaluation/eval_results.json](output/evaluation/eval_results.json).

**Held-out detector results (600-sentence gold sample, per-category thresholds):**

| Category | Precision | Recall | F1 | Accuracy | Support |
|---|---|---|---|---|---|
| best_time | **0.800** | 0.667 | **0.727** | 0.910 | 108 |
| **crowd_level** | **0.551** ⚠ | 0.778 | **0.645** | 0.872 | 90 |
| cost_level | **0.917** | 0.852 | **0.883** | 0.947 | 142 |
| **Macro avg** | 0.756 | 0.765 | **0.761** | — | — |

→ `cost_level` is essentially solved; `crowd_level` improved from F1 0.535 → **0.645** after the per-category-negative fix + expanded anchors, but remains the weakest link (precision 0.55 — still the main false-positive source).

**Inter-annotator agreement (2 independent annotators, [iaa_results.json](output/evaluation/iaa_results.json)):**

| Layer | Field | % agree | Cohen's κ |
|---|---|---|---|
| Detection | best_time | 97.3% | **0.904** almost perfect |
| Detection | crowd_level | 98.0% | **0.921** almost perfect |
| Detection | cost_level | 96.6% | **0.904** almost perfect |
| Normalization | cost_band | 74.3% | 0.640 substantial |
| Normalization | crowd_ordinal (weighted) | 59.1% | 0.674 substantial |
| Normalization | time_of_day | 69.2% | 0.574 moderate |
| Normalization | season / day_type | 92% / 85% | 0.469 / 0.475 (moderate; high raw agreement, κ deflated by label imbalance) |

→ **Detection labels are highly reliable (κ ≈ 0.91)** — this validates the F1 evaluation above. Normalization labels are substantial on the main fields; the two low-κ categorical fields still have high raw agreement.

**Baseline comparison ([baseline_results.json](output/evaluation/baseline_results.json), [run_eval_baseline.py](run_eval_baseline.py)):**

- *Detection — keyword matching vs SBERT:* macro-F1 **0.750 (keyword) vs 0.761 (SBERT)**. Margin is thin on F1, but SBERT is more **precise** (0.756 vs 0.715) — it trades recall for fewer false tags, and generalizes beyond a fixed keyword list. (Gold sample is borderline-weighted, which flatters the lexical baseline.)
- *Normalization — majority-class vs regex:* regex **wins clearly** on the main fields — time_of_day 0.873 vs 0.600, cost_band 0.689 vs 0.381, season 0.583 vs 0.462.
- *day_type — fixed.* Was the one field losing to baseline (0.476). Data-driven fix (removed bare "holiday" which matched tourists' own trips; removed descriptive crowd words from the avoidance flip to match documented intent) improved it to **macro-F1 0.667 (was 0.484), accuracy 0.907, acc-when-produced 0.550** — far fewer false positives. Still the lowest-support field (n=16); remaining gap is honest and small.

**Rule provenance (answers "how did you choose the rule words?"):** [analyze_norm_vocab.py](analyze_norm_vocab.py) mines the most frequent unigrams/bigrams in the sentences the detector actually tagged per category, reports **rule coverage** (% of tagged sentences a rule fires on), and lists frequent terms the rules currently **miss** ([norm_vocab_report.json](output/evaluation/norm_vocab_report.json)). So the rule vocabulary is corpus-frequency-derived and measured — not asserted.

---

## 3. Known issues & weaknesses

| # | Issue | Where | Severity |
|---|---|---|---|
| W1 | Crowd detector precision still low (**0.55**, was 0.45) — improved via per-category negative anchors + expanded positives, but generic "popular place" descriptions still leak in | [analysis/detector.py](analysis/detector.py) — crowd anchors / negatives | Med (was High) |
| W2 | Coverage gap: only **90 / 180** places have all three aspects; 36 have no `best_time` profile, 61 have no `crowd` profile | aggregation output | High |
| W3 | Three duplicate profile entries (Negombo Beach, Negombo Fish Market, Sri Pada) caused by TripAdvisor "Unclaimed…" suffix in scraped names | extraction filenames + profile keys | Low (cosmetic; deduped in places list, not in profile dict) |
| W4 | English-only — all non-English reviews are filtered out by `langdetect` (Sinhala/Tamil reviews are dropped) | [preprocessing/language_filter.py](preprocessing/language_filter.py) | Medium |
| ~~W5~~ | ✅ **Resolved** — per-category thresholds (0.34/0.34/0.30) now chosen from the F1-peak of the threshold sweep ([run_eval_threshold_sweep.py](run_eval_threshold_sweep.py)) | [analysis/detector.py](analysis/detector.py) | Resolved |
| ~~W6~~ | ✅ **Resolved** — cost LKR cutoffs (500/1,500/5,000) now placed on the 33rd/67th/92nd percentiles of 708 observed fee amounts ([cost_threshold_analysis.json](output/evaluation/cost_threshold_analysis.json)) | [extraction/normalizer.py](extraction/normalizer.py) | Resolved |
| W7 | Reviews scraped capped at 200 per POI; long-tail places have very low `based_on` (some profiles built from <5 reviews → low statistical reliability) | [config/settings.py](config/settings.py) | Medium |
| W8 | `crowd.confidence` is computed against the dominant single level — for a 1–5 ordinal scale this under-counts adjacent agreement (a 4 and 5 vote disagree under this metric) | [aggregation/aggregator.py:139](aggregation/aggregator.py#L139) | Low |
| W9 | No temporal analysis — `travelDate` and `publishedDate` are captured but never used (e.g., crowd patterns by month/year are not extracted) | aggregation step | Medium |
| W10 | No recommendation/query interface — final profiles exist but no end-user-facing artifact (the goal stated in [RESEARCH_PLAN.md](RESEARCH_PLAN.md) Step 7) | — | High |
| ~~W11~~ | ✅ **Resolved** — gold sample expanded to 600 sentences via borderline sampling; supports now best_time 108 · crowd 90 · cost 142 | [output/evaluation/eval_results.json](output/evaluation/eval_results.json) | Resolved |

---

## 4. Recommended work — prioritized

### P0 · Required to publish a credible paper

1. 🟡 **Crowd detector (W1) — improved, not finished.** Done: applied per-category negative anchors (bug fix), added crowd positives from FN analysis, lowered threshold to 0.34 → F1 0.535→0.645. Still open: precision 0.55. Next lever — add "popular spot / famous / well-known" *negative* anchors and inspect the remaining 57 crowd false positives in `eval_results.json`.

2. ✅ **Threshold tuning curve (W5) — done.** [run_eval_threshold_sweep.py](run_eval_threshold_sweep.py) sweeps thresholds and reports per-category F1; per-category cutoffs (0.34/0.34/0.30) set to each category's F1 peak. Curves in [threshold_sweep.csv](output/evaluation/threshold_sweep.csv).

3. ✅ **Expand the gold sample (W11) — done.** 300 → 600 sentences via borderline sampling ([run_eval_sample_v2.py](run_eval_sample_v2.py)); supports best_time 108 · crowd 90 · cost 142.

4. ✅ **Cost-threshold justification (W6) — done.** Cutoffs 500/1,500/5,000 LKR placed on the 33rd/67th/92nd percentiles of 708 observed fees; analysis in [cost_threshold_analysis.json](output/evaluation/cost_threshold_analysis.json); extraction + aggregation re-run.

### P1 · Improve coverage / quality

5. **Boost coverage to 150+/180 places with all 3 aspects (W2).** Two levers:
   - Re-scrape long-tail places that hit the scraper's 200-cap with a second pass.
   - Loosen the per-category threshold for low-coverage aspects.

6. **Use the temporal signal (W9).** Add a lightweight aggregator pass keyed on `travelDate` month → produces `crowd_by_month` and `cost_by_year`. This unlocks "best month to visit" claims that are currently asserted but not data-backed.

7. **Ordinal-aware crowd confidence (W8).** Change the crowd confidence formula to weight adjacent-level agreement (e.g., `(top + 0.5·neighbours) / total`). One-line change in [aggregation/aggregator.py](aggregation/aggregator.py).

8. **Clean up the 3 duplicate-profile keys (W3).** Add the suffix-stripper from `_place_display_name` *before* writing to `output/extraction/` so duplicates never enter the aggregator.

### P2 · Step 7 — Recommendation Engine (the missing pipeline stage)

9. **Build the user-facing query layer.** Smallest viable artifact = a Python function `recommend(poi_name, travel_date, preferences)` that reads `_poi_profiles.json` and renders the formatted block from [RESEARCH_PLAN.md](RESEARCH_PLAN.md) Step 7. Phase plan:
   - **9a**: CLI prototype (1 day).
   - **9b**: FastAPI wrapper around it (1 day).
   - **9c**: Streamlit / simple HTML front-end with a POI dropdown + date picker (2 days).

10. **Cross-POI recommender.** Use the POI profiles + a content-based similarity (cosine on aspect vectors: time-of-day one-hot + crowd 1-5 + cost 1-5) to answer *"places similar to Sigiriya for a quiet morning visit"*.

### P3 · Stretch / future research

11. **Multilingual support (W4).** Add Sinhala+Tamil reviews via `xlm-r`-based SBERT (`paraphrase-multilingual-MiniLM-L12-v2`) — keeps the same anchor-based zero-shot architecture.

12. **Aspect-aware recommendation re-ranking.** Combine the structured profiles with sentence-level retrieval so the recommender returns *evidence quotes*, not just normalized values.

13. **Compare against a supervised baseline.** Fine-tune a small classifier (e.g., DistilBERT) on the expanded gold sample and report it as the supervised baseline — gives the paper a "zero-shot vs supervised" headline number.

14. **External validation.** Cross-check the aggregated `cost.median_lkr` against published Sri Lanka Tourism Board entry-fee data; compute a calibration plot.

### P4 · Reporting / dissemination

15. **Finish the interim → final report.** Current draft: [interim_report_module1_filled_v2.docx](interim_report_module1_filled_v2.docx). Sections to add: improved crowd-detector results (after P0-1), threshold curves (after P0-2), expanded gold sample (after P0-3), cost-threshold justification (after P0-4).

16. **Reproducibility pass.** Pin all dep versions in [requirements.txt](requirements.txt) (currently has loose `>=` bounds), add a top-level `Makefile` or `run_all.py` to chain the six pipeline scripts.

---

## 5. Suggested execution order (next ~2–3 weeks)

```
Week 1 — Fix the weak link & make the eval defensible
  ├─ P0-1  Crowd detector improvements      (1–2 days)
  ├─ P0-2  Threshold tuning curve            (0.5 day)
  └─ P0-4  Cost threshold from data          (0.5 day)

Week 2 — Expand evidence & coverage
  ├─ P0-3  Expand gold sample to 600        (3 days, mostly labeling)
  ├─ P1-5  Coverage boost (re-scrape + retag)(1 day)
  └─ P1-6  Temporal aggregator              (0.5 day)

Week 3 — Build Step 7 + write up
  ├─ P2-9  Recommendation engine MVP        (2 days)
  ├─ P1-7  Ordinal crowd confidence          (0.5 day)
  └─ P4-15 Update report with new numbers   (2 days)
```

---

## 6. Open questions for the supervisor

1. Is the unsupervised / zero-shot framing acceptable as the paper's headline contribution, or does the committee expect a supervised baseline comparison (P3-13)?
2. Should the recommender (Step 7) be in the same paper or split into a follow-up?
3. Is single-language (English) sufficient for the dissertation scope, or is multilingual coverage a hard requirement?
4. Target venue and required min sample size for the evaluation set?
