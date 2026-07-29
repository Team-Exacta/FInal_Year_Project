# Session Report — Data-Driven Anchors & Normalization Validation

**Purpose of this document:** a complete, self-contained record of a work session that made
two research improvements to the UGC-Analysis pipeline. This chat session will be deleted;
this file is the handoff so a **new chat can pick up exactly where this left off** without
re-deriving context. Read this file first in any follow-up session.

Companion document: [RESEARCH_IMPROVEMENTS.md](RESEARCH_IMPROVEMENTS.md) — the
plain-language, supervisor-facing version of the same story (current state, full pipeline
flow, contribution statement). This report is the more detailed technical/process log.

---

## 0. Why this work happened

The user is doing a **university 4th-year research project**: an aspect-based travel review
pipeline for Sri Lanka tourism (turns ~48k TripAdvisor reviews of ~180 places into
planner-ready profiles for `best_time`, `crowd_level`, `cost_level`). Two things were
identified as **not research-defensible**:

1. The "zero-shot" detector's anchor sentences (`analysis/detector.py :: CATEGORY_ANCHORS`)
   were **hand-written by intuition** — no reproducible way to justify "how did you get
   these?" (the user's explicit concern: *"you can't just say an LLM generated them"*).
2. The regex normalization (`extraction/normalizer.py`) — called the **"core research
   contribution"** — had **never been validated**. The existing 600-sentence gold set only
   checks binary category detection (yes/no), never whether the normalizer maps text to the
   *correct value* (e.g. "go early" → EARLY_MORNING).

Two work items were defined and both are now **complete**:
- **Part A** — make anchors data-driven, prove (or disprove) they're better via ablation.
- **Part B** — build and run a full validation of the normalization step.

A **Part D (baseline comparison)** was scoped but **not yet built** — see §6 Remaining Work.

---

## 1. Part A — Data-driven anchor mining + ablation

### What was built
| File | Purpose |
|---|---|
| `analysis/anchor_mining.py` | Mines anchors from the real corpus: cluster sentences (KMeans) → take each cluster's medoid (most representative real sentence) as an anchor. Includes a **leakage guard** (excludes gold-set sentences from the mining pool) and an **embedding cache** (`output/analysis/_corpus_embeddings.npz` + `_corpus_sentences.json`, ~169k sentences, so re-tuning doesn't require re-encoding). |
| `run_anchor_ablation.py` | Runs the detector's contrastive scoring on the same 600-sentence gold set under 3 anchor configs (hand / mined / hybrid), sweeping thresholds, reporting P/R/F1. |
| `analysis/mined_anchors.json` | Output of the mining — the 12 medoid anchors per category + cluster sizes (evidence artifact). |
| `output/evaluation/anchor_ablation.{csv,json}` | The ablation result table. |

### The recipe (for the "how did you get these" defense)
1. Encode every corpus sentence with SBERT (cached).
2. For each category, keep sentences whose contrastive net-score is in a recall-generous
   band AND whose *best-fitting* category (argmax across all 3) is this one (prevents
   cross-category contamination — a critical fix made mid-session, see §1.2).
3. Exclude any sentence appearing in the 600-sentence gold set (leakage guard — verified
   599/600 excluded).
4. KMeans-cluster the pool (k≈14–60, tuned); take each cluster's **medoid** (real sentence
   closest to the cluster centroid) as an anchor; sort by cluster size (frequency).

### 1.1 Refactor required first
`analysis/detector.py :: ReviewAnalysisDetector.__init__` was made to accept optional
`category_anchors`, `negative_anchors`, `category_negative_anchors` params (default to the
original module constants) — this lets the ablation harness swap anchor banks without
touching globals. **Verified backward-compatible** (existing callers unaffected).

### 1.2 Bugs found and fixed during mining (before the final ablation)
- **Cross-category contamination**: first mining pass let a sentence join *any* category's
  pool if its score cleared that category's threshold, so an ambiguous "beach... not
  crowded... clean" sentence became the top-ranked **cost** anchor. Fixed by assigning each
  sentence to its **argmax** (best-fitting) category only.
- **k too small**: first pass used k=14 across ~169k sentences → huge, vague clusters. Raised
  to k=60 (with a quality/confidence-margin filter tested too).

### 1.3 Final result — HONEST NEGATIVE FINDING

| Config | Macro-F1 | best_time F1 | crowd F1 | cost F1 |
|---|---|---|---|---|
| **hand-written (current, unchanged)** | **0.7659** ← winner | 0.733 | 0.647 | 0.883 |
| mined (data-driven) | 0.6684–0.694 | 0.65–0.71 | 0.55–0.57 | 0.78–0.80 |
| hybrid (hand + mined) | 0.7178–0.744 | **0.716–0.778** (best of all 3) | 0.61–0.65 | 0.80–0.82 |

**Data-driven anchors did NOT beat the hand-written ones.** This is treated as a legitimate,
defensible research result, not a failure — the ablation harness was itself validated: the
"hand" config run through the new harness reproduced the known production score (0.766 ≈ the
real 0.76), and the leakage guard was confirmed working (599/600 gold excluded). One partial
positive: the **hybrid config improved best_time recall** (0.71→0.84 in one run), suggesting
mined anchors add coverage where the hand set was thin — a possible future refinement, not
pursued further this session.

**Production impact: NONE.** `analysis/detector.py`'s `CATEGORY_ANCHORS` were never changed —
still the original 10/14/12 hand-written anchors per category. Only the injection mechanism
(§1.1) was added; it's inert unless explicitly used.

### 1.4 Why it likely lost
The hand-written anchors were already refined via a prior false-negative analysis (see the
"Additions from 600-sample FN analysis" comments already in `detector.py` before this
session). Naive cluster medoids pick the *generic center* of broad clusters, introducing
off-topic noise (e.g. beach/restaurant sentences leaking into the cost pool, arrival-time
sentences into crowd), which hurt precision more than the added recall helped.

---

## 2. Part B — Normalization validation

### What was built
| File | Purpose |
|---|---|
| `run_eval_norm_sample.py` | Builds `output/evaluation/norm_gold_sample.csv` — extends the existing 600-sentence gold set with blank "true value" columns (only for category-positive sentences) + pre-fills the system's current normalized predictions for reference. Re-running preserves any labels already entered. |
| `labeling_ui_norm.py` | Streamlit UI (`python -m streamlit run labeling_ui_norm.py`) for a human to enter the correct canonical value per positive sentence: time_of_day, season, day_type, crowd_ordinal (1–5, **NOT_STATED option added**), cost_band, amount_lkr. |
| `evaluation/norm_scorer.py` + `run_eval_norm_score.py` | Scores system predictions against human labels: accuracy, macro-F1, **confusion matrices**, coverage/abstention rate for categorical fields; MAE + off-by-one + **quadratic-weighted Cohen's κ** for the ordinal crowd scale; exact-match for amount_lkr. Scored **conditionally** (only on true positives, isolating normalization quality from detection quality). **Verified correct against a hand-built synthetic fixture with known expected metrics** before trusting it on real data. |
| `run_eval_iaa.py` | Inter-annotator-agreement (Cohen's κ) computation — **κ-ready stub**; runs and reports real numbers once a second annotator's file exists (e.g. `norm_gold_sample_ann2.csv`); currently prints setup instructions since no second annotator has labeled yet. |
| `analyze_norm_vocab.py` | Separate evidence report: frequency tables of words/phrases in the corpus per category + **rule coverage %** (how often at least one existing regex rule fires) + the vocabulary of sentences the rules currently **miss** (data-driven "what to add next" list). Output: `output/evaluation/norm_vocab_report.json`. |
| `report_stats.py` | Corpus statistics: reviews per place, and how many reviews per place/category carry a detected suggestion. Output: `output/analysis/_review_stats.csv`. |

### 2.1 Vocabulary/coverage evidence (`analyze_norm_vocab.py`, run early in session)
Rule coverage by category (before the fixes in §2.3): best_time 71.5%, **crowd_level 47.7%**
(the weakest — matches crowd being the weakest detection category too), cost_level 80.6%.
Missed-sentence term analysis flagged concrete, data-driven gaps, e.g. crowd rules missing
"tourists"(65×), "overcrowded"(27×) in the sentences they failed to match — these specific
gaps were among the ones fixed in §2.3.

### 2.2 Labeling progress
User labeled **319 of ~340 positive sentences** (108 best_time / 90 crowd / 142 cost need
labels; 319 done as of this report — roughly 21 remain for full coverage). Labels live in
`output/evaluation/norm_gold_sample.csv`; backups exist at
`norm_gold_sample.pre_normfix.bak.csv` and `norm_gold_sample.bak2.csv`.

### 2.3 Real bugs found via labeling, and fixes applied (`extraction/normalizer.py`)
The user found these by hand while labeling — this is the validation loop working as
intended. Root cause common to most: crowd/cost normalizers only read the **extracted span**
(which strips negation/context), not the raw sentence; best_time already read raw sentences,
which is why time-of-day negation worked first but crowd/cost didn't until fixed.

| Bug reported | Example | Fix |
|---|---|---|
| Negation not caught (time) | "Don't climb mid day" tagged AFTERNOON, should be EARLY_MORNING | Added `_resolve_time_of_day()` — negation-aware; if the only time mentioned is negated, infers the complement (hot+negated→EARLY_MORNING; negated AFTERNOON→EVENING; negated EVENING/NIGHT→EARLY_MORNING) |
| "late afternoon" misfiled | "late afternoon for sunset"=EVENING but "late afternoon stroll"=AFTERNOON per the user's own labels | Removed "late afternoon" as an unconditional AFTERNOON/EVENING trigger; it now resolves from context (sunset/dusk cues → EVENING) |
| Weekday/weekend flipped | "avoid weekends", "crowded during weekends" → system said WEEKEND, should recommend WEEKDAY | Added `_resolve_day_type()` — explicit-avoidance-aware (`_DAY_AVOID` regex) and separately, crowd-descriptor-aware (`_DAY_AVOID` extended to include crowd words) so "crowded during holiday and weekend" avoids both and recommends WEEKDAY |
| Multi-time sentences | "morning or evening" → system picked evening, user consistently labeled morning | Added `_TIME_PREFERENCE` tie-break order (EARLY_MORNING > MID_MORNING > EVENING > AFTERNOON > NIGHT) |
| Crowd negation not caught | "not overly crowded" tagged PACKED(5), should be QUIET | Added `_CROWD_NEG` regex (negation within 3 words of a crowd term) → flips to level 2 (QUIET) |
| Missed crowd terms | "Too much of a crowd", "overcrowded" not matched at all | Added patterns to `_CROWD_LEVELS` (PACKED tier: `over ?crowded`, `too much...crowd`; BUSY tier: `lots/loads/plenty of crowd/people`, `quite some people`) |
| Cost negation not caught | "not cheap" tagged LOW, should be HIGH | (Existing `_COST_EVAL` already had "not cheap"→HIGH pattern; verified working with raw-sentence fix) |
| "not much" not recognized as low cost | "charges to enter not much" → missed entirely | Added LOW patterns: "not much", "not a lot", "nominal", "minimal", "small/little/tiny fee" |
| USD amounts not parsed | "25 USD", "$100" → missed | Extended `_AMOUNT_PATTERN` with USD-symbol and USD-word groups; added `USD_TO_LKR = 300` conversion |
| Wrong price picked (local vs foreigner) | "200 rupees for locals and 16000 rupees for foreigners" → system picked 200 | Rewrote `_parse_amount_lkr()` to detect "foreign/tourist" vs "local/resident" context around each number and **prefer the foreigner price** (the tourist-facing planner's relevant number) |
| Curly-quote apostrophes broke negation regexes | "Don't go in weekend" (curly ' U+2019) didn't match `don't` (straight ') | Added `_clean()` helper — normalizes smart quotes to straight ones before all regex matching; applied in all three normalize_* functions |

All normalize functions (`normalize_best_time`, `normalize_crowd`, `normalize_cost`) were
changed to accept and use `raw_sentences` (not just the extracted span) so negation/context
survives. `extraction/pipeline.py` was updated to pass raw sentences to all three (previously
only best_time got them).

### 2.4 Validation results (on 319 labels, scorer output verified against synthetic fixture first)

| Field | n | Accuracy | Coverage | Headline metric |
|---|---|---|---|---|
| **time_of_day** | 108 | 0.880 | 0.583 | macro-F1 0.802 |
| **crowd (ordinal 1–5)** | 90 | exact 0.717 (of 53 scored) | 0.589 | **quadratic-weighted κ 0.808**, MAE 0.396, off-by-one 0.906 |
| **amount_lkr** | 55 | exact-match 0.873 | 0.891 | — |
| cost_band | 142 | 0.563 | 0.725 | macro-F1 0.532 (weakest field — see §6) |
| season | 108 | 0.926* | 0.111 | macro-F1 0.577 |
| day_type | 108 | 0.889* | 0.194 | macro-F1 0.484 |

*season/day_type raw accuracy is inflated by the dominant NOT_STATED class (rarely stated in
text); macro-F1 is the fairer number. Full confusion matrices saved in
`output/evaluation/norm_eval_results.json`.

**Honest caveat for the thesis:** best_time and cost rule patches were partly informed by
these same 319 labels (the labeling loop *found* the bugs), so those accuracies are somewhat
optimistic/fit-to-data. **Crowd is the least biased** of the three (fixes were more general
negation/vocabulary additions, not per-sentence tuning) and it is also the strongest result
(κ 0.81 = "almost perfect" agreement band).

### 2.5 Pipeline re-run (fixes propagated to production)
After the fixes, the user ran:
```
python run_extraction.py --no-resume
python run_aggregation.py
```
This re-processed all 180 places with the corrected normalizer and rebuilt
`output/aggregation/_poi_profiles.json` (the final deliverable). **Coverage improved**
(fewer abstentions):

| | Before fix | After fix |
|---|---|---|
| crowd profiles populated | 101 / 180 | **109 / 180** |
| cost profiles populated | 121 / 180 | **127 / 180** |
| all three populated | 67 / 180 | **75 / 180** |
| best_time profiles populated | 127 / 180 | 127 / 180 (unchanged) |

Verified via file timestamps (all three extraction/aggregation outputs rewritten) and a
spot-check sample profile (Ambewela_Farms) showing sensible values post-fix.

---

## 3. Corpus statistics (via `report_stats.py`)
- **180 places, 48,364 total reviews** (avg 269/place).
- Reviews carrying a detected suggestion: best_time 841 (1.7%), crowd_level 610 (1.3%),
  cost_level 1,036 (2.1%), **any category 2,352 (4.9%)**.
- Per-place breakdown saved to `output/analysis/_review_stats.csv`.
- **Research point:** the signal is sparse (~5% of reviews) — this is why per-place
  aggregation across hundreds of reviews (Step 6) matters; no single review is reliable
  alone.

---

## 4. Complete list of new/modified files this session

**New files:**
- `RESEARCH_IMPROVEMENTS.md` — supervisor-facing summary (keep this updated)
- `analysis/anchor_mining.py`, `analysis/mined_anchors.json`
- `run_anchor_ablation.py`
- `analyze_norm_vocab.py`
- `run_eval_norm_sample.py`
- `labeling_ui_norm.py`
- `evaluation/norm_scorer.py`, `run_eval_norm_score.py`
- `run_eval_iaa.py`
- `report_stats.py`
- This file: `SESSION_REPORT_anchors_and_normalization.md`

**Modified files:**
- `analysis/detector.py` — injectable anchor banks (backward-compatible; production anchors
  unchanged)
- `extraction/normalizer.py` — all the negation/vocabulary/currency/quote fixes in §2.3
- `extraction/pipeline.py` — passes raw sentences to all three normalize_* calls

**Generated/output artifacts:**
- `output/evaluation/anchor_ablation.{csv,json}`
- `output/evaluation/norm_vocab_report.json`
- `output/evaluation/norm_gold_sample.csv` (+ 2 backups)
- `output/evaluation/norm_eval_results.json`
- `output/analysis/_corpus_embeddings.npz`, `_corpus_sentences.json` (mining cache)
- `output/analysis/_review_stats.csv`
- `output/aggregation/_poi_profiles.json` — regenerated with fixed normalizer
- `output/extraction/*.json`, `_normalized_profiles.json`, `_all_spans.csv` — regenerated

---

## 5. Key numbers to remember (quick reference)

- **Detection (unchanged, existing):** macro-F1 **0.76** on 600 gold sentences
  (best_time 0.73, crowd 0.65, cost 0.88).
- **Anchor ablation:** hand-written wins at **0.766**; mined 0.67–0.69; hybrid 0.72–0.74.
- **Normalization validation (319 labels):** time_of_day 0.88 acc / 0.80 F1; crowd κ **0.81**
  / MAE 0.40; amount_lkr 0.87 exact; cost_band 0.56 (weakest); season/day_type high acc but
  low F1 (sparse fields).
- **Corpus:** 48,364 reviews / 180 places; ~4.9% carry a detected suggestion.
- **Coverage after normalizer fixes:** crowd 109/180, cost 127/180, all-three 75/180 places.

---

## 6. Remaining work (not done — for the next session)

Ordered roughly by value/effort:

1. **Finish labeling** — ~21 of 340 positives still unlabeled (319 done). Optional but gives
   full coverage.
2. **Improve cost_band (weakest metric, 0.56 acc/F1)** — likely fix: replace the hand-set LKR
   cutoffs (`_amount_to_level()` in `normalizer.py`: <500 LOW, <2000 MODERATE, <5000 HIGH,
   else VERY_HIGH) with **data-derived percentiles** of the observed `amount_lkr`
   distribution — same "derive from data" defensibility logic used elsewhere. This was
   flagged as a roadmap item from the start (`RESEARCH_IMPROVEMENTS.md` §8) and specifically
   recommended as the next step at the end of this session.
3. **Part D — Baseline comparison** (scoped, not built): run the same 600-sentence gold set
   through (a) a keyword-matching baseline and (b) an LLM few-shot baseline (fixed,
   documented prompt), and tabulate F1 vs free/offline/reproducible against the SBERT method.
   Answers "why not just use an LLM?" — the last major structural gap in the research
   narrative. **This is what the user was about to greenlight when this session ended.**
4. **Small hygiene/bug fixes** (identified, not yet fixed): stale "NLI" docstring in
   `run_analysis.py:4`; stale "0.38 threshold" references in `DOCUMENTATION.md` /
   `PROJECT_STATUS.md` / `run_eval_threshold_sweep.py` docstring (authoritative values are
   the per-category 0.34/0.34/0.30 in `config/settings.py`); `spacy` + `en_core_web_sm`
   missing from `requirements.txt`.
5. **Second annotator → real Cohen's κ** — tooling (`run_eval_iaa.py`) is ready; needs an
   actual second person to independently label a copy of `norm_gold_sample.csv`
   (`norm_gold_sample_ann2.csv`) or `gold_label_sample.csv`.
6. **Step 7 — recommendation/query engine** — not started; the only unbuilt pipeline stage
   (e.g. "find a cheap, quiet, morning-best waterfall"). Large scope, likely a separate
   session/phase.
7. Roadmap-only, not scoped in detail: multilingual (Sinhala/Tamil currently dropped),
   temporal trend analysis.

---

## 7. How to resume in a new chat

Tell the new session: *"Read SESSION_REPORT_anchors_and_normalization.md and
RESEARCH_IMPROVEMENTS.md in the repo root, then continue with [pick from §6]."* Both files
are self-contained — no need to re-explain the project background.
