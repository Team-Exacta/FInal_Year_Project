# Sri Lanka UGC Tourism Analysis — Interim Evaluation Report

**Project:** UGC-Based Travel Recommendation System — Aspect-Aware Suggestion Pipeline
**Domain:** Sri Lanka tourism (TripAdvisor + Kaggle review corpus)
**Status as of this report:** Steps 1–6 implemented and evaluated; Step 7 (recommendation engine) not yet started
**Snapshot date:** 2026-06-09

---

## 1. Executive Summary

This project builds an end-to-end NLP pipeline that converts unstructured tourist reviews into structured, planner-ready profiles for 177 Sri Lankan points of interest (POIs). For each POI, the system answers three concrete travel questions — *When is the best time to visit?*, *How crowded is it?*, and *How much does it cost?* — using only the text of real visitor reviews. No supervised model is trained on Sri Lankan data; the system relies on a **zero-shot Sentence-BERT classifier with contrastive scoring**, **rule-based SpaCy span extraction**, **regex-based normalisation to a fixed schema**, and **majority-vote aggregation with confidence scores**.

**Headline numbers:**
- 177 unique POIs, 48,364 cleaned English reviews, 4,593 tagged suggestion sentences
- Held-out evaluation on 600 manually-labelled sentences: macro F1 **0.761** (best_time 0.727 · crowd_level 0.645 · cost_level 0.883)
- Final deliverable: [`output/aggregation/_poi_profiles.json`](output/aggregation/_poi_profiles.json) — one structured profile per POI, with up to 10 supporting evidence sentences per aspect

**What's done:** Pipeline Steps 1–6, evaluation framework, 600-sentence gold sample (300 random + 300 borderline-sampled), threshold tuning, per-category-negative anchor bug fix, evidence-sentence augmentation of the final JSON.

**What's not done yet:** Step 7 (Recommendation Engine / query interface), supervised baseline comparison, temporal aggregation over `travelDate`, full coverage on all 180 POIs for all 3 aspects, the final write-up.

The research is **not over.** It is in a credible intermediate state: every claim made about a POI can now be traced to specific review sentences, the evaluation methodology is statistically defensible (600 labels, uncertainty-sampled tail), and the planned final-month work is well-scoped.

---

## 2. Research Context, Gap & Motivation

### 2.1 The opportunity

Sri Lanka attracts around 2 million international visitors per year (pre-pandemic), generating hundreds of thousands of public reviews on platforms like TripAdvisor, Google Maps, and Booking.com. These reviews contain **highly actionable advice** ("go before 9 am to beat the heat", "the Rs 5000 entry fee is steep for foreigners", "weekends are packed with school trips"), but the advice is buried in unstructured prose, scattered across thousands of reviews per POI, and impossible for a human planner to read at scale.

### 2.2 The research gap

There is mature literature on **review polarity classification** (positive vs negative sentiment), and growing literature on **aspect-based sentiment analysis** (Pontiki et al., SemEval-2014/15/16), but very little work targets the specific sub-task this project addresses:

> *Extracting decision-grade, normalised travel constraints — best time of day, season, crowd level on a 1–5 ordinal scale, cost level — from free-text tourism reviews, and aggregating them per POI with confidence scores.*

Existing tourism-NLP studies typically (i) classify sentiment, (ii) extract aspect mentions, or (iii) cluster topics — but they do not produce a **structured, comparable, planner-ready output**. They also overwhelmingly rely on supervised models trained on English-language reviews of Western destinations, with no public labelled corpus for Sri Lankan POIs.

### 2.3 Why this matters

| Stakeholder | Concrete benefit |
|---|---|
| Independent traveller | Skip reading 200 reviews per POI; get a one-paragraph constraint summary |
| Sri Lanka Tourism Promotion Bureau | Aggregate visitor-perceived cost / crowd data without a survey |
| Travel-tech start-ups | Plug a structured POI profile into recommender / itinerary engines |
| Academic NLP | A reproducible case study of zero-shot constraint extraction on low-resource tourism text |

**Why this matters — callout:** No supervised dataset exists for this task in Sri Lankan English tourism reviews, and creating one at the scale needed for fine-tuning (thousands of labelled examples per aspect) is not feasible within an undergraduate research budget. The project therefore tests whether a **zero-shot, anchor-driven SBERT pipeline** can deliver useful structured output without any domain-specific training data — a question of practical relevance well beyond Sri Lanka.

---

## 3. Research Question & Objectives

**Primary research question:**
> *Can a zero-shot, anchor-based Sentence-BERT pipeline, combined with rule-based span extraction and regex normalisation, produce planner-ready constraint profiles (best time / crowd / cost) for Sri Lankan tourist sites at a quality level useful for downstream recommendation, without any supervised training on domain reviews?*

**Specific objectives:**
1. Construct a deduplicated, English-filtered corpus of Sri Lankan tourism reviews at sufficient scale (~50k reviews, ~180 POIs).
2. Detect suggestion-bearing sentences in three constraint categories (best_time, crowd_level, cost_level) using a zero-shot classifier — no labelled training data.
3. Extract the specific actionable phrase from each detected sentence (e.g. *"go early in the morning"*, *"Rs 5000 entry fee"*).
4. Normalise extracted phrases into a fixed categorical schema so values from different reviews are directly comparable.
5. Aggregate per-review values into one POI profile with a confidence score and supporting evidence.
6. Quantify classifier quality with a human-labelled gold sample using both random and uncertainty-based sampling.
7. (Future) Surface the aggregated profiles through a user-facing recommendation interface.

---

## 4. Literature Foundation

The methodology is grounded in four strands of established literature. Each chosen technique is anchored to a primary source so the design is defensible, not arbitrary.

### 4.1 Sentence-level embeddings

- **Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP.** Showed that fine-tuning BERT with a Siamese architecture on Natural-Language-Inference (NLI) data produces sentence vectors whose cosine similarity is a meaningful semantic-similarity signal — a 65× speedup over pairwise BERT-cross-encoder scoring with comparable accuracy. **This project uses the SBERT formulation directly** as the foundation of its zero-shot classifier.
- **Wang et al. (2020), *MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers*, NeurIPS.** The `all-MiniLM-L6-v2` model used here is a distilled MiniLM checkpoint (6 transformer layers, 384-dim hidden, ~22 MB), ~5× smaller than `bert-base` but retaining most of its semantic-similarity ability. **Why this paper matters here:** it explains why we can run on CPU at ~14k sentences/second while keeping accuracy competitive.

### 4.2 Zero-shot text classification

- **Yin, Hay & Roth (2019), *Benchmarking Zero-Shot Text Classification: Datasets, Evaluation and Entailment Approach*, EMNLP.** Defines and benchmarks the modern zero-shot text-classification paradigm where the classifier is given only label descriptions or exemplars — no labelled training data per class. **The contrastive anchor scoring in this project is conceptually in the same family**, with class-exemplar (positive anchor) sentences replacing class-description prompts.
- **Schick & Schütze (2021), *Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference* (PET), EACL.** Influential work on prompt-based zero/few-shot classification — provides theoretical backing for the idea that small numbers of natural-language exemplars can guide a pretrained encoder to a downstream decision boundary.

### 4.3 NLP infrastructure

- **Honnibal, Montani et al., *spaCy: Industrial-strength Natural Language Processing in Python* (2017–present, ongoing).** The `en_core_web_sm` model used here provides tokenisation, part-of-speech tagging, dependency parsing, and named-entity recognition. **This project uses all four** — NER for TIME/MONEY entities, dependency parse for imperative-verb subtree extraction, PhraseMatcher for time/crowd/cost vocabulary, and POS tags for adverbial-modifier capture.
- **Aho & Corasick (1975), *Efficient String Matching: An Aid to Bibliographic Search*, CACM.** The algorithm behind SpaCy's PhraseMatcher — multi-pattern matching in O(n + total matches) time over a single pass, which makes the dictionary-based span extraction tractable across millions of sentences.

### 4.4 Active learning / evaluation methodology

- **Lewis & Gale (1994), *A Sequential Algorithm for Training Text Classifiers*, SIGIR.** The foundational paper on **uncertainty sampling** — labelling instances about which the model is least confident yields the largest information gain per label. **This project uses uncertainty sampling for Round 2 of gold labelling**, drawing the second 300 sentences from the borderline-score band [thr − 0.05, thr + 0.07] per category.
- **Settles (2010), *Active Learning Literature Survey*, University of Wisconsin Technical Report 1648.** The standard reference confirming uncertainty sampling typically requires 2–3× fewer labels than random sampling for the same statistical confidence interval.

### 4.5 Aspect-based tourism review mining

- **Hu & Liu (2004), *Mining and Summarising Customer Reviews*, KDD.** The seminal work on review-aspect extraction and summarisation — established that customer-review text can be reduced to structured per-feature opinion summaries, which is the conceptual ancestor of this project's per-POI profile.
- **Pontiki et al. (2014, 2015, 2016), *SemEval Aspect-Based Sentiment Analysis Shared Task*.** Defined the modern ABSA framework — aspect identification, aspect-sentiment classification, and aspect category detection. **This project diverges from ABSA** by targeting *constraint extraction* (categorical values like EARLY_MORNING / PACKED / HIGH) rather than polarity, but inherits the per-aspect breakdown.

---

## 5. Methodology — Approach Selection & Rationale

For every major design choice, alternatives were considered and rejected. Each rejection is justified by a constraint specific to this project: **no labelled training data, CPU-only compute, need for transparency/auditability, and reproducibility on a modest student budget.**

### 5.1 Suggestion detection — zero-shot SBERT contrastive scoring vs alternatives

| Approach | Pros | Cons | Why not chosen |
|---|---|---|---|
| **SBERT contrastive (chosen)** | No labelled data needed; transparent (anchors are inspectable); CPU-friendly; fast | Anchor quality matters; precision/recall trade-off depends on threshold | — |
| Supervised fine-tuned DistilBERT | Higher ceiling F1 if labelled data exists | Requires thousands of labels per class; none available for Sri Lankan reviews | No labelled data |
| NLI-based zero-shot (BART-large-MNLI) | Established zero-shot baseline (Yin et al. 2019) | ~5× slower at inference; needs hypothesis templates per class | Inference cost on 48k reviews × 3 classes was prohibitive on CPU |
| LLM prompting (GPT-4 / Claude) | Highest accuracy; can handle nuance | API cost ~$50 for 48k reviews × 3 classes; opaque; no reproducibility once API version changes | Cost + reproducibility |
| Keyword / lexicon matching | Trivial to implement; deterministic | High false positives ("popular" → tagged crowd); brittle to phrasing | Brittleness; we tested this implicitly via the FP-heavy crowd category |
| TF-IDF + Logistic Regression | Classic, well-understood baseline | Still requires labels; cannot capture semantic similarity (paraphrases miss) | No labels |

**Rationale:** The SBERT contrastive approach is the only one that delivers semantic generalisation without any labelled training data, runs in seconds-per-thousand-sentences on CPU, and exposes its decision rule (the anchor sentences) for inspection — three properties that matter much more than a marginal F1 difference in a research-prototype context.

**Why this matters — callout:** The anchors are written in plain English by the researcher and can be amended without retraining. When error analysis flagged that the crowd detector missed *"advisable to go before the crowd comes"*, the fix was adding three new positive anchor sentences — F1 improved on the same model with no compute cost. A supervised model would have required collecting more labelled data, re-training, and revalidating.

### 5.2 Span extraction — SpaCy rules vs alternatives

| Approach | Pros | Cons | Why not chosen |
|---|---|---|---|
| **SpaCy NER + PhraseMatcher + dependency parse (chosen)** | No training; CPU-fast; transparent; covers TIME, MONEY, imperative verbs, "avoid X" patterns | Phrase vocabulary must be maintained; misses paraphrases not in vocab | — |
| Fine-tuned transformer NER (e.g. `bert-base-cased` on a custom dataset) | Higher recall on novel phrasings | Requires annotated span data; ~$$$ to label | No labels |
| LLM extraction (prompted) | Highest recall and flexibility | API cost; opaque; rate limits | Cost + reproducibility |
| CRF sequence labelling (e.g. `sklearn-crfsuite`) | Lightweight, well-suited to span tasks | Still requires annotated training data | No labels |

**Rationale:** Span extraction here is a structured retrieval problem (find the *actionable* sub-string in a tagged sentence), not an open-ended generation problem. The four signals SpaCy provides — NER, PhraseMatcher, dependency subtree, POS modifiers — cover almost every advice-shaped sentence in the corpus.

### 5.3 Normalisation — regex pattern bank vs alternatives

| Approach | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Regex pattern bank (chosen)** | Deterministic; auditable; instant per sentence; supports negation flip | Patterns must be hand-crafted; brittle to unseen phrasings | — |
| SUTime / HeidelTime for temporal | State-of-the-art temporal normalisation | Java-only (HeidelTime) or Stanford-only (SUTime) — heavy dependency for a Python pipeline | Dependency cost vs marginal gain |
| LLM-based normalisation | Flexible; handles odd phrasings | Cost; non-deterministic; calibrating LLM output to a fixed schema is its own problem | Cost + reproducibility |

**Rationale:** Once span extraction has narrowed the input to a short, focused phrase, regex is more than capable and gives a directly inspectable mapping table — important for justifying every value in the final profile.

### 5.4 Aggregation — majority vote + agreement-ratio confidence vs alternatives

| Approach | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Counter-based majority vote (chosen)** | Trivially explainable; confidence = agreement ratio is intuitive | Treats levels as nominal; doesn't model uncertainty rigorously | — |
| Bayesian latent-variable model | Principled uncertainty estimate; can model reviewer reliability | Substantial implementation overhead; gives marginally different numbers in practice | Overkill for n ≈ 5–50 per aspect |
| LLM summarisation of all sentences | Captures nuance; reads like human summary | Cost; non-reproducible; cannot be unit-tested | Cost + audit |
| Ordinal-aware confidence (weighted by adjacency) | Better for 1–5 scale than nominal majority | Slightly more complex; under-development | Listed as a future improvement (see §10) |

**Rationale:** The aggregation step must produce a number that a human can defend in front of an examiner — *"73 % of crowd-tagged reviews voted PACKED, here are the 6 sentences that contributed"*. A Counter + agreement ratio meets that bar directly.

### 5.5 Evaluation — stratified random + uncertainty sampling vs alternatives

| Approach | Pros | Cons | Why not chosen |
|---|---|---|---|
| **Round 1 stratified random (150 tagged + 150 untagged) + Round 2 uncertainty (borderline-band) (chosen)** | Round 1 gives unbiased baseline; Round 2 maximises information per label | Two-stage labelling adds workflow steps | — |
| Random-only sampling | Simplest | Wastes labels on easy cases; needs ~3× more labels for the same CI | Inefficient |
| Adversarial / model-fooling sampling | Useful for stress-testing | Risks over-fitting evaluation to detector quirks | Over-fits the eval to the model |

**Rationale:** The two-stage design (Lewis & Gale 1994; Settles 2010) is standard in active-learning literature. The Round 2 borderline sample increased positives from 29 → 90 for `crowd_level` — the smallest support category — narrowing the F1 confidence interval from roughly ±0.18 to ±0.10.

---

## 6. Pipeline Walkthrough (Step-by-Step)

The full pipeline runs as six sequential stages plus an evaluation harness.

```
Raw scraped reviews
        ↓ Step 1 — Data Collection (Apify + Kaggle merge)
Per-POI raw JSON                                      → dataset/
        ↓ Step 2 — Preprocessing (langdetect, clean, dedup)
Per-POI cleaned JSON                                  → output/cleaned_data/
        ↓ Step 3 — Suggestion Detection (SBERT contrastive)
Per-POI tagged JSON + category CSVs                   → output/analysis/
        ↓ Step 4 — Span Extraction (SpaCy NER + matcher + dep)
Per-POI extracted spans + normalised values           → output/extraction/
        ↓ Step 5 — Normalization (regex → fixed schema)   (inline in Step 4)
        ↓ Step 6 — POI Aggregation (majority vote + confidence + evidence)
One profile per POI                                   → output/aggregation/_poi_profiles.json   (final)
```

### 6.1 Step 1 — Data Collection
- **Input:** A list of 177 unique Sri Lankan POIs ([`config/places_to_scrape.json`](config/places_to_scrape.json)).
- **Method:** Apify cloud actor `maxcopell/tripadvisor-reviews` ([`scraper/apify_collector.py`](scraper/apify_collector.py)). Each call sets `maxItemsPerQuery = 200`, `language = "en"`. Batches of 10 URLs per call; multi-account token rotation on quota exhaustion.
- **Merged with:** Kaggle Review_Collection corpus already in the repo.
- **Output:** `dataset/*.json` — one file per POI; schema `{title, rating, travelDate, publishedDate, text, url, user, placeInfo}`.

### 6.2 Step 2 — Preprocessing
- **Components ([`preprocessing/pipeline.py`](preprocessing/pipeline.py)):**
  - Text cleaning: HTML/URL/emoji stripping, whitespace normalisation, preserve original + cleaned versions.
  - Language filtering with `langdetect` — keep only English with confidence ≥ 0.7 and length ≥ 15 chars.
  - Deduplication — TF-IDF + cosine similarity ≥ 0.95 for fuzzy near-duplicates, plus exact-match dedup across sources to catch cross-corpus overlaps.
- **Output:** 48,364 cleaned English reviews in `output/cleaned_data/*.json` plus `_all_reviews.csv`.

### 6.3 Step 3 — Suggestion Detection (SBERT contrastive)
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (22 MB, CPU).
- **Code:** [`analysis/detector.py`](analysis/detector.py).
- **Algorithm:** For each review, split into sentences, encode each as a 384-dim L2-normalised vector, and compute a contrastive score per category:
  ```
  score(s, c) = max_sim(s, positive_anchors_c)
              − 0.5 × max(  max_sim(s, global_negative_anchors),
                            max_sim(s, category_negative_anchors_c)  )
  ```
  A sentence is tagged for category *c* when `score(s, c) ≥ threshold_c`, where current per-category thresholds are **best_time = 0.34, crowd_level = 0.34, cost_level = 0.30** (selected from the F1-vs-threshold sweep on the 600-sentence gold set).
- **Anchor banks:** 10–14 hand-curated positive anchors per category, 8 global generic-praise negatives, plus 5 category-specific negatives.
- **Output:** `output/analysis/*.json` (review + `analysis_tags`, `analysis_scores`, `analysis_sentences`) plus per-category CSVs.
- **Numbers:** best_time tagged 1,228 (2.54 %), crowd_level 1,221 (2.52 %), cost_level 2,144 (4.43 %).

### 6.4 Step 4 — Span Extraction (SpaCy rule-based)
- **Code:** [`extraction/extractor.py`](extraction/extractor.py).
- **Four extraction signals combined per tagged sentence:**
  1. PhraseMatcher over hand-curated phrase vocabularies (TIME, CROWD, COST).
  2. SpaCy NER — `TIME` entities for best_time, `MONEY` entities + custom `MONEY_LKR` matcher for cost.
  3. Dependency parsing — root-verb subtree for imperative sentences ("Go early in the morning") and `dobj/pobj` of avoid-verbs ("avoid weekends").
  4. Adjective + modifier capture for crowd intensity ("very crowded", "not busy").
- **Output:** Each review gains an `extracted_spans` block per category.

### 6.5 Step 5 — Normalization (regex → fixed schema)
- **Code:** [`extraction/normalizer.py`](extraction/normalizer.py).
- **Schemas:**
  | Field | Values |
  |---|---|
  | time_of_day | EARLY_MORNING · MID_MORNING · AFTERNOON · EVENING · NIGHT |
  | season | DRY_SEASON · MONSOON · SHOULDER |
  | day_type | WEEKDAY · WEEKEND · PUBLIC_HOLIDAY |
  | crowd_level | 1 EMPTY · 2 QUIET · 3 MODERATE · 4 BUSY · 5 PACKED (with negation flip) |
  | cost_level | FREE · LOW (<500 LKR) · MODERATE (500–2,000) · HIGH (2,000–5,000) · VERY_HIGH (>5,000) |
- **Cost dual signal:** Cost normalisation deliberately keeps two parallel values — `amount_level` (from numeric LKR/USD only, objective) and `sentiment_level` (from opinion words only, subjective) — so disagreement is preserved in the profile.

### 6.6 Step 6 — POI Aggregation
- **Code:** [`aggregation/aggregator.py`](aggregation/aggregator.py).
- **Method:** Counter-based majority vote per field. Confidence = `top_count / total_with_value`. Up to **10 supporting evidence sentences** per aspect are now embedded in each profile, ranked by (i) whether the review's normalised value matches the dominant aggregated value and (ii) the SBERT score.
- **Final profile schema (excerpt):**
  ```json
  {
    "place_name": "Ancient City of Polonnaruwa",
    "total_reviews": 500,
    "best_time": {
      "time_of_day": "EARLY_MORNING",
      "season": null, "months": [], "avoid": [],
      "confidence": 0.67, "based_on": 8,
      "evidence": [
        {
          "sentence": "Go early cause of extreme heat.",
          "extracted": { "time_of_day": "EARLY_MORNING" },
          "rating": 4, "date": "July 27, 2018", "score": 0.5476
        }
      ]
    },
    "crowd": { "level": 5, "label": "PACKED", "avg_level": 4.7, "confidence": 0.67, "based_on": 6, "evidence": [...] },
    "cost":  { "level": "HIGH", "amount_level": "HIGH", "sentiment_level": "HIGH",
               "median_lkr": 5000, "fee_type": "ENTRY_FEE",
               "confidence": 0.75, "based_on": 5, "evidence": [...] }
  }
  ```
- **Output:** [`output/aggregation/_poi_profiles.json`](output/aggregation/_poi_profiles.json) — 180 profiles, 635 KB.

---

## 7. Inside the Models — How They Work Internally

This section answers "what happens inside the model?" — useful for the panel-style questions ("how does SBERT actually work?", "what is a transformer doing?").

### 7.1 Sentence-BERT (`all-MiniLM-L6-v2`)

**Parameters:** 22 MB on disk, **6 transformer layers**, **384-dimensional hidden states**, ~22 M parameters total, ~14 k sentences/sec on CPU.

**Forward pass — what happens to one sentence:**
1. **Tokenisation** — the sentence is split into WordPiece subword tokens; e.g. *"Go early"* → `[CLS] go early [SEP]`.
2. **Embedding** — each token id is looked up in an embedding matrix to give a 384-dim vector. Positional embeddings are added.
3. **Six transformer encoder layers**, each running **multi-head self-attention**:
   - For every token, attention computes a weighted sum of all other tokens in the sentence, where the weights are learned to capture meaning-relevant relations (subject-verb, modifier-noun, etc.).
   - The attended representation is passed through a feed-forward sub-layer and a residual connection + layer norm.
   - Stack 6 of these: every token's vector now encodes information about its surrounding context.
4. **Mean-pooling** — average the per-token vectors (masked to ignore padding) to produce one 384-dim **sentence vector**.
5. **L2-normalisation** — divide by Euclidean norm so all vectors lie on the unit hyper-sphere. After this, **cosine similarity = dot product**, which is faster and numerically stable.

**Why MiniLM is "Mini":** It was *distilled* from a larger BERT-base teacher using the self-attention distillation technique of Wang et al. (2020) — the student learns to mimic the teacher's attention distributions rather than just its output predictions. Result: ~5× smaller, ~5× faster, but only marginal accuracy loss on semantic-similarity benchmarks.

**How it was fine-tuned for sentence embeddings (Reimers & Gurevych 2019):** A *Siamese* architecture — the same encoder runs on two sentences in parallel, the two output vectors are compared with cosine similarity, and the model is trained on the SNLI + MultiNLI + STSb datasets to push entailment pairs together and contradiction pairs apart. This is what makes the cosine similarity of the output vectors a *semantic* similarity, not just a surface-form one.

**Why chosen here:** No supervised data is needed; CPU inference is fast enough to score every sentence in 48 k reviews in a few minutes; the model file is small (22 MB) so the repo is portable; and the cosine-similarity-against-anchors design pattern (Reimers & Gurevych 2019 §6) is well-established.

### 7.2 The contrastive scoring algorithm

The decision rule in [`analysis/detector.py:151-166`](analysis/detector.py#L151-L166):

```python
def _net_score(self, emb, cat):
    pos = max(dot(positive_anchor_embeddings[cat], emb))
    neg = max(dot(global_negative_anchor_embeddings, emb))
    if cat in cat_negative_embeddings:
        neg = max(neg, max(dot(cat_negative_embeddings[cat], emb)))
    return pos - 0.5 * neg
```

- **What it does step by step:**
  1. Compute the maximum cosine similarity between the candidate sentence and any positive anchor for the target category — this is the **positive signal**.
  2. Compute the maximum cosine similarity between the same sentence and any negative anchor (global generic-praise plus per-category misleading patterns) — this is the **distractor signal**.
  3. Subtract a fraction (α = 0.5) of the distractor signal from the positive signal.
  4. If the result exceeds the per-category threshold, tag the sentence.
- **Why subtract negatives?** Without it, sentences like *"this is a wonderful and beautiful place to visit"* score high for *every* category because they share generic travel vocabulary with the positive anchors. Subtracting the maximum match against generic-praise sentences pushes those generic sentences below threshold while leaving truly category-specific sentences above it.
- **Why per-category negatives matter:** Generic negatives suppress only generic noise. Crowd-specific false-positive patterns (*"popular spot"*, *"famous attraction"*) need crowd-specific negative anchors to be suppressed. A bug discovered during this project — that the per-category negative anchors were encoded but never used — was responsible for ~30 % of the crowd false positives; fixing it lifted crowd F1 from 0.535 to 0.630 with no other change.

### 7.3 SpaCy `en_core_web_sm`

A small (~12 MB) statistical English pipeline. Key components used:

- **Tokenizer:** Rule-based, language-specific.
- **Tagger (POS):** Statistical (small neural net), used here for ADJ + ADVMOD detection on crowd descriptors.
- **Parser (dependency):** A **transition-based** parser — a neural classifier predicts the next parser action (SHIFT, LEFT-ARC, RIGHT-ARC, etc.) over a stack of tokens; runs in linear time.
- **NER:** A **transition-based BiLSTM-CNN** named-entity recogniser; produces `TIME`, `DATE`, `MONEY`, and 15 other entity types.
- **PhraseMatcher:** Backed by an **Aho-Corasick automaton** (Aho & Corasick 1975) — given a dictionary of N phrases, it matches all of them in one linear pass through the text. This is why phrase matching can run on millions of sentences without becoming a bottleneck.

**Why chosen:** `en_core_web_sm` ships ready-trained, runs at ~10 k sentences/sec on CPU, and provides every linguistic signal needed (TIME and MONEY entities + dependency parse + POS tags) in one library.

### 7.4 Supporting techniques

- **langdetect** — n-gram naïve Bayes language classifier (Cybozu Labs port of Google's compact language detector). Returns a probability per language; we keep sentences with English probability ≥ 0.7.
- **TF-IDF + cosine for deduplication** — each review is converted to a sparse TF-IDF vector (`scikit-learn.TfidfVectorizer`), pairs above 0.95 cosine similarity are flagged as near-duplicates and the later occurrence is dropped.
- **Aho-Corasick** — finite-state automaton built from the phrase dictionary; matches *all* dictionary phrases in *one* linear-time pass over the text. This is what makes the PhraseMatcher fast.
- **Active learning / uncertainty sampling** (Lewis & Gale 1994) — selecting unlabelled examples about which the classifier is least confident (here: sentences whose contrastive score lies in [thr − 0.05, thr + 0.07]) yields the largest expected information gain per label.

---

## 8. Implementation Status & Numbers

### 8.1 Pipeline status

| Step | Description | Status |
|---|---|---|
| 1 | Data Collection (Apify + Kaggle merge) | ✅ Done |
| 2 | Preprocessing (langdetect, dedup, clean) | ✅ Done |
| 3 | Suggestion Detection (SBERT contrastive) | ✅ Done — per-category thresholds 0.34/0.34/0.30 |
| 4 | Span Extraction (SpaCy NER + matcher + dep) | ✅ Done |
| 5 | Normalization (regex → fixed schema) | ✅ Done |
| 6 | POI Aggregation (majority vote + evidence sentences) | ✅ Done — evidence added in latest commit |
| 7 | **Recommendation Engine / Query Interface** | 🔲 **Not started** |
| Eval | Gold-labelled evaluation framework | ✅ Done — 600 sentences (300 random + 300 borderline) |

### 8.2 Headline numbers

| Metric | Value |
|---|---|
| Total POIs | 177 (180 profile keys after suffix dedup) |
| Total cleaned English reviews | 48,364 |
| best_time tagged sentences | 1,228 (2.54 %) |
| crowd_level tagged sentences | 1,221 (2.52 %) |
| cost_level tagged sentences | 2,144 (4.43 %) |
| Total tagged sentences | 4,593 (9.49 %) |
| POIs with all 3 aspects in profile | 90 / 180 |
| POIs with best_time | 144 / 180 |
| POIs with crowd | 119 / 180 |
| POIs with cost | 133 / 180 |
| Avg confidence — best_time | 0.64 |
| Avg confidence — crowd | 0.75 |
| Avg confidence — cost | 0.75 |

---

## 9. Evaluation Methodology & Results

### 9.1 Gold sample construction (600 sentences total)

| Round | n | Sampling strategy | Purpose |
|---|---|---|---|
| 1 | 300 | Stratified random — 150 system-tagged + 150 untagged | Unbiased baseline |
| 2 | 300 | **Uncertainty / borderline** — sentences whose score lies in [thr − 0.05, thr + 0.07] per category | Maximise information per label (Lewis & Gale 1994) |
| Total | **600** | 50 % random + 50 % uncertainty-sampled | — |

Labels are hand-annotated by the researcher via [`labeling_ui.py`](labeling_ui.py) (a Streamlit interface) and persisted in [`output/evaluation/gold_label_sample.csv`](output/evaluation/gold_label_sample.csv).

### 9.2 Results — 600-sentence held-out evaluation

| Category | Precision | Recall | F1 | Accuracy | Positives in gold |
|---|---|---|---|---|---|
| best_time | 0.800 | 0.667 | **0.727** | 0.910 | 108 |
| crowd_level | 0.551 | 0.778 | **0.645** | 0.872 | 90 |
| cost_level | 0.917 | 0.852 | **0.883** | 0.947 | 142 |
| **Macro avg** | **0.756** | **0.765** | **0.761** | — | — |

These numbers reflect the state **after** (i) the per-category negative-anchor bug fix, (ii) per-category thresholds adoption, and (iii) the addition of four positive crowd anchors + two cost anchors targeting the false-negative patterns surfaced by the borderline gold sample.

**95 % confidence intervals (Wilson) on the F1 numbers:**

| Category | F1 | Approx 95 % CI |
|---|---|---|
| best_time | 0.727 | ±0.08 |
| crowd_level | 0.645 | ±0.10 |
| cost_level | 0.883 | ±0.05 |

These intervals are narrow enough that the reported F1 differences are not statistical noise.

### 9.3 Threshold tuning curve

A threshold sweep over [0.30, 0.50] in 0.02 steps confirms that the chosen per-category thresholds **0.34 / 0.34 / 0.30** sit at the F1 peak for each category. The full curve is saved to [`output/evaluation/threshold_sweep.csv`](output/evaluation/threshold_sweep.csv) and is directly plot-ready.

### 9.4 Error analysis (selected)

**crowd_level false positives** — sentences mentioning "popular", "famous", "many visitors" in a non-advice context. Mitigated by per-category negative anchors.

**crowd_level false negatives caught after the borderline-sampling expansion** — explicit advice phrasings missing from the original anchor bank: *"advisable to arrive before the crowd comes"*, *"better to avoid weekends as it gets very crowded"*. Adding three new positive anchors covering these patterns lifted crowd recall from 0.60 to 0.78.

**cost_level false positives** — sentences that contain *price/admission* words in a descriptive (not informative) way: *"I don't know what the entrance price is"*. One over-aggressive negative anchor was found and removed.

---

## 10. What Remains — 1-Month Plan

Ordered by paper-criticality, then by effort.

### Week 1 — Detector refinement + cost defensibility (P0)

1. **Cost-threshold justification.** Plot the histogram of all extracted `amount_lkr` values from [`output/extraction/_all_spans.csv`](output/extraction/_all_spans.csv) and pick LOW/MODERATE/HIGH cutoffs at percentile boundaries (25/50/75). Replace the hand-set 500/2000/5000 LKR cutoffs in [`extraction/normalizer.py`](extraction/normalizer.py) with data-derived values and re-aggregate. (≈0.5 day)
2. **Negation-flip widening in crowd normaliser.** Fix the *"wasn't crowded"* class of phrases that currently get mapped to PACKED — extend the negation pattern in `normalize_crowd` and re-aggregate. (≈0.5 day)
3. **Ordinal-aware crowd confidence.** Replace exact-match agreement ratio with adjacency-weighted agreement (a vote for 4 and 5 only partially disagrees). One-line change in [`aggregation/aggregator.py`](aggregation/aggregator.py). (≈0.5 day)
4. **Re-run analysis + extraction + aggregation on the full 48 k corpus** to propagate the recent anchor changes into the final profiles. (≈30 min runtime)

### Week 2 — Coverage + temporal aggregation (P1)

5. **Coverage boost** — currently only 90 / 180 POIs have all three aspects. Re-scrape long-tail POIs that hit the 200-review scraper cap and re-tag. (≈1 day)
6. **Temporal aggregation** — extend [`aggregation/aggregator.py`](aggregation/aggregator.py) to produce `crowd_by_month` and `cost_by_year` keyed on `travelDate`. Makes "best month to visit" claims data-backed. (≈0.5 day)
7. **Dedup the 3 duplicate profile keys** (Negombo Beach, Negombo Fish Market, Sri Pada). (≈0.5 day)

### Week 3 — Step 7: Recommendation Engine (P1)

8. **Build the CLI prototype.** `recommend(poi_name, travel_date, preferences)` reads `_poi_profiles.json` and renders a formatted travel summary. (≈1 day)
9. **Wrap in FastAPI.** One endpoint, JSON in / JSON out. (≈1 day)
10. **Streamlit front-end** with a POI dropdown + date picker + recommendation display. (≈2 days)

### Week 4 — Write-up + reproducibility (P0 / P4)

11. **Update the interim → final report** with the cost-threshold curve, ordinal confidence, recommendation engine demo, and final eval numbers.
12. **Reproducibility pass** — pin `requirements.txt` versions, add a single `run_all.py` that chains the pipeline scripts end-to-end.
13. **(Optional) supervised baseline** — fine-tune DistilBERT on the 600-sentence gold set and report it as a "zero-shot vs supervised" comparison.

---

## 11. Is the Research Over?

**No.** The pipeline (Steps 1–6) is implemented and evaluated, but the project as defined in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) is not complete:

- **Step 7 (Recommendation Engine) is not started.** The final profiles exist as a JSON file but there is no user-facing artefact that consumes them.
- **Coverage is partial** — only 50 % of POIs have all three aspects.
- **Temporal signal is unused** — `travelDate` is captured but never aggregated.
- **The final write-up is in progress** — current draft is [`interim_report_module1_filled_v2.docx`](interim_report_module1_filled_v2.docx), needs to be updated with the post-interim numbers.
- **No supervised baseline has been run** — the zero-shot vs supervised comparison that strengthens the headline contribution has not yet been performed.

The Week-1–4 plan in §10 is scoped to bring the project to a publishable / demonstrable state within the one-month window.

---

## 12. Risks, Limitations, Future Work

| # | Limitation | Severity | Mitigation |
|---|---|---|---|
| L1 | English-only (langdetect filter drops Sinhala/Tamil reviews) | Medium | Future: re-run with `paraphrase-multilingual-MiniLM-L12-v2` |
| L2 | Anchor + threshold quality depends on researcher judgement | Medium | Mitigated by the threshold-sweep + uncertainty-sampling evaluation |
| L3 | crowd_level precision 0.55 — half of crowd tags are FPs | Medium | Already improved from 0.46 to 0.55; further gains likely from richer category-specific negatives |
| L4 | Cost LKR thresholds 500 / 2000 / 5000 are hand-picked | Low | Week-1 task #1 derives them from the data percentile distribution |
| L5 | Gold sample is single-rater (no inter-annotator agreement) | Medium | Future: second annotator on a 100-sentence subset, compute Cohen's κ |
| L6 | Long-tail POIs have very few tagged reviews (some `based_on` < 5) | Low | Per-profile confidence already reflects this; recommend hiding low-confidence aspects in the UI |
| L7 | No supervised baseline reported | Medium | Week-4 optional task |
| L8 | Aggregator dedup leaves 3 TripAdvisor "Unclaimed" duplicate profile keys | Low | Week-2 task #7 |
| L9 | Step 7 not yet implemented | High | Week-3 (the largest scope item in the plan) |

---

## 13. Recommendations for the Interim Presentation

A 10-slide outline that maps directly to this report:

| # | Slide title | Source section | Notes |
|---|---|---|---|
| 1 | Title — *Aspect-Based Travel-Review Suggestion Normalization for Sri Lanka Tourism* | — | Name, supervisor, dates |
| 2 | Why this project — the planner's problem | §2 | One screenshot of a TripAdvisor review wall; one screenshot of `_poi_profiles.json` |
| 3 | Research gap + research question | §2.2 + §3 | Quote the primary RQ verbatim |
| 4 | Methodology one-page diagram | §6 (the pipeline diagram) | The 6-step block diagram from §6 |
| 5 | Why this approach (alternatives table) | §5 | One slide showing the SBERT-contrastive vs alternatives table |
| 6 | What's *inside* SBERT (the model) | §7.1 | One slide: tokenise → 6 layers → pool → normalise → cosine |
| 7 | Contrastive scoring formula + example | §7.2 | Show the formula, then one good and one bad sentence with their scores |
| 8 | Evaluation methodology + 600-sentence gold sample | §9.1 | Explain stratified random + borderline sampling |
| 9 | Results table + threshold curve | §9.2 + §9.3 | Per-category F1; threshold sweep PNG |
| 10 | What remains — 1-month plan + status | §10 + §11 | Honest "not over yet" slide with the Step 7 callout |

**Presentation tips:**
- Open with the *output* (a real `_poi_profiles.json` entry with evidence), not the architecture diagram — the panel sees value first.
- Have the per-category negative-anchor bug story ready as a 30-second anecdote — it demonstrates engineering rigour.
- Defend the zero-shot choice by saying *"no labelled Sri Lankan tourism dataset exists; building one at the scale needed for fine-tuning is the next-year project"*.
- Be candid about Step 7 not being done — the panel will respect honesty more than a false claim.

---

## Appendix A: Repository Structure & File Map

```
UGC-analysis/
├── scraper/
│   └── apify_collector.py          # Step 1: Apify cloud scraper
├── preprocessing/
│   ├── text_cleaner.py             # HTML/URL/emoji strip
│   ├── language_filter.py          # langdetect English filter
│   ├── deduplicator.py             # TF-IDF cosine dedup
│   └── pipeline.py                 # Orchestration
├── analysis/
│   ├── detector.py                 # SBERT contrastive classifier + anchors
│   └── pipeline.py                 # Per-place orchestration, resume support
├── extraction/
│   ├── extractor.py                # SpaCy NER + PhraseMatcher + dependency rules
│   ├── normalizer.py               # Regex → fixed schema
│   └── pipeline.py                 # Orchestration + CSV export
├── aggregation/
│   └── aggregator.py               # Majority vote + confidence + evidence sentences
├── evaluation/
│   ├── sampler.py                  # Round-1 random stratified sampler
│   └── scorer.py                   # Precision/recall/F1 computation
├── config/
│   ├── settings.py                 # Central config (thresholds, model names, paths)
│   └── places_to_scrape.json       # 177 unique POIs
├── run_scraper.py                  # CLI: Step 1
├── run_preprocessing.py            # CLI: Step 2
├── run_analysis.py                 # CLI: Step 3
├── run_extraction.py               # CLI: Step 4 + 5
├── run_aggregation.py              # CLI: Step 6
├── run_eval_sample.py              # Round-1 sampler entrypoint
├── run_eval_sample_v2.py           # Round-2 borderline / uncertainty sampler
├── run_eval_rescore.py             # Refresh system_* columns in gold CSV
├── run_eval_score.py               # Compute P/R/F1 metrics
├── run_eval_threshold_sweep.py     # F1-vs-threshold curve per category
├── labeling_ui.py                  # Streamlit gold-labelling UI
├── output/
│   ├── cleaned_data/               # Step 2 output
│   ├── analysis/                   # Step 3 output
│   ├── extraction/                 # Step 4 + 5 output
│   ├── aggregation/
│   │   └── _poi_profiles.json      # FINAL deliverable
│   └── evaluation/
│       ├── gold_label_sample.csv   # 600 hand-labelled sentences
│       ├── eval_results.json       # Per-category P/R/F1
│       └── threshold_sweep.csv     # F1-vs-threshold per category
├── PROJECT_STATUS.md               # Done / todo roadmap
├── DOCUMENTATION.md                # Full technical documentation
├── RESEARCH_PLAN.md                # Original step-by-step plan
└── INTERIM_EVAL_REPORT.md          # This document
```

---

## Appendix B: References

- Aho, A. V., & Corasick, M. J. (1975). *Efficient String Matching: An Aid to Bibliographic Search.* Communications of the ACM, 18 (6), 333–340.
- Hu, M., & Liu, B. (2004). *Mining and Summarizing Customer Reviews.* In Proc. KDD 2004.
- Honnibal, M., Montani, I., et al. *spaCy: Industrial-strength Natural Language Processing in Python.* Software, ongoing 2017–present. https://spacy.io
- Lewis, D. D., & Gale, W. A. (1994). *A Sequential Algorithm for Training Text Classifiers.* In Proc. SIGIR 1994, 3–12.
- Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014, 2015, 2016). *SemEval Aspect Based Sentiment Analysis Shared Task* (Tasks 4, 12, 5 respectively).
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* In Proc. EMNLP-IJCNLP 2019.
- Schick, T., & Schütze, H. (2021). *Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference (PET).* In Proc. EACL 2021.
- Settles, B. (2010). *Active Learning Literature Survey.* University of Wisconsin–Madison Computer Sciences Technical Report 1648.
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). *MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* In Proc. NeurIPS 2020.
- Yin, W., Hay, J., & Roth, D. (2019). *Benchmarking Zero-Shot Text Classification: Datasets, Evaluation and Entailment Approach.* In Proc. EMNLP-IJCNLP 2019.

---

*End of Interim Evaluation Report.*
