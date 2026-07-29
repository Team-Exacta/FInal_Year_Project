# Defense Cheat-Sheet — Module 1 (Suggestion Detection & Normalization)

**Project:** Aspect-Based Travel Review Suggestion Normalization · Sri Lanka tourism
**Pipeline:** 48,364 cleaned reviews · 177 POIs · macro-F1 = 0.76 on held-out gold sample
**One-line defense of the whole system:** *every value is corpus-derived and measured — validated against human labels — not guessed or blindly LLM-generated.*

---

## 1. Why SBERT + zero-shot?

**Zero-shot** = classify into categories with **no labeled training set** — you define each category's meaning (via anchor sentences) and match new text by semantic similarity.
- **Why:** no labeled training data existed at the start; easy to adjust (edit anchors, not retrain); reproducible.
- **Alternatives:** supervised fine-tuned BERT (needs big labeled set), keyword rules (miss paraphrases), per-sentence LLM (costly, non-reproducible at 48k scale).

**SBERT** (Sentence-BERT, Reimers & Gurevych 2019) = encodes a whole sentence into one vector so similarity = cosine of vectors. Model: `all-MiniLM-L6-v2` (~22 MB, ~14k sentences/sec on CPU).
- **Why:** fast, CPU-only, free, offline, reproducible, sentence-level embeddings.
- **Alternatives:** Universal Sentence Encoder (heavier), OpenAI/Cohere embedding APIs (paid, online, non-reproducible), TF-IDF (no paraphrase understanding).

**Precise description of the method** — not pure zero-shot classification but **contrastive anchor scoring:**
`score = max(sim to positive anchors) − 0.5 × max(sim to negative anchors)`, then per-category threshold.
Negative anchors stop generic praise ("amazing place, highly recommend") from firing a tag.

---

## 2. Anchors — definition, evaluation, growth

**How defined:** hand-written seed sentences per category (fact-dense positives + generic-praise negatives). LLM may have helped *draft* seeds, but they are **human-curated and evaluation-gated** — nothing ships unmeasured.

**How evaluated:**
- **Threshold sweep** (`run_eval_threshold_sweep.py`) → precision/recall/F1 curve per category, proving cutoffs (best_time 0.34, crowd 0.34, cost 0.30) sit near the F1 peak — chosen, not guessed.
- **Anchor-ablation harness** — detector accepts injectable anchor banks, so swapping anchor sets and measuring F1 uses identical code.
- New anchors are labeled *"Additions from 600-sample FN analysis"*: false negatives found in evaluation → new anchors → re-run sweep to confirm no regression.

**Growing anchors:** human-in-the-loop, evaluation-gated — mine false negatives → add covering anchor → re-measure. Could be automated (cluster unmatched high-signal sentences) but currently curated.

---

## 3. Normalization — method, why, structure origin

**Method:** rule-based / regex mapping to a **fixed controlled vocabulary:**
- TIME_OF_DAY: EARLY_MORNING / MID_MORNING / AFTERNOON / EVENING / NIGHT
- SEASON: DRY / MONSOON / SHOULDER
- CROWD_LEVEL: 1 EMPTY … 5 PACKED
- COST_LEVEL: FREE / LOW (<500) / MODERATE (500–1500) / HIGH (1500–5000) / VERY_HIGH (>5000) LKR

**Cost cutoffs are data-driven**, not guessed: 500 / 1500 / 5000 LKR sit on the 33rd / 67th / 92nd percentiles of all 708 extracted fee amounts (`output/evaluation/cost_threshold_analysis.json`), splitting fees into 33% / 34% / 25% / 8% bands.

**Why rule-based:** transparent & auditable (each value traces to a regex), deterministic/reproducible, and handles hard cases embeddings can't — **negation** ("not crowded" → QUIET), **avoidance/complement** ("avoid weekends" → recommend WEEKDAY), **foreigner-vs-local pricing** ("16000 for foreigners").
- **Alternatives:** LLM-to-JSON (non-reproducible, hallucination, cost), supervised NER (needs labeled spans), pure numeric ML (can't express rules cleanly).

**Where the structure (EVENING, MORNING…) came from:** a **data-grounded taxonomy**, not invented — read a sample of detected sentences → observe recurring tourist vocabulary ("sunrise," "before 9," "avoid the afternoon heat") → group into a small closed set → write regex mapping surface phrases to each bucket. Specificity ordering and tie-breaks were tuned to match how human annotators labeled the same sentences.

---

## 4. Final evaluation & purpose

**How evaluated:** human-labeled **gold sample** scored with **precision / recall / F1 per category** → **macro-F1 = 0.76**. Two-round sampling: Round 1 random (300); Round 2 **borderline-zone** (`run_eval_sample_v2.py`) — labels near the decision threshold where the classifier is most uncertain, maximizing information per label.

**Purpose:** convert 48k unstructured reviews (177 POIs) into **planner-ready, evidence-backed profiles** answering *"when to go, how crowded, what cost?"* Each aggregated value now carries up to **10 ranked source sentences** (auditable back to real reviews). End goal (Step 8, not yet built): feed profiles into a **trip-recommendation / query interface**.

---

## Likely follow-up traps & crisp rebuttals

| Trap question | Crisp answer |
|---|---|
| *"How do you know your gold labels are correct?"* | 2 independent annotators; **Cohen's κ = 0.90–0.92 (almost perfect)** on detection labels ([iaa_results.json](output/evaluation/iaa_results.json)) — the eval rests on agreed labels, not one opinion. |
| *"Did an LLM write your anchors/schema?"* | LLM may draft seeds, but the **measurement** (threshold sweep + F1 vs human labels), not the LLM, decides what stays. |
| *"Why 0.5 for the negative weight (alpha)?"* | Contrastive penalty tuned so generic praise scores below threshold while real signal survives; validated on the gold sample. |
| *"Per-category thresholds — isn't that overfitting?"* | Each category's score distribution is shifted by its own negative anchors, so a shared cutoff is wrong; each threshold is the F1 peak on held-out labels. |
| *"macro-F1 0.76 — is that good? compared to what?"* | vs a keyword baseline (0.750): SBERT wins on **precision** (0.756 vs 0.715), not just F1, and generalizes beyond a fixed word list. vs majority-class normalization: regex wins big on time_of_day (0.873 vs 0.600) and cost_band (0.718 vs 0.381). |
| *"How did you choose the rule words — guess, or a mechanism?"* | Corpus **frequency analysis** ([analyze_norm_vocab.py](analyze_norm_vocab.py)): rules track the most frequent expressions in the tagged sentences; it also reports rule coverage and the frequent terms the rules miss. Data-derived and measured, not asserted. |
| *"Any field where your method loses to the baseline?"* | **day_type** was (0.476); a data-driven rule fix raised it to macro-F1 0.667 / acc 0.907. Remaining small gap on acc-when-produced is honest — it's the lowest-support field (n=16). |
| *"Why regex over an LLM for normalization?"* | Reproducibility + auditability + explicit handling of negation/complement/foreigner-pricing that a similarity score gets wrong. |
| *"How do you know the taxonomy is complete?"* | Buckets are corpus-derived from observed vocabulary; unmatched sentences are reviewed in FN analysis and fold back into anchors/rules. |
