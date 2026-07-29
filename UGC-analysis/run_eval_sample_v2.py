"""Uncertainty / borderline sentence sampler — Round 2 of gold labelling.

Round 1 (run_eval_sample.py) drew 300 sentences at random (50% system-tagged,
50% untagged). Most of those landed in the "easy" zone where the detector is
already confident, so each extra label adds little new information — and the
crowd_level category ended up with only 29 positives, too few for a tight F1.

This script samples 300 NEW sentences from the BORDERLINE zone — sentences
whose contrastive score lies just above or just below the category threshold.
Borderline sentences are precisely the false-positive and false-negative
candidates, so each label has the highest possible information gain.

How it works
------------
For each category:
  1. Load the per-place analysis JSONs (contain analysis_scores and the
     matched sentences for tagged reviews).
  2. Identify reviews whose max score falls in [thr - below_band, thr + above_band].
  3. Re-encode every sentence from those reviews with the same SBERT model
     (so we get per-sentence scores, not just review-level max).
  4. Keep sentences whose own score is in the borderline band.
  5. Stratified sample: ~50 just-above + ~50 just-below = ~100 per category.
  6. Append to gold_label_sample.csv (existing 300 labels are preserved;
     duplicates are skipped).

After running this, label the new rows with labeling_ui.py, then re-run
run_eval_score.py to get a tighter F1 on 600 labelled sentences.

Usage:
  python run_eval_sample_v2.py
  python run_eval_sample_v2.py --target-per-cat 100
  python run_eval_sample_v2.py --below 0.05 --above 0.07
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np

from analysis.detector import ReviewAnalysisDetector
from config.settings import (
    ANALYSIS_OUTPUT_DIR, ANALYSIS_MODEL,
    ANALYSIS_SCORE_THRESHOLDS, ANALYSIS_BATCH_SIZE,
)

CATEGORIES = ["best_time", "crowd_level", "cost_level"]
EVAL_DIR = os.path.join("output", "evaluation")
EXISTING_CSV = os.path.join(EVAL_DIR, "gold_label_sample.csv")
OUT_CSV = os.path.join(EVAL_DIR, "gold_label_sample.csv")  # appended in-place


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing(csv_path):
    """Return (rows, set_of_existing_sentence_keys) — sentence_key is lower-cased + stripped."""
    if not os.path.exists(csv_path):
        return [], set()
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = {_sentence_key(r["sentence"]) for r in rows}
    return rows, keys


def _sentence_key(s):
    return " ".join((s or "").lower().split())


def collect_candidate_reviews(below_band, above_band):
    """Iterate every analysis JSON and yield (place_key, review_idx, review, candidate_cats).

    candidate_cats = list of categories where the review's max score is in the
    borderline band [thr - below_band, thr + above_band] for that category.
    """
    out = []
    files = sorted(f for f in os.listdir(ANALYSIS_OUTPUT_DIR)
                   if f.endswith(".json") and not f.startswith("_"))
    for filename in files:
        place_key = filename[:-5]
        with open(os.path.join(ANALYSIS_OUTPUT_DIR, filename),
                  encoding="utf-8") as f:
            try:
                reviews = json.load(f)
            except Exception:
                continue
        for idx, review in enumerate(reviews):
            if not isinstance(review, dict):
                continue
            scores = review.get("analysis_scores", {}) or {}
            cands = []
            for cat in CATEGORIES:
                thr = ANALYSIS_SCORE_THRESHOLDS[cat]
                s = float(scores.get(cat, 0.0))
                if (thr - below_band) <= s <= (thr + above_band):
                    cands.append(cat)
            if cands:
                out.append((place_key, idx, review, cands))
    return out


def _split_sentences(review):
    import nltk
    nltk.download("punkt_tab", quiet=True)
    text = review.get("text_display") or review.get("text") or ""
    title = (review.get("title") or "").strip()
    if title:
        text = title + ". " + text
    return [s.strip() for s in nltk.sent_tokenize(text) if len(s.strip()) >= 15]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-cat", type=int, default=100,
                        help="Approx new labelled sentences per category (default 100)")
    parser.add_argument("--below", type=float, default=0.05,
                        help="Borderline band size below each threshold")
    parser.add_argument("--above", type=float, default=0.07,
                        help="Borderline band size above each threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- Load existing labelled sentences (don't re-label) ---------------
    existing_rows, existing_keys = load_existing(EXISTING_CSV)
    print(f"Existing CSV: {len(existing_rows)} rows already labelled / present.")
    print(f"Per-category thresholds: {ANALYSIS_SCORE_THRESHOLDS}")
    print(f"Borderline band: [thr − {args.below}, thr + {args.above}]")

    # ---- Find candidate reviews (review-level borderline) ----------------
    print("\nScanning analysis output for borderline reviews...")
    cands = collect_candidate_reviews(args.below, args.above)
    print(f"  {len(cands)} reviews with at least one borderline category.")

    # Deduplicate (place, review_idx) — a review may be borderline for >1 cat
    unique_reviews = {(p, i): (p, i, r, c) for p, i, r, c in cands}
    print(f"  {len(unique_reviews)} unique reviews to re-encode.")

    # ---- Re-encode all sentences from those reviews ----------------------
    print("\nLoading SBERT model + re-encoding sentences from borderline reviews...")
    det = ReviewAnalysisDetector(ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLDS,
                                 ANALYSIS_BATCH_SIZE)
    det._ensure_model()

    # Flatten: every (place, review_idx, sentence) gets one row
    flat = []   # (place, review_idx, sentence)
    sent_texts = []
    for (place, idx), (_, _, review, _) in unique_reviews.items():
        for s in _split_sentences(review):
            flat.append((place, idx, s))
            sent_texts.append(s)

    if not sent_texts:
        print("No sentences to encode — nothing to do.")
        return

    print(f"  Encoding {len(sent_texts)} sentences (one-time cost)...")
    embs = det._model.encode(
        sent_texts, batch_size=ANALYSIS_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )

    # Per-category net scores
    per_cat_scores = {c: [det._net_score(embs[i], c) for i in range(len(sent_texts))]
                      for c in CATEGORIES}

    # ---- Per-category borderline candidates ------------------------------
    print("\nFiltering per-sentence borderline candidates...")
    by_cat = defaultdict(lambda: {"above": [], "below": []})
    for i, (place, ridx, sent) in enumerate(flat):
        key = _sentence_key(sent)
        if key in existing_keys:
            continue  # already labelled — skip
        for cat in CATEGORIES:
            thr = ANALYSIS_SCORE_THRESHOLDS[cat]
            s = per_cat_scores[cat][i]
            if thr <= s <= thr + args.above:
                by_cat[cat]["above"].append((place, sent, s, i))
            elif thr - args.below <= s < thr:
                by_cat[cat]["below"].append((place, sent, s, i))

    for cat in CATEGORIES:
        a = len(by_cat[cat]["above"]); b = len(by_cat[cat]["below"])
        print(f"  {cat:<14} candidates: above-thr={a}  below-thr={b}")

    # ---- Stratified sample ------------------------------------------------
    print("\nSampling (stratified above/below per category)...")
    selected = []
    seen_in_round = set()
    half = args.target_per_cat // 2
    for cat in CATEGORIES:
        for band in ("above", "below"):
            pool = by_cat[cat][band]
            random.shuffle(pool)
            picked = 0
            for place, sent, _s, flat_i in pool:
                k = _sentence_key(sent)
                if k in seen_in_round:
                    continue
                seen_in_round.add(k)
                # Build system_* columns from the actual per-sentence scores
                # (so the labeller sees what the system would have decided).
                sys_flags = {c: int(per_cat_scores[c][flat_i] >=
                                    ANALYSIS_SCORE_THRESHOLDS[c])
                             for c in CATEGORIES}
                selected.append({
                    "place":              place,
                    "sentence":           sent,
                    "system_best_time":   sys_flags["best_time"],
                    "system_crowd_level": sys_flags["crowd_level"],
                    "system_cost_level":  sys_flags["cost_level"],
                    "true_best_time":     "",
                    "true_crowd_level":   "",
                    "true_cost_level":    "",
                    "notes":              f"v2:{cat}:{band}",
                })
                picked += 1
                if picked >= half:
                    break
            print(f"  {cat:<14} {band:<5}: picked {picked}")

    print(f"\nNew sentences selected: {len(selected)}")

    # ---- Append to existing CSV (preserve all existing rows + their labels)
    fieldnames = [
        "place", "sentence",
        "system_best_time", "system_crowd_level", "system_cost_level",
        "true_best_time", "true_crowd_level", "true_cost_level",
        "notes",
    ]
    combined = existing_rows + selected
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print(f"\n{OUT_CSV} now has {len(combined)} rows "
          f"({len(existing_rows)} existing + {len(selected)} new).")
    print()
    print("Next steps:")
    print("  1. python labeling_ui.py              # label the new rows")
    print("     (rows with empty true_* columns are the new ones)")
    print("  2. python run_eval_rescore.py         # refresh system_* with latest detector")
    print("  3. python run_eval_score.py           # tighter F1 on combined set")


if __name__ == "__main__":
    main()
