"""Anchor ablation — does deriving anchors from data beat hand-written ones?

Runs the detector's contrastive scoring on the SAME 600-sentence gold set under
three positive-anchor configurations and reports precision / recall / F1:

  A) hand      — the hand-written CATEGORY_ANCHORS (current baseline)
  B) mined     — data-driven medoid anchors from analysis/anchor_mining.py
  C) hybrid    — hand-written + mined combined

The negative-anchor banks (NEGATIVE_ANCHORS, CATEGORY_NEGATIVE_ANCHORS) are held
CONSTANT across configs, so the comparison isolates the effect of how the
*positive* anchors were obtained.

For each config and category we sweep the score threshold and report the F1-peak
operating point (the project already uses per-category thresholds), then the
macro average. This is the headline "before vs after" table for the thesis.

Prerequisite:
  python -m analysis.anchor_mining      # writes analysis/mined_anchors.json

Run:
  python run_anchor_ablation.py
"""

import csv
import json
import os

import numpy as np

from analysis.detector import (
    ReviewAnalysisDetector, CATEGORIES, NEGATIVE_ALPHA,
    CATEGORY_ANCHORS, NEGATIVE_ANCHORS, CATEGORY_NEGATIVE_ANCHORS,
)
from analysis.anchor_mining import load_mined_anchor_bank, MINED_ANCHORS_FILE
from config.settings import (
    ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLDS, ANALYSIS_SCORE_THRESHOLD,
    ANALYSIS_BATCH_SIZE, OUTPUT_DIR,
)

GOLD_CSV = os.path.join(OUTPUT_DIR, "evaluation", "gold_label_sample.csv")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")

# Threshold sweep grid (wider than the production sweep — mined anchors can shift
# the score distribution, so the F1-optimal cutoff may move).
SWEEP_START, SWEEP_STOP, SWEEP_STEP = 0.10, 0.60, 0.01


def _encode(model, texts):
    return np.asarray(
        model.encode(texts, batch_size=ANALYSIS_BATCH_SIZE,
                     normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )


def _net_scores(gold_embs, pos_emb, neg_emb, cat_neg_emb):
    if pos_emb is None or len(pos_emb) == 0:
        # No positive anchors for this category → nothing can be tagged.
        return np.full(gold_embs.shape[0], -1.0, dtype=np.float32)
    pos = (pos_emb @ gold_embs.T).max(axis=0)
    neg = (neg_emb @ gold_embs.T).max(axis=0)
    if cat_neg_emb is not None and len(cat_neg_emb) > 0:
        neg = np.maximum(neg, (cat_neg_emb @ gold_embs.T).max(axis=0))
    return pos - NEGATIVE_ALPHA * neg


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _best_threshold(scores, gold):
    """Sweep thresholds, return the F1-peak operating point."""
    best = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": -1.0,
            "tp": 0, "fp": 0, "fn": 0}
    t = SWEEP_START
    while t <= SWEEP_STOP + 1e-9:
        thr = round(t, 4)
        tp = fp = fn = 0
        for s, g in zip(scores, gold):
            pred = 1 if s >= thr else 0
            if pred and g:
                tp += 1
            elif pred and not g:
                fp += 1
            elif not pred and g:
                fn += 1
        p, r, f1 = _prf(tp, fp, fn)
        if f1 > best["f1"]:
            best = {"threshold": thr, "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
        t += SWEEP_STEP
    return best


def main():
    # ---- Load gold sample -------------------------------------------------
    with open(GOLD_CSV, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if all(r.get(f"true_{c}", "").strip() != "" for c in CATEGORIES)]
    if not rows:
        print("No fully-labeled gold rows found.")
        return
    sentences = [r["sentence"] for r in rows]
    truth = {c: [int(r.get(f"true_{c}", 0) or 0) for r in rows] for c in CATEGORIES}
    print(f"Loaded {len(rows)} fully-labeled gold sentences.")
    print("Gold positives — " + ", ".join(f"{c}: {sum(truth[c])}" for c in CATEGORIES))

    # ---- Build anchor configs --------------------------------------------
    configs = {"hand": CATEGORY_ANCHORS}
    if os.path.exists(MINED_ANCHORS_FILE):
        mined = load_mined_anchor_bank(MINED_ANCHORS_FILE)
        configs["mined"] = mined
        configs["hybrid"] = {c: CATEGORY_ANCHORS[c] + mined.get(c, []) for c in CATEGORIES}
    else:
        print(f"NOTE: {MINED_ANCHORS_FILE} not found — running 'hand' config only. "
              f"Run `python -m analysis.anchor_mining` first for the full ablation.")

    # ---- Encode once ------------------------------------------------------
    det = ReviewAnalysisDetector(ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLD, ANALYSIS_BATCH_SIZE)
    det._ensure_model()
    model = det._model
    gold_embs = _encode(model, sentences)
    neg_emb = _encode(model, NEGATIVE_ANCHORS)
    cat_neg_emb = {c: _encode(model, CATEGORY_NEGATIVE_ANCHORS[c]) for c in CATEGORIES}

    # ---- Evaluate each config --------------------------------------------
    results = {}
    for name, cat_anchors in configs.items():
        print(f"\n{'='*64}\nCONFIG: {name}\n{'='*64}")
        pos_emb = {c: _encode(model, cat_anchors[c]) for c in CATEGORIES}
        per_cat = {}
        print(f"{'category':<14}{'#anch':>6}{'best_thr':>10}{'Prec':>8}{'Rec':>8}{'F1':>8}")
        print("-" * 62)
        for c in CATEGORIES:
            scores = _net_scores(gold_embs, pos_emb[c], neg_emb, cat_neg_emb[c])
            best = _best_threshold(scores, truth[c])
            best["n_anchors"] = len(cat_anchors[c])
            per_cat[c] = best
            print(f"{c:<14}{len(cat_anchors[c]):>6}{best['threshold']:>10.2f}"
                  f"{best['precision']:>8.3f}{best['recall']:>8.3f}{best['f1']:>8.3f}")
        macro_p = round(sum(per_cat[c]["precision"] for c in CATEGORIES) / 3, 4)
        macro_r = round(sum(per_cat[c]["recall"] for c in CATEGORIES) / 3, 4)
        macro_f1 = round(2 * macro_p * macro_r / (macro_p + macro_r), 4) if (macro_p + macro_r) else 0.0
        mean_f1 = round(sum(per_cat[c]["f1"] for c in CATEGORIES) / 3, 4)
        print("-" * 62)
        print(f"{'MACRO':<14}{'':>6}{'':>10}{macro_p:>8.3f}{macro_r:>8.3f}{macro_f1:>8.3f}"
              f"   (mean per-cat F1 = {mean_f1})")
        results[name] = {"per_category": per_cat, "macro_precision": macro_p,
                         "macro_recall": macro_r, "macro_f1": macro_f1, "mean_f1": mean_f1}

    # ---- Winner + save ----------------------------------------------------
    winner = max(results, key=lambda n: results[n]["macro_f1"])
    print(f"\n{'='*64}")
    print("SUMMARY (macro-F1 by config):")
    for name in results:
        star = "  <-- winner" if name == winner else ""
        print(f"  {name:<8} macro-F1 = {results[name]['macro_f1']:.4f}"
              f"  (mean per-cat F1 {results[name]['mean_f1']:.4f}){star}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "anchor_ablation.json"), "w", encoding="utf-8") as f:
        json.dump({"winner": winner, "results": results}, f, indent=2)
    with open(os.path.join(EVAL_DIR, "anchor_ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "category", "n_anchors", "best_threshold",
                    "precision", "recall", "f1", "tp", "fp", "fn"])
        for name, res in results.items():
            for c in CATEGORIES:
                m = res["per_category"][c]
                w.writerow([name, c, m["n_anchors"], m["threshold"], m["precision"],
                            m["recall"], m["f1"], m["tp"], m["fp"], m["fn"]])
            w.writerow([name, "MACRO", "", "", res["macro_precision"],
                        res["macro_recall"], res["macro_f1"], "", "", ""])

    print(f"\nSaved -> {os.path.join(EVAL_DIR, 'anchor_ablation.csv')}")
    print(f"Saved -> {os.path.join(EVAL_DIR, 'anchor_ablation.json')}")


if __name__ == "__main__":
    main()
