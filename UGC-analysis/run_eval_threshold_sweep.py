"""Threshold tuning curve for the suggestion detector.

Encodes every sentence in the gold-labeled sample ONCE, then sweeps the
contrastive-score threshold and reports precision / recall / F1 per category
at each value. This makes the chosen 0.38 threshold defensible (the F1 curve
shows it is near-optimal) instead of an intuition-based choice.

It also prints the per-category best threshold — useful because cost_level
and crowd_level peak at different points, motivating per-category thresholds.

Usage:
  python run_eval_threshold_sweep.py
  python run_eval_threshold_sweep.py --start 0.30 --stop 0.50 --step 0.02
  python run_eval_threshold_sweep.py --csv output/evaluation/gold_label_sample.csv

Outputs:
  - console table (one block per category)
  - output/evaluation/threshold_sweep.json   (machine-readable curve)
  - output/evaluation/threshold_sweep.csv    (for plotting in Excel)
"""

import argparse
import csv
import json
import os

from analysis.detector import ReviewAnalysisDetector
from config.settings import (
    ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLD,
    ANALYSIS_SCORE_THRESHOLDS, ANALYSIS_BATCH_SIZE,
)

CATEGORIES = ["best_time", "crowd_level", "cost_level"]
DEFAULT_CSV = os.path.join("output", "evaluation", "gold_label_sample.csv")
EVAL_DIR = os.path.join("output", "evaluation")


def _prf(tp, fp, fn):
    """Return (precision, recall, f1) from raw counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--stop", type=float, default=0.50)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()

    # ---- Load gold sample -------------------------------------------------
    with open(args.csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if all(r.get(f"true_{c}", "").strip() != "" for c in CATEGORIES)]

    if not rows:
        print("No fully-labeled rows found. Fill in true_* columns first.")
        return

    sentences = [r["sentence"] for r in rows]
    truth = {c: [int(r.get(f"true_{c}", 0) or 0) for r in rows] for c in CATEGORIES}
    print(f"Loaded {len(rows)} fully-labeled sentences from {args.csv}")
    print(f"Gold positives — "
          + ", ".join(f"{c}: {sum(truth[c])}" for c in CATEGORIES))

    # ---- Encode every sentence once --------------------------------------
    det = ReviewAnalysisDetector(ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLD,
                                 ANALYSIS_BATCH_SIZE)
    det._ensure_model()
    embs = det._model.encode(
        sentences, batch_size=ANALYSIS_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )

    # Contrastive net score per sentence per category (threshold-independent)
    scores = {c: [det._net_score(embs[i], c) for i in range(len(rows))]
              for c in CATEGORIES}

    # ---- Sweep ------------------------------------------------------------
    thresholds = []
    t = args.start
    while t <= args.stop + 1e-9:
        thresholds.append(round(t, 4))
        t += args.step

    sweep = {c: [] for c in CATEGORIES}
    for cat in CATEGORIES:
        cur_thr = ANALYSIS_SCORE_THRESHOLDS.get(cat, ANALYSIS_SCORE_THRESHOLD)
        print(f"\n=== {cat}   (current threshold: {cur_thr}) ===")
        print(f"{'thresh':>8} {'Prec':>7} {'Rec':>7} {'F1':>7} "
              f"{'TP':>4} {'FP':>4} {'FN':>4}")
        print("-" * 48)
        for thr in thresholds:
            tp = fp = fn = 0
            for i in range(len(rows)):
                pred = 1 if scores[cat][i] >= thr else 0
                gold = truth[cat][i]
                if pred == 1 and gold == 1:
                    tp += 1
                elif pred == 1 and gold == 0:
                    fp += 1
                elif pred == 0 and gold == 1:
                    fn += 1
            precision, recall, f1 = _prf(tp, fp, fn)
            sweep[cat].append({
                "threshold": thr, "precision": round(precision, 4),
                "recall": round(recall, 4), "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
            })
            mark = "  <- current" if abs(thr - cur_thr) < 1e-9 else ""
            print(f"{thr:>8.2f} {precision:>7.3f} {recall:>7.3f} "
                  f"{f1:>7.3f} {tp:>4} {fp:>4} {fn:>4}{mark}")

        best = max(sweep[cat], key=lambda r: r["f1"])
        print(f"  best F1 = {best['f1']:.3f} at threshold {best['threshold']}")

    # ---- Best-threshold summary ------------------------------------------
    print("\n" + "=" * 48)
    print("Per-category best threshold (by F1):")
    for cat in CATEGORIES:
        best = max(sweep[cat], key=lambda r: r["f1"])
        cur_thr = ANALYSIS_SCORE_THRESHOLDS.get(cat, ANALYSIS_SCORE_THRESHOLD)
        cur = next((r for r in sweep[cat]
                    if abs(r["threshold"] - cur_thr) < 1e-9), None)
        cur_f1 = f"{cur['f1']:.3f}" if cur else "n/a (off-grid)"
        print(f"  {cat:<14} best {best['threshold']} (F1 {best['f1']:.3f})"
              f"  |  current {cur_thr} (F1 {cur_f1})")

    # ---- Save -------------------------------------------------------------
    os.makedirs(EVAL_DIR, exist_ok=True)
    json_path = os.path.join(EVAL_DIR, "threshold_sweep.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)

    csv_path = os.path.join(EVAL_DIR, "threshold_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "threshold", "precision", "recall",
                         "f1", "tp", "fp", "fn"])
        for cat in CATEGORIES:
            for r in sweep[cat]:
                writer.writerow([cat, r["threshold"], r["precision"],
                                 r["recall"], r["f1"], r["tp"], r["fp"], r["fn"]])

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}  (plot F1 vs threshold per category in Excel)")


if __name__ == "__main__":
    main()
