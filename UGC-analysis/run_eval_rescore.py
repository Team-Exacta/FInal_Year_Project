"""Re-score existing labeled sentences with the current detector.

The gold_label_sample.csv has system_* columns frozen from when it was
first generated. After updating anchors and re-running analysis, run this
script to update those system_* columns using the live detector — without
touching the human true_* labels.

Usage:
  python run_eval_rescore.py
"""

import csv
import os

from analysis.detector import ReviewAnalysisDetector
from config.settings import ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLDS, ANALYSIS_BATCH_SIZE

CSV_PATH = os.path.join("output", "evaluation", "gold_label_sample.csv")

def main():
    # Load CSV
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sentences = [r["sentence"] for r in rows]

    print(f"Re-scoring {len(sentences)} sentences with current detector...")
    print(f"  Model      : {ANALYSIS_MODEL}")
    print(f"  Thresholds : {ANALYSIS_SCORE_THRESHOLDS}")
    print()

    # Load detector
    det = ReviewAnalysisDetector(ANALYSIS_MODEL, ANALYSIS_SCORE_THRESHOLDS, ANALYSIS_BATCH_SIZE)
    det._ensure_model()

    embs = det._model.encode(
        sentences, batch_size=ANALYSIS_BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )

    cats = ["best_time", "crowd_level", "cost_level"]
    for i, row in enumerate(rows):
        for cat in cats:
            score = det._net_score(embs[i], cat)
            row[f"system_{cat}"] = 1 if score >= ANALYSIS_SCORE_THRESHOLDS[cat] else 0

    # Write back — preserve all columns including human labels
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. system_* columns updated in {CSV_PATH}")
    print("Now run: python run_eval_score.py")

if __name__ == "__main__":
    main()
