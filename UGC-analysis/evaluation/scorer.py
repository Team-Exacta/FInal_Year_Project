"""Computes precision, recall, F1 from a human-labeled CSV.

Expected CSV columns:
  sentence, system_best_time, system_crowd_level, system_cost_level,
  true_best_time, true_crowd_level, true_cost_level

For each category, reports:
  - Precision  = TP / (TP + FP)  — of what the system tagged, how much is right
  - Recall     = TP / (TP + FN)  — of what is truly relevant, how much did we catch
  - F1         = 2 * P * R / (P + R)
  - Accuracy   = (TP + TN) / total

Also shows error buckets:
  - False Positives: system tagged but human said no
  - False Negatives: human said yes but system missed
"""

import csv
import json
import os
from collections import defaultdict

_CATEGORIES = ["best_time", "crowd_level", "cost_level"]

EVAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output", "evaluation"
)
RESULTS_FILE = os.path.join(EVAL_DIR, "eval_results.json")


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def score(csv_path: str, print_errors: bool = True, max_errors: int = 5) -> dict:
    """
    Read labeled CSV and compute per-category evaluation metrics.

    Parameters
    ----------
    csv_path : str
        Path to the gold-labeled CSV (output of sampler.py after human annotation).
    print_errors : bool
        Whether to print example false positives and false negatives.
    max_errors : int
        Max examples to show per error type per category.

    Returns
    -------
    dict
        Per-category metrics + overall summary.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Skip rows where human hasn't filled in labels yet
            if any(row.get(f"true_{c}", "").strip() == "" for c in _CATEGORIES):
                continue
            rows.append(row)

    if not rows:
        print("No fully-labeled rows found. Fill in true_* columns and re-run.")
        return {}

    print(f"Evaluating on {len(rows)} fully-labeled sentences.\n")

    results     = {}
    error_log   = defaultdict(lambda: {"fp": [], "fn": []})

    for cat in _CATEGORIES:
        sys_col  = f"system_{cat}"
        true_col = f"true_{cat}"

        tp = fp = fn = tn = 0
        for row in rows:
            sys  = int(row.get(sys_col, 0) or 0)
            true = int(row.get(true_col, 0) or 0)

            if sys == 1 and true == 1:
                tp += 1
            elif sys == 1 and true == 0:
                fp += 1
                error_log[cat]["fp"].append(row["sentence"])
            elif sys == 0 and true == 1:
                fn += 1
                error_log[cat]["fn"].append(row["sentence"])
            else:
                tn += 1

        total     = tp + fp + fn + tn
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        accuracy  = round((tp + tn) / total, 4) if total > 0 else 0.0

        results[cat] = {
            "precision": precision,
            "recall":    recall,
            "f1":        _f1(precision, recall),
            "accuracy":  accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": tp + fn,  # true positives in gold set
        }

    # Print table
    print(f"{'Category':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} {'Support':>8}")
    print("-" * 60)
    for cat, m in results.items():
        print(f"{cat:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} {m['support']:>8}")

    # Macro averages
    macro_p  = round(sum(m["precision"] for m in results.values()) / len(results), 4)
    macro_r  = round(sum(m["recall"]    for m in results.values()) / len(results), 4)
    macro_f1 = _f1(macro_p, macro_r)
    print("-" * 60)
    print(f"{'Macro avg':<20} {macro_p:>6.3f} {macro_r:>6.3f} {macro_f1:>6.3f}")

    results["macro"] = {"precision": macro_p, "recall": macro_r, "f1": macro_f1}

    # Error analysis
    if print_errors:
        print()
        for cat in _CATEGORIES:
            fps = error_log[cat]["fp"][:max_errors]
            fns = error_log[cat]["fn"][:max_errors]
            if fps or fns:
                print(f"=== {cat} — Error Analysis ===")
            if fps:
                print(f"  False Positives (system said yes, human said no) [{len(error_log[cat]['fp'])} total]:")
                for s in fps:
                    print(f"    - {s[:100]}")
            if fns:
                print(f"  False Negatives (system missed, human said yes) [{len(error_log[cat]['fn'])} total]:")
                for s in fns:
                    print(f"    - {s[:100]}")
            if fps or fns:
                print()

    # Save results
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")

    return results
