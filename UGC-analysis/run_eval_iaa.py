"""Inter-annotator agreement (Cohen's kappa) — Part B, step 4 (kappa-ready).

Gold labels in this project are currently single-rater, which reviewers flag as a
reliability risk (limitation L5). This script computes agreement once a SECOND
annotator has independently labeled a copy of the sheet, for both:

  * binary detection labels  (true_best_time / true_crowd_level / true_cost_level)
      annotator 1: output/evaluation/gold_label_sample.csv
      annotator 2: output/evaluation/gold_label_sample_ann2.csv
  * normalized values        (time_of_day, season, day_type, crowd_ordinal, cost_band)
      annotator 1: output/evaluation/norm_gold_sample.csv
      annotator 2: output/evaluation/norm_gold_sample_ann2.csv

Metrics: percent agreement + Cohen's kappa (unweighted for nominal fields,
quadratic-weighted for the ordinal crowd scale). Rows are matched by sentence; only
rows both annotators labeled are counted.

How to use:
  1. Copy the sheet for a second annotator, e.g.
       cp output/evaluation/norm_gold_sample.csv output/evaluation/norm_gold_sample_ann2.csv
     and have them BLANK the true_* columns and relabel independently.
  2. python run_eval_iaa.py

If a second file is absent this prints setup instructions and exits (nothing to do).
"""

import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "evaluation")

BINARY_FIELDS = ["true_best_time", "true_crowd_level", "true_cost_level"]
NOMINAL_FIELDS = ["true_time_of_day", "true_season", "true_day_type", "true_cost_band"]
ORDINAL_FIELDS = ["true_crowd_ordinal"]


def _agreement(a1, a2, weights=None):
    """Percent agreement + Cohen's kappa over paired label lists (blanks dropped)."""
    pairs = [(x, y) for x, y in zip(a1, a2)
             if str(x).strip() != "" and str(y).strip() != ""]
    if not pairs:
        return None
    y1 = [str(x).strip() for x, _ in pairs]
    y2 = [str(y).strip() for _, y in pairs]
    pct = sum(a == b for a, b in zip(y1, y2)) / len(pairs)
    n_used = len(pairs)
    if weights == "quadratic":
        # Weighted kappa needs numeric ordinal values. Keep ONLY pairs where
        # BOTH annotators gave a digit (drop NOT_STATED etc.) — filtering the two
        # lists separately would misalign the pairs and corrupt the score.
        num = [(int(a), int(b)) for a, b in zip(y1, y2) if a.isdigit() and b.isdigit()]
        if len(num) < 2:
            return {"n": len(pairs), "percent_agreement": round(pct, 4), "kappa": None}
        y1 = [a for a, _ in num]
        y2 = [b for _, b in num]
        n_used = len(num)
    try:
        kappa = cohen_kappa_score(y1, y2, weights=weights)
    except ValueError:
        kappa = None
    return {"n": n_used, "percent_agreement": round(pct, 4),
            "kappa": round(float(kappa), 4) if kappa is not None else None}


def _run(file1, file2, fields, weights=None, label=""):
    if not (os.path.exists(file1) and os.path.exists(file2)):
        print(f"[{label}] second annotator file not found "
              f"({os.path.basename(file2)}) — skipping.")
        return {}
    d1 = pd.read_csv(file1, dtype=str).fillna("").set_index("sentence")
    d2 = pd.read_csv(file2, dtype=str).fillna("").set_index("sentence")
    common = d1.index.intersection(d2.index)
    d1, d2 = d1.loc[common], d2.loc[common]
    print(f"\n=== {label}  ({len(common)} shared sentences) ===")
    print(f"{'field':<20}{'n':>6}{'%agree':>9}{'kappa':>8}")
    print("-" * 43)
    out = {}
    for fld in fields:
        if fld not in d1.columns or fld not in d2.columns:
            continue
        w = weights if fld in ORDINAL_FIELDS else None
        res = _agreement(d1[fld].tolist(), d2[fld].tolist(), weights=w)
        out[fld] = res
        if res:
            k = "n/a" if res["kappa"] is None else f"{res['kappa']:.3f}"
            print(f"{fld:<20}{res['n']:>6}{res['percent_agreement']:>9.3f}{k:>8}")
        else:
            print(f"{fld:<20}     — no overlapping labels —")
    return out


def main():
    print("Inter-annotator agreement (Cohen's kappa)")
    print("kappa guide: <0 poor, 0.01-0.20 slight, 0.21-0.40 fair, "
          "0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.0 almost perfect.")

    any_run = False
    r1 = _run(os.path.join(EVAL_DIR, "gold_label_sample.csv"),
              os.path.join(EVAL_DIR, "gold_label_sample_ann2.csv"),
              BINARY_FIELDS, label="Binary detection labels")
    any_run = any_run or bool(r1)

    r2 = _run(os.path.join(EVAL_DIR, "norm_gold_sample.csv"),
              os.path.join(EVAL_DIR, "norm_gold_sample_ann2.csv"),
              NOMINAL_FIELDS + ORDINAL_FIELDS, weights="quadratic",
              label="Normalized values (crowd = quadratic-weighted)")
    any_run = any_run or bool(r2)

    if not any_run:
        print("\nNo second-annotator files yet. To enable IAA:")
        print("  1. Copy a sheet, e.g. norm_gold_sample.csv -> norm_gold_sample_ann2.csv")
        print("  2. Have a 2nd person blank the true_* columns and relabel independently.")
        print("  3. Re-run: python run_eval_iaa.py")


if __name__ == "__main__":
    main()
