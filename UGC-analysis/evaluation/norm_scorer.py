"""Score the NORMALIZATION step against the value gold layer (Part B, step 3).

Measures whether the extractor+regex-normalizer maps text to the CORRECT canonical
value, separately from the detection step. For each field it reports:

  * accuracy + macro-F1 (categorical fields)
  * a confusion matrix (where/how the rules mislabel — the key "explain it" artifact)
  * coverage / abstention (how often the normalizer produced no value at all)
  * for the ordinal crowd 1-5: MAE, off-by-one accuracy, quadratic-weighted kappa

Denominators:
  * primary   — all TRUE positives for the category (isolates normalization quality
                from the detector; a sentence the detector missed is still scored,
                so this measures the normalizer's own capability).
  * detected  — the subset the detector also tagged (system_<cat> == 1); this is the
                end-to-end behaviour in production. Both are reported.

Abstention handling: a categorical field where the system produced nothing is scored
as predicting "NOT_STATED" (so abstaining on a truly not-stated sentence is correct,
and abstaining on a concrete one is a miss).

Run:  python run_eval_norm_score.py
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")
NORM_CSV = os.path.join(EVAL_DIR, "norm_gold_sample.csv")
RESULTS_FILE = os.path.join(EVAL_DIR, "norm_eval_results.json")

# field -> (category, true col, system col)
CATEGORICAL_FIELDS = [
    ("time_of_day", "best_time",   "true_time_of_day", "system_time_of_day"),
    ("season",      "best_time",   "true_season",      "system_season"),
    ("day_type",    "best_time",   "true_day_type",    "system_day_type"),
    ("cost_band",   "cost_level",  "true_cost_band",   "system_cost_band"),
]
ABSTAIN = "NOT_STATED"


def _pos_mask(df, cat, detected_only=False):
    m = df[f"true_{cat}"].astype(str).str.strip() == "1"
    if detected_only and f"system_{cat}" in df.columns:
        m &= df[f"system_{cat}"].astype(str).str.strip() == "1"
    return m


def _confusion_dict(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "matrix": cm.tolist()}


def _print_confusion(title, cm):
    labels, matrix = cm["labels"], cm["matrix"]
    w = max(10, *(len(l) for l in labels))
    short = [l[:w] for l in labels]
    print(f"\n  Confusion — {title}  (rows=true, cols=system)")
    print("    " + "".join(f"{s:>{w+1}}" for s in short))
    for name, r in zip(short, matrix):
        print(f"    {name:<{w}} " + "".join(f"{v:>{w+1}}" for v in r))


def _score_categorical(df, field, cat, tcol, scol, detected_only=False):
    sub = df[_pos_mask(df, cat, detected_only)]
    sub = sub[sub[tcol].astype(str).str.strip() != ""]        # human-labeled only
    n = len(sub)
    if n == 0:
        return None
    y_true = sub[tcol].astype(str).str.strip().tolist()
    sys_raw = sub[scol].astype(str).str.strip().tolist()
    y_pred = [s if s != "" else ABSTAIN for s in sys_raw]      # abstain -> NOT_STATED
    coverage = float(np.mean([s != "" for s in sys_raw]))
    labels = sorted(set(y_true) | set(y_pred))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    # accuracy among cases where the system actually produced a value
    produced = [(t, s) for t, s in zip(y_true, sys_raw) if s != ""]
    acc_given = (float(np.mean([t == s for t, s in produced])) if produced else 0.0)
    return {
        "n": n, "coverage": round(coverage, 4), "abstention": round(1 - coverage, 4),
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "accuracy_when_produced": round(acc_given, 4),
        "confusion": _confusion_dict(y_true, y_pred, labels),
    }


def _score_crowd(df, detected_only=False):
    sub = df[_pos_mask(df, "crowd_level", detected_only)]
    sub = sub[sub["true_crowd_ordinal"].astype(str).str.strip() != ""]
    n = len(sub)
    if n == 0:
        return None
    true = sub["true_crowd_ordinal"].astype(str).str.strip()
    sysr = sub["system_crowd_ordinal"].astype(str).str.strip()
    coverage = float(np.mean([s != "" for s in sysr]))

    both = [(int(t), int(s)) for t, s in zip(true, sysr)
            if t.isdigit() and s.isdigit()]
    res = {"n": n, "coverage": round(coverage, 4), "abstention": round(1 - coverage, 4),
           "n_scored": len(both)}
    if both:
        yt = np.array([t for t, _ in both]); yp = np.array([s for _, s in both])
        res["exact_accuracy"] = round(float(np.mean(yt == yp)), 4)
        res["mae"] = round(float(np.mean(np.abs(yt - yp))), 4)
        res["off_by_one_accuracy"] = round(float(np.mean(np.abs(yt - yp) <= 1)), 4)
        if len(set(yt.tolist())) > 1 and len(set(yp.tolist())) > 1:
            res["quadratic_weighted_kappa"] = round(
                float(cohen_kappa_score(yt, yp, weights="quadratic")), 4)
        else:
            res["quadratic_weighted_kappa"] = None
    # confusion incl. ABSTAIN
    yt_all = [t for t in true]
    yp_all = [s if s != "" else "ABSTAIN" for s in sysr]
    labels = [str(i) for i in range(1, 6)] + ["ABSTAIN"]
    labels = [l for l in labels if l in set(yt_all) | set(yp_all)]
    res["confusion"] = _confusion_dict(yt_all, yp_all, labels)
    return res


def _score_amount(df):
    sub = df[_pos_mask(df, "cost_level")]
    sub = sub[sub["true_amount_lkr"].astype(str).str.strip() != ""]
    def _num(x):
        x = str(x).replace(",", "").strip()
        try:
            return int(float(x))
        except ValueError:
            return None
    pairs = [(_num(t), _num(s)) for t, s in
             zip(sub["true_amount_lkr"], sub["system_amount_lkr"])]
    pairs = [(t, s) for t, s in pairs if t is not None]
    if not pairs:
        return None
    exact = float(np.mean([t == s for t, s in pairs]))
    coverage = float(np.mean([s is not None for _, s in pairs]))
    return {"n": len(pairs), "exact_match": round(exact, 4),
            "coverage": round(coverage, 4)}


def score_df(df: pd.DataFrame, verbose: bool = True) -> dict:
    results = {"categorical": {}, "crowd_ordinal": {}, "amount_lkr": {}}

    for field, cat, tcol, scol in CATEGORICAL_FIELDS:
        primary = _score_categorical(df, field, cat, tcol, scol, detected_only=False)
        detected = _score_categorical(df, field, cat, tcol, scol, detected_only=True)
        results["categorical"][field] = {"primary": primary, "detected": detected}

    results["crowd_ordinal"] = {"primary": _score_crowd(df, False),
                                "detected": _score_crowd(df, True)}
    results["amount_lkr"] = _score_amount(df)

    if verbose:
        _report(results)
    return results


def _report(results):
    print("\n" + "=" * 70)
    print("NORMALIZATION EVALUATION  (primary = all true positives)")
    print("=" * 70)
    print(f"{'field':<14}{'n':>5}{'cov':>7}{'acc':>7}{'macroF1':>9}{'acc|prod':>10}")
    print("-" * 70)
    any_data = False
    for field, _c, _t, _s in CATEGORICAL_FIELDS:
        m = results["categorical"][field]["primary"]
        if not m:
            print(f"{field:<14}   — no labels yet —")
            continue
        any_data = True
        print(f"{field:<14}{m['n']:>5}{m['coverage']:>7.2f}{m['accuracy']:>7.3f}"
              f"{m['macro_f1']:>9.3f}{m['accuracy_when_produced']:>10.3f}")

    c = results["crowd_ordinal"]["primary"]
    if c and c.get("n_scored"):
        any_data = True
        print(f"\ncrowd (ordinal 1-5): n={c['n']} scored={c['n_scored']} "
              f"coverage={c['coverage']:.2f}")
        print(f"  exact_acc={c.get('exact_accuracy')}  MAE={c.get('mae')}  "
              f"off_by_one={c.get('off_by_one_accuracy')}  "
              f"weighted_kappa={c.get('quadratic_weighted_kappa')}")
    elif c:
        print("\ncrowd (ordinal): — no labels yet —")

    a = results["amount_lkr"]
    if a:
        any_data = True
        print(f"\ncost amount_lkr: n={a['n']} exact_match={a['exact_match']} "
              f"coverage={a['coverage']}")

    if any_data:
        for field, _c, _t, _s in CATEGORICAL_FIELDS:
            m = results["categorical"][field]["primary"]
            if m:
                _print_confusion(field, m["confusion"])
        if c and c.get("confusion"):
            _print_confusion("crowd_ordinal", c["confusion"])
    else:
        print("\nNo value labels found yet. Label with:  streamlit run labeling_ui_norm.py")


def score(csv_path: str = NORM_CSV) -> dict:
    if not os.path.exists(csv_path):
        print(f"Not found: {csv_path}. Run run_eval_norm_sample.py first.")
        return {}
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    results = score_df(df, verbose=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    score()
