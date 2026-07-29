"""Simple baselines, for comparison against the SBERT detector + regex normalizer.

Research needs a "compared to what?" reference point. This script measures two
deliberately naive baselines on the SAME gold sheets — no new labeling, and it
does NOT touch eval_results.json / norm_eval_results.json (writes its own file).

  1. DETECTION baseline — keyword matching.
     A sentence is tagged for a category if it contains any keyword for that
     category. Keywords are lifted from the detector anchors + normalizer
     vocabulary, so it is a fair lexical baseline, not a strawman. No semantics,
     no negation handling. Scored against true_* in gold_label_sample.csv and
     compared to the SBERT detector (eval_results.json).

  2. NORMALIZATION baseline — majority class.
     For each normalized field, always predict the single most common gold value.
     Accuracy = frequency of that value among labeled positives. Compared to the
     regex normalizer (norm_eval_results.json). Shows the rules beat "always guess
     the most frequent answer".

Usage:
  python run_eval_baseline.py

Outputs:
  output/evaluation/baseline_results.json
"""

import csv
import json
import os
from collections import Counter

CATEGORIES = ["best_time", "crowd_level", "cost_level"]
EVAL_DIR = os.path.join("output", "evaluation")
GOLD_CSV = os.path.join(EVAL_DIR, "gold_label_sample.csv")
NORM_CSV = os.path.join(EVAL_DIR, "norm_gold_sample.csv")
SBERT_RESULTS = os.path.join(EVAL_DIR, "eval_results.json")
NORM_RESULTS = os.path.join(EVAL_DIR, "norm_eval_results.json")
OUT_FILE = os.path.join(EVAL_DIR, "baseline_results.json")

# Keyword banks — drawn from the detector anchors and normalizer patterns.
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
KEYWORDS = {
    "best_time": ([
        "morning", "afternoon", "evening", "night", "sunrise", "sunset", "dawn",
        "dusk", "midday", "noon", "dry season", "monsoon", "rainy", "wet season",
        "shoulder season", "weekday", "weekend", "holiday", "poya", "early",
        "am", "pm", "o'clock", "peak season", "off season",
    ] + MONTHS),
    "crowd_level": [
        "crowd", "crowded", "packed", "busy", "quiet", "calm", "peaceful",
        "empty", "deserted", "queue", "queues", "tourist", "tourists", "people",
        "overcrowded", "overrun", "swarming", "rush", "jam", "solitude",
    ],
    "cost_level": [
        "fee", "price", "priced", "ticket", "cost", "costs", "rupee", "rupees",
        "rs", "lkr", "dollar", "dollars", "usd", "free", "expensive", "cheap",
        "affordable", "admission", "charge", "entry", "entrance", "pricey",
        "value", "conservation fee",
    ],
}


def _f1(p, r):
    return round(2 * p * r / (p + r), 4) if (p + r) else 0.0


def _kw_tag(sentence: str, cat: str) -> int:
    """1 if any category keyword occurs as a whole-ish token in the sentence."""
    text = " " + sentence.lower().replace(",", " ").replace(".", " ") + " "
    for kw in KEYWORDS[cat]:
        if " " in kw:
            if kw in sentence.lower():
                return 1
        elif f" {kw} " in text:
            return 1
    return 0


# ---------------------------------------------------------------------------
# 1. Detection baseline — keyword matching
# ---------------------------------------------------------------------------

def detection_baseline() -> dict:
    rows = []
    with open(GOLD_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any(row.get(f"true_{c}", "").strip() == "" for c in CATEGORIES):
                continue
            rows.append(row)

    results = {}
    for cat in CATEGORIES:
        tp = fp = fn = tn = 0
        for row in rows:
            pred = _kw_tag(row["sentence"], cat)
            true = int(row.get(f"true_{cat}", 0) or 0)
            if pred and true:
                tp += 1
            elif pred and not true:
                fp += 1
            elif not pred and true:
                fn += 1
            else:
                tn += 1
        p = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
        r = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
        results[cat] = {"precision": p, "recall": r, "f1": _f1(p, r),
                        "accuracy": round((tp + tn) / len(rows), 4),
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": tp + fn}
    mp = round(sum(results[c]["precision"] for c in CATEGORIES) / 3, 4)
    mr = round(sum(results[c]["recall"] for c in CATEGORIES) / 3, 4)
    results["macro"] = {"precision": mp, "recall": mr, "f1": _f1(mp, mr)}
    results["_n"] = len(rows)
    return results


# ---------------------------------------------------------------------------
# 2. Normalization baseline — majority class per field
# ---------------------------------------------------------------------------

NORM_FIELDS = {
    "time_of_day": "true_time_of_day",
    "season": "true_season",
    "day_type": "true_day_type",
    "crowd_ordinal": "true_crowd_ordinal",
    "cost_band": "true_cost_band",
}


def normalization_baseline() -> dict:
    if not os.path.exists(NORM_CSV):
        return {}
    rows = list(csv.DictReader(open(NORM_CSV, encoding="utf-8")))
    out = {}
    for field, col in NORM_FIELDS.items():
        vals = [r[col].strip() for r in rows
                if r.get(col, "").strip() not in ("", "NOT_STATED")]
        if not vals:
            continue
        counts = Counter(vals)
        majority, top = counts.most_common(1)[0]
        out[field] = {
            "n": len(vals),
            "majority_class": majority,
            "majority_baseline_accuracy": round(top / len(vals), 4),
            "distinct_values": len(counts),
        }
    return out


def main():
    print("=" * 64)
    print("BASELINE EVALUATION (keyword detection + majority-class normalization)")
    print("=" * 64)

    det = detection_baseline()
    print(f"\n1. DETECTION — keyword baseline  (n={det['_n']} sentences)\n")
    print(f"{'Category':<14}{'Prec':>7}{'Rec':>7}{'F1':>7}")
    print("-" * 35)
    for c in CATEGORIES:
        m = det[c]
        print(f"{c:<14}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}")
    print("-" * 35)
    print(f"{'Macro':<14}{det['macro']['precision']:>7.3f}"
          f"{det['macro']['recall']:>7.3f}{det['macro']['f1']:>7.3f}")

    # Side-by-side vs SBERT
    if os.path.exists(SBERT_RESULTS):
        sbert = json.load(open(SBERT_RESULTS, encoding="utf-8"))
        print("\n   Keyword baseline vs SBERT detector (macro-F1):")
        b = det["macro"]["f1"]
        s = sbert.get("macro", {}).get("f1", 0.0)
        print(f"     keyword baseline : {b:.3f}")
        print(f"     SBERT (yours)    : {s:.3f}")
        print(f"     improvement      : +{round(s - b, 3)}  F1")

    nrm = normalization_baseline()
    if nrm:
        print("\n2. NORMALIZATION — majority-class baseline (accuracy to beat)\n")
        print(f"{'Field':<16}{'n':>5}{'majority':>16}{'base_acc':>10}")
        print("-" * 48)
        for field, m in nrm.items():
            print(f"{field:<16}{m['n']:>5}{m['majority_class']:>16}"
                  f"{m['majority_baseline_accuracy']:>10.3f}")
        if os.path.exists(NORM_RESULTS):
            norm_res = json.load(open(NORM_RESULTS, encoding="utf-8"))
            cat = norm_res.get("categorical", {})
            print("\n   Regex normalizer accuracy_when_produced (from norm_eval_results.json),")
            print("   compare each against the majority baseline above:")
            for field in nrm:
                awp = (cat.get(field, {}).get("primary", {}) or {}).get("accuracy_when_produced")
                if awp is not None:
                    print(f"     {field:<16} regex={awp:.3f}  vs  "
                          f"baseline={nrm[field]['majority_baseline_accuracy']:.3f}")

    json.dump({"detection_keyword": det, "normalization_majority": nrm},
              open(OUT_FILE, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
