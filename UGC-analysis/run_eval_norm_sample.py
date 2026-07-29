"""Build the NORMALIZATION gold layer (Part B, step 1).

The existing gold set (output/evaluation/gold_label_sample.csv) only has BINARY
category labels (is this sentence about best_time / crowd / cost — yes/no). It
never checks whether the regex normalizer maps the text to the CORRECT canonical
value. This script produces a second gold sheet for exactly that.

For every gold sentence it:
  * carries over the binary true_* labels (so we know which sentences are
    positives — only those need a normalized-value label),
  * runs the current extractor + normalizer and records the SYSTEM's predicted
    canonical values (shown to the annotator for reference, and used later by the
    scorer as the "system" side of the comparison),
  * adds BLANK true_* value columns for a human to fill:
      best_time : true_time_of_day, true_season, true_day_type
      crowd     : true_crowd_ordinal   (1-5)
      cost      : true_cost_band (FREE/LOW/MODERATE/HIGH/VERY_HIGH), true_amount_lkr

Re-running is safe: any human labels already in norm_gold_sample.csv are preserved
(only the system_* columns are refreshed).

Output: output/evaluation/norm_gold_sample.csv
Run   : python run_eval_norm_sample.py
Next  : label it (streamlit run labeling_ui.py), then python run_eval_norm_score.py
"""

import csv
import os

from extraction.extractor import SpanExtractor
from extraction.normalizer import normalize_best_time, normalize_crowd, normalize_cost
from config.settings import SPACY_MODEL, OUTPUT_DIR

EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")
GOLD_CSV = os.path.join(EVAL_DIR, "gold_label_sample.csv")
OUT_CSV = os.path.join(EVAL_DIR, "norm_gold_sample.csv")

CATEGORIES = ["best_time", "crowd_level", "cost_level"]

# Human-label columns (blank for the annotator). Kept separate from the binary
# true_best_time / true_crowd_level / true_cost_level columns carried over.
TRUE_VALUE_COLS = [
    "true_time_of_day", "true_season", "true_day_type",   # best_time
    "true_crowd_ordinal",                                  # crowd
    "true_cost_band", "true_amount_lkr",                   # cost
]

FIELDNAMES = (
    ["place", "sentence",
     "true_best_time", "true_crowd_level", "true_cost_level",
     # binary detection flags (carried over — let the scorer condition on
     # "correctly detected" as well as on true positives)
     "system_best_time", "system_crowd_level", "system_cost_level",
     # system predictions (reference + scored later)
     "system_time_of_day", "system_season", "system_day_type",
     "system_crowd_ordinal", "system_crowd_label",
     "system_cost_band", "system_amount_lkr"]
    + TRUE_VALUE_COLS
    + ["notes"]
)


def _system_norm(ex: SpanExtractor, sentence: str) -> dict:
    """Run the current extractor + normalizer on one sentence."""
    bt = normalize_best_time(ex.extract_best_time([sentence]), [sentence])
    cl = normalize_crowd(ex.extract_crowd([sentence]), [sentence])
    co = normalize_cost(ex.extract_cost([sentence]), [sentence])
    day_type = bt.get("recommend_day") or (bt.get("avoid") or [None])[0]
    return {
        "system_time_of_day":  bt.get("time_of_day") or "",
        "system_season":       bt.get("season") or "",
        "system_day_type":     day_type or "",
        "system_crowd_ordinal": cl.get("crowd_level") if cl.get("crowd_level") is not None else "",
        "system_crowd_label":  cl.get("crowd_label") or "",
        "system_cost_band":    co.get("cost_level") or "",
        "system_amount_lkr":   co.get("amount_lkr") if co.get("amount_lkr") is not None else "",
    }


def _load_existing_labels() -> dict:
    """Preserve any human value-labels already entered (keyed by sentence)."""
    if not os.path.exists(OUT_CSV):
        return {}
    saved = {}
    with open(OUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            saved[row["sentence"]] = {c: row.get(c, "") for c in TRUE_VALUE_COLS + ["notes"]}
    return saved


def main():
    if not os.path.exists(GOLD_CSV):
        print(f"Gold set not found: {GOLD_CSV}. Run run_eval_sample.py first.")
        return

    with open(GOLD_CSV, encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f))

    existing = _load_existing_labels()
    if existing:
        print(f"Preserving human labels for {len(existing)} sentences already in {OUT_CSV}.")

    print(f"Running extractor + normalizer on {len(gold_rows)} sentences...")
    ex = SpanExtractor(SPACY_MODEL)

    out_rows = []
    n_pos = {c: 0 for c in CATEGORIES}
    for i, g in enumerate(gold_rows, 1):
        sentence = g["sentence"]
        row = {k: "" for k in FIELDNAMES}
        row["place"] = g.get("place", "")
        row["sentence"] = sentence
        for c in CATEGORIES:
            row[f"true_{c}"] = g.get(f"true_{c}", "")
            row[f"system_{c}"] = g.get(f"system_{c}", "")
            if str(g.get(f"true_{c}", "")).strip() == "1":
                n_pos[c] += 1
        row.update(_system_norm(ex, sentence))
        # restore any previously-entered human value labels
        prev = existing.get(sentence)
        if prev:
            for c in TRUE_VALUE_COLS + ["notes"]:
                row[c] = prev.get(c, "")
        out_rows.append(row)
        if i % 100 == 0:
            print(f"  {i}/{len(gold_rows)}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows -> {OUT_CSV}")
    print("Positives needing a value label — "
          + ", ".join(f"{c}: {n_pos[c]}" for c in CATEGORIES))
    print("\nNext: label the true_* value columns (streamlit run labeling_ui.py),")
    print("      then run: python run_eval_norm_score.py")


if __name__ == "__main__":
    main()
