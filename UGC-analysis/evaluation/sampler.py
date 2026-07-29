"""Stratified sentence sampler for manual gold-label evaluation.

Produces a CSV where each row is one sentence. The human annotator fills in:
  true_best_time, true_crowd_level, true_cost_level  (1 = yes, 0 = no)

Sampling strategy (to capture both FP and FN):
  - 50% from sentences the system tagged (to find false positives)
  - 50% from sentences the system did NOT tag (to find false negatives)
  Per-category breakdown is preserved in the 'system_*' columns so the
  annotator can see what the system decided.
"""

import csv
import json
import os
import random

from config.settings import ANALYSIS_OUTPUT_DIR

_CATEGORIES = ["best_time", "crowd_level", "cost_level"]

EVAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(ANALYSIS_OUTPUT_DIR)), "output", "evaluation"
)
os.makedirs(EVAL_DIR, exist_ok=True)

SAMPLE_CSV = os.path.join(EVAL_DIR, "gold_label_sample.csv")


def build_sample(n_total: int = 300, seed: int = 42, places: list = None) -> str:
    """
    Sample n_total sentences (50% tagged / 50% untagged) and write to CSV.

    Parameters
    ----------
    n_total : int
        Total sentences to sample.
    seed : int
        Random seed for reproducibility.
    places : list, optional
        Restrict sampling to these place keys. Samples all 180 if None.

    Returns
    -------
    str
        Path to the generated CSV file.
    """
    random.seed(seed)

    place_files = sorted(
        f[:-5] for f in os.listdir(ANALYSIS_OUTPUT_DIR)
        if f.endswith(".json") and not f.startswith("_")
    )
    if places:
        place_files = [p for p in place_files if p in places]

    tagged_pool   = []  # (place, sentence, {cat: True/False})
    untagged_pool = []

    for place_key in place_files:
        src = os.path.join(ANALYSIS_OUTPUT_DIR, f"{place_key}.json")
        with open(src, encoding="utf-8") as f:
            reviews = json.load(f)

        for review in reviews:
            tags  = review.get("analysis_tags", {})
            sents = review.get("analysis_sentences", {})

            # Collect all sentences from this review
            seen = set()

            # Tagged sentences (system said yes for at least one category)
            for cat in _CATEGORIES:
                if tags.get(cat):
                    for sent in sents.get(cat, []):
                        if sent not in seen:
                            seen.add(sent)
                            row = {
                                "place":              place_key,
                                "sentence":           sent,
                                "system_best_time":   int(bool(tags.get("best_time"))),
                                "system_crowd_level": int(bool(tags.get("crowd_level"))),
                                "system_cost_level":  int(bool(tags.get("cost_level"))),
                                # Blank columns for human annotator to fill
                                "true_best_time":     "",
                                "true_crowd_level":   "",
                                "true_cost_level":    "",
                                "notes":              "",
                            }
                            tagged_pool.append(row)

            # Untagged reviews — pick the first sentence as a negative sample
            if not any(tags.get(c) for c in _CATEGORIES):
                text = review.get("text_clean", "") or review.get("text", "")
                # Take first meaningful sentence (>20 chars)
                for sent in text.split("."):
                    sent = sent.strip()
                    if len(sent) > 20:
                        row = {
                            "place":              place_key,
                            "sentence":           sent,
                            "system_best_time":   0,
                            "system_crowd_level": 0,
                            "system_cost_level":  0,
                            "true_best_time":     "",
                            "true_crowd_level":   "",
                            "true_cost_level":    "",
                            "notes":              "",
                        }
                        untagged_pool.append(row)
                        break

    # Stratified sample: 50% tagged / 50% untagged
    n_tagged   = min(n_total // 2, len(tagged_pool))
    n_untagged = min(n_total - n_tagged, len(untagged_pool))

    sampled = (
        random.sample(tagged_pool, n_tagged) +
        random.sample(untagged_pool, n_untagged)
    )
    random.shuffle(sampled)

    # Write CSV
    fieldnames = [
        "place", "sentence",
        "system_best_time", "system_crowd_level", "system_cost_level",
        "true_best_time", "true_crowd_level", "true_cost_level",
        "notes",
    ]
    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    print(f"Sampled {len(sampled)} sentences ({n_tagged} tagged / {n_untagged} untagged)")
    print(f"CSV written to: {SAMPLE_CSV}")
    print()
    print("Next step: open the CSV, fill in true_best_time / true_crowd_level / true_cost_level")
    print("           (1 = yes, 0 = no) then run: python run_eval_score.py")

    return SAMPLE_CSV
