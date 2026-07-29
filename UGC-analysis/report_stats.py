"""Corpus + suggestion statistics.

Answers: how many reviews carry a suggestion in each category (best_time /
crowd / cost), and how many reviews each place has. Reads the Step-3 analysis
output (output/analysis/*.json), which already has analysis_tags per review.

Run:  python report_stats.py
Out:  console summary + output/analysis/_review_stats.csv (per place)
"""

import csv
import json
import os

from config.settings import ANALYSIS_OUTPUT_DIR

CATS = ["best_time", "crowd_level", "cost_level"]
OUT_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, "_review_stats.csv")


def main():
    files = sorted(f for f in os.listdir(ANALYSIS_OUTPUT_DIR)
                   if f.endswith(".json") and not f.startswith("_"))
    rows = []
    total_reviews = 0
    cat_totals = {c: 0 for c in CATS}
    any_total = 0

    for fname in files:
        with open(os.path.join(ANALYSIS_OUTPUT_DIR, fname), encoding="utf-8") as f:
            reviews = json.load(f)
        n = len(reviews)
        total_reviews += n
        cc = {c: 0 for c in CATS}
        any_c = 0
        for r in reviews:
            tags = r.get("analysis_tags", {})
            hit = False
            for c in CATS:
                if tags.get(c):
                    cc[c] += 1
                    cat_totals[c] += 1
                    hit = True
            if hit:
                any_c += 1
        any_total += any_c
        rows.append({
            "place": fname[:-5], "reviews": n,
            "best_time": cc["best_time"], "crowd_level": cc["crowd_level"],
            "cost_level": cc["cost_level"], "any_suggestion": any_c,
        })

    rows.sort(key=lambda r: r["reviews"], reverse=True)
    n_places = len(rows)

    def pct(x):
        return f"{100*x/total_reviews:4.1f}%" if total_reviews else "  0%"

    print("=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"Places           : {n_places}")
    print(f"Total reviews    : {total_reviews:,}")
    print(f"Avg reviews/place: {total_reviews/n_places:.0f}" if n_places else "")
    print()
    print("Reviews carrying a suggestion (of total reviews):")
    for c in CATS:
        print(f"  {c:<12} {cat_totals[c]:>7,}   {pct(cat_totals[c])}")
    print(f"  {'ANY':<12} {any_total:>7,}   {pct(any_total)}")
    print()
    print(f"Top 15 places by review count:")
    print(f"  {'place':<40}{'rev':>6}{'best':>6}{'crowd':>6}{'cost':>6}")
    for r in rows[:15]:
        print(f"  {r['place'][:40]:<40}{r['reviews']:>6}{r['best_time']:>6}"
              f"{r['crowd_level']:>6}{r['cost_level']:>6}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["place", "reviews", "best_time",
                                          "crowd_level", "cost_level", "any_suggestion"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-place table ({n_places} rows) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
