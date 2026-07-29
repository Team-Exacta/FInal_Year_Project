"""Orchestrates the review analysis tagging pipeline (best_time / crowd_level / cost_level)."""

import json
import os
import time
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CLEANED_DATA_DIR,
    ANALYSIS_OUTPUT_DIR,
    ANALYSIS_MODEL,
    ANALYSIS_SCORE_THRESHOLD,
    ANALYSIS_SCORE_THRESHOLDS,
    ANALYSIS_BATCH_SIZE,
    ANALYSIS_PROGRESS_FILE,
)
from analysis.detector import ReviewAnalysisDetector, CATEGORIES  # noqa: F401


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def load_progress():
    if os.path.exists(ANALYSIS_PROGRESS_FILE):
        with open(ANALYSIS_PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress):
    progress["last_updated"] = datetime.now().isoformat()
    with open(ANALYSIS_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Place processing
# ---------------------------------------------------------------------------

def load_place_reviews(json_path: str) -> list:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [r for r in data if isinstance(r, dict)]
    except Exception as e:
        print(f"  Warning: could not load {json_path}: {e}")
        return []


def process_place(place_key: str, reviews: list, detector: ReviewAnalysisDetector):
    """Tag all reviews for a place. Returns (enriched_reviews, stats_dict)."""
    enriched = detector.tag_reviews_batch(reviews)

    stats = {"place": place_key, "total_reviews": len(enriched)}
    for cat in CATEGORIES:
        count = sum(1 for r in enriched if r.get("analysis_tags", {}).get(cat))
        scores = [r.get("analysis_scores", {}).get(cat, 0.0) for r in enriched]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
        stats[f"{cat}_count"] = count
        stats[f"{cat}_rate"] = round(count / len(enriched), 4) if enriched else 0.0
        stats[f"{cat}_avg_score"] = avg_score

    return enriched, stats


def save_place_output(enriched_reviews: list, place_key: str):
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, f"{place_key}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_reviews, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Consolidated outputs
# ---------------------------------------------------------------------------

def build_consolidated_csvs():
    """Build _all_tagged.csv and one CSV per category from output/analysis/*.json."""
    all_rows = []

    for filename in sorted(os.listdir(ANALYSIS_OUTPUT_DIR)):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        filepath = os.path.join(ANALYSIS_OUTPUT_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        for r in reviews:
            if not isinstance(r, dict):
                continue
            place_info = r.get("placeInfo") if isinstance(r.get("placeInfo"), dict) else {}
            tags   = r.get("analysis_tags", {})
            scores = r.get("analysis_scores", {})
            sents  = r.get("analysis_sentences", {})

            row = {
                "place_file":        filename,
                "place_name":        place_info.get("name", "") or filename.replace(".json", ""),
                "rating":            r.get("rating"),
                "publishedDate":     r.get("publishedDate", ""),
                "url":               r.get("url", ""),
                "source":            r.get("_source", ""),
                "title":             r.get("title", ""),
                "text_display":      r.get("text_display", "") or r.get("text", ""),
            }
            for cat in CATEGORIES:
                row[cat]               = tags.get(cat, False)
                row[f"{cat}_score"]    = scores.get(cat, 0.0)
                row[f"{cat}_sentences"] = json.dumps(sents.get(cat, []))
            all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(ANALYSIS_OUTPUT_DIR, "_all_tagged.csv"),
                  index=False, encoding="utf-8")

    for cat in CATEGORIES:
        df_cat = df_all[df_all[cat] == True].copy()  # noqa: E712
        cols = ["place_file", "place_name", "rating", "publishedDate", "url",
                "source", "title", "text_display", f"{cat}_score", f"{cat}_sentences"]
        df_cat[cols].to_csv(
            os.path.join(ANALYSIS_OUTPUT_DIR, f"_{cat}.csv"),
            index=False, encoding="utf-8",
        )
        print(f"  {cat}: {len(df_cat)} tagged reviews -> _{cat}.csv")

    print(f"  All reviews: {len(df_all)} total -> _all_tagged.csv")


def build_stats_json(all_place_stats: list, model_name: str, threshold):
    summary = {
        "total_reviews": sum(s["total_reviews"] for s in all_place_stats),
        "places_processed": len(all_place_stats),
    }
    for cat in CATEGORIES:
        summary[f"{cat}_count"] = sum(s.get(f"{cat}_count", 0) for s in all_place_stats)
        total = summary["total_reviews"]
        summary[f"{cat}_rate"] = round(summary[f"{cat}_count"] / total, 4) if total else 0.0

    stats = {
        "model_used": model_name,
        "threshold": threshold,
        "run_date": datetime.now().isoformat(),
        "summary": summary,
        "per_place": all_place_stats,
    }
    with open(os.path.join(ANALYSIS_OUTPUT_DIR, "_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _collect_stats_from_files() -> list:
    """Recompute per-place stats by reading every output JSON.

    Called at the end of run_pipeline (including resume runs) so the final
    _stats.json always reflects every place in the output folder, not just
    the ones processed in the current session.
    """
    stats = []
    for filename in sorted(os.listdir(ANALYSIS_OUTPUT_DIR)):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        place_key = filename.replace(".json", "")
        try:
            with open(os.path.join(ANALYSIS_OUTPUT_DIR, filename),
                      encoding="utf-8") as f:
                reviews = json.load(f)
        except Exception:
            continue
        if not reviews:
            continue
        n = len(reviews)
        entry = {"place": place_key, "total_reviews": n}
        for cat in CATEGORIES:
            count = sum(1 for r in reviews
                        if isinstance(r, dict) and
                        r.get("analysis_tags", {}).get(cat))
            entry[f"{cat}_count"] = count
            entry[f"{cat}_rate"]  = round(count / n, 4) if n else 0.0
        stats.append(entry)
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(resume=True, places_filter=None, threshold_override=None):
    # threshold_override (scalar from CLI) applies to all categories.
    # Otherwise use the per-category dict from settings.
    threshold  = (threshold_override if threshold_override is not None
                  else ANALYSIS_SCORE_THRESHOLDS)
    model_name = ANALYSIS_MODEL

    print("=" * 60)
    print("REVIEW ANALYSIS PIPELINE")
    print(f"Model     : {model_name}")
    print(f"Threshold : {threshold}")
    print("=" * 60)

    detector = ReviewAnalysisDetector(
        model_name=model_name,
        score_threshold=threshold,
        batch_size=ANALYSIS_BATCH_SIZE,
    )

    place_files = sorted(
        f for f in os.listdir(CLEANED_DATA_DIR)
        if f.endswith(".json") and not f.startswith("_")
    )

    progress = load_progress() if resume else {"completed": [], "failed": []}
    done_set = set(progress["completed"])

    pending = [f for f in place_files
               if f.replace(".json", "") not in done_set]
    if places_filter:
        pending = [f for f in pending
                   if f.replace(".json", "") in places_filter]

    total = len(place_files)
    skipped = len(done_set)
    print(f"\nPlaces: {total} total | {skipped} already done | {len(pending)} to process\n")

    all_place_stats = []

    for i, filename in enumerate(pending, start=skipped + 1):
        place_key = filename.replace(".json", "")
        json_path = os.path.join(CLEANED_DATA_DIR, filename)
        reviews   = load_place_reviews(json_path)

        if not reviews:
            progress["failed"].append(place_key)
            save_progress(progress)
            continue

        t0 = time.time()
        enriched, stats = process_place(place_key, reviews, detector)
        elapsed = time.time() - t0

        save_place_output(enriched, place_key)
        all_place_stats.append(stats)
        progress["completed"].append(place_key)
        save_progress(progress)

        bt  = stats.get("best_time_count", 0)
        cl  = stats.get("crowd_level_count", 0)
        co  = stats.get("cost_level_count", 0)
        n   = stats["total_reviews"]
        print(
            f"[{i:3d}/{total}] {place_key:<30s}: {n} reviews | "
            f"best_time: {bt} ({bt/n*100:.1f}%) | "
            f"crowd: {cl} ({cl/n*100:.1f}%) | "
            f"cost: {co} ({co/n*100:.1f}%) | "
            f"{elapsed:.1f}s"
        )

    print("\nBuilding consolidated CSVs...")
    build_consolidated_csvs()

    print("Writing stats...")
    # Recompute stats from every output file so resume produces correct totals.
    all_place_stats = _collect_stats_from_files()
    build_stats_json(all_place_stats, model_name, threshold)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Output: {ANALYSIS_OUTPUT_DIR}")
    print("=" * 60)

    return {"places_processed": len(all_place_stats)}
