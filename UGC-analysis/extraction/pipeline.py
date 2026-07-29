"""Extraction + normalization pipeline.

Loads output/analysis/*.json → runs SpanExtractor + normalizer on every
tagged review → writes enriched JSONs to output/extraction/ and builds
_normalized_profiles.json + _all_spans.csv.
"""

import csv
import json
import os
import sys

from config.settings import (
    ANALYSIS_OUTPUT_DIR,
    EXTRACTION_OUTPUT_DIR,
    EXTRACTION_PROGRESS_FILE,
    SPACY_MODEL,
)
from extraction.extractor import SpanExtractor
from extraction.normalizer import normalize_best_time, normalize_crowd, normalize_cost  # noqa: F401

_CATEGORIES = ["best_time", "crowd_level", "cost_level"]

_NORMALIZERS = {
    "best_time":   normalize_best_time,
    "crowd_level": normalize_crowd,
    "cost_level":  normalize_cost,
}


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _load_progress() -> set:
    if os.path.exists(EXTRACTION_PROGRESS_FILE):
        with open(EXTRACTION_PROGRESS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    return set()


def _save_progress(completed: set):
    with open(EXTRACTION_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


# ---------------------------------------------------------------------------
# Core per-place processing
# ---------------------------------------------------------------------------

def _process_place(place_key: str, extractor: SpanExtractor) -> list:
    """Return list of enriched review dicts for one place."""
    src = os.path.join(ANALYSIS_OUTPUT_DIR, f"{place_key}.json")
    with open(src, encoding="utf-8") as f:
        reviews = json.load(f)

    enriched = []
    for review in reviews:
        tags = review.get("analysis_tags", {})
        sents = review.get("analysis_sentences", {})

        _extract_fn = {
            "best_time":   extractor.extract_best_time,
            "crowd_level": extractor.extract_crowd,
            "cost_level":  extractor.extract_cost,
        }

        spans = {}
        normalized = {}
        for cat in _CATEGORIES:
            if tags.get(cat):
                raw = sents.get(cat, [])
                spans[cat] = _extract_fn[cat](raw)
                # All normalizers now take the raw sentences too, so negation and
                # local/foreigner context survive (the span alone drops them).
                normalized[cat] = _NORMALIZERS[cat](spans[cat], raw)
            else:
                spans[cat] = {}
                normalized[cat] = {}

        review["extracted_spans"] = spans
        review["normalized"] = normalized
        enriched.append(review)

    return enriched


def _aggregate_place(place_key: str, enriched: list) -> dict:
    """Build a simple frequency-based normalized profile for one place."""
    from collections import Counter

    tod_counter    = Counter()
    season_counter = Counter()
    month_counter  = Counter()
    avoid_counter  = Counter()
    crowd_scores   = []
    crowd_when_c   = Counter()
    cost_levels    = []
    amounts        = []
    fee_types      = Counter()

    for r in enriched:
        norm = r.get("normalized", {})

        bt = norm.get("best_time", {})
        if bt:
            if bt.get("time_of_day"):
                tod_counter[bt["time_of_day"]] += 1
            if bt.get("season"):
                season_counter[bt["season"]] += 1
            for m in bt.get("months", []):
                month_counter[m] += 1
            for a in bt.get("avoid", []):
                avoid_counter[a] += 1
            if bt.get("recommend_day"):
                avoid_counter[f"RECOMMEND_{bt['recommend_day']}"] += 1

        cl = norm.get("crowd_level", {})
        if cl and cl.get("crowd_level") is not None:
            crowd_scores.append(cl["crowd_level"])
            for w in cl.get("crowd_when", []):
                crowd_when_c[w] += 1

        co = norm.get("cost_level", {})
        if co and co.get("cost_level"):
            cost_levels.append(co["cost_level"])
        if co and co.get("amount_lkr") is not None:
            amounts.append(co["amount_lkr"])
        if co and co.get("fee_type"):
            fee_types[co["fee_type"]] += 1

    def _top(counter, n=1):
        common = counter.most_common(n)
        return [k for k, _ in common]

    tagged_counts = {
        cat: sum(1 for r in enriched if r.get("analysis_tags", {}).get(cat))
        for cat in _CATEGORIES
    }

    profile = {
        "place_key":    place_key,
        "review_count": len(enriched),
        "tagged_counts": tagged_counts,
        "best_time": {
            "time_of_day":   (_top(tod_counter) or [None])[0],
            "season":        (_top(season_counter) or [None])[0],
            "months":        _top(month_counter, 12),
            "avoid":         _top(avoid_counter, 3),
            "tod_dist":      dict(tod_counter),
            "season_dist":   dict(season_counter),
        },
        "crowd": {
            "avg_level":  round(sum(crowd_scores) / len(crowd_scores), 2) if crowd_scores else None,
            "peak_level": max(crowd_scores) if crowd_scores else None,
            "crowd_when": _top(crowd_when_c, 3),
        },
        "cost": {
            "dominant_level": Counter(cost_levels).most_common(1)[0][0] if cost_levels else None,
            "median_lkr":     sorted(amounts)[len(amounts) // 2] if amounts else None,
            "fee_type":       (_top(fee_types) or [None])[0],
            "cost_dist":      dict(Counter(cost_levels)),
        },
    }
    return profile


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _write_spans_csv(all_rows: list):
    path = os.path.join(EXTRACTION_OUTPUT_DIR, "_all_spans.csv")
    fieldnames = [
        "place_key", "category",
        # best_time
        "time_spans", "action", "avoid",
        "time_of_day", "season", "months", "recommend_day",
        # crowd
        "crowd_spans", "crowd_when",
        "crowd_level", "crowd_label",
        # cost
        "price_spans", "evaluation", "fee_type",
        "cost_level", "amount_lkr",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def _build_csv_rows(place_key: str, enriched: list) -> list:
    rows = []
    for r in enriched:
        spans = r.get("extracted_spans", {})
        norm  = r.get("normalized", {})
        for cat in _CATEGORIES:
            if not r.get("analysis_tags", {}).get(cat):
                continue
            sp = spans.get(cat, {})
            nm = norm.get(cat, {})
            row = {"place_key": place_key, "category": cat}
            if cat == "best_time":
                row["time_spans"]    = "|".join(sp.get("time_spans", []))
                row["action"]        = sp.get("action") or ""
                row["avoid"]         = "|".join(sp.get("avoid", []))
                row["time_of_day"]   = nm.get("time_of_day") or ""
                row["season"]        = nm.get("season") or ""
                row["months"]        = "|".join(nm.get("months", []))
                row["recommend_day"] = nm.get("recommend_day") or ""
            elif cat == "crowd_level":
                row["crowd_spans"] = "|".join(sp.get("crowd_spans", []))
                row["crowd_when"]  = "|".join(sp.get("crowd_when", []))
                row["crowd_level"] = nm.get("crowd_level") or ""
                row["crowd_label"] = nm.get("crowd_label") or ""
            elif cat == "cost_level":
                row["price_spans"] = "|".join(sp.get("price_spans", []))
                row["evaluation"]  = sp.get("evaluation") or ""
                row["fee_type"]    = sp.get("fee_type") or ""
                row["cost_level"]  = nm.get("cost_level") or ""
                row["amount_lkr"]  = nm.get("amount_lkr") if nm.get("amount_lkr") is not None else ""
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_pipeline(
    resume: bool = True,
    places_filter: list = None,
    verbose: bool = True,
):
    extractor = SpanExtractor(SPACY_MODEL)

    # Discover place keys from analysis output
    all_files = sorted(
        f[:-5] for f in os.listdir(ANALYSIS_OUTPUT_DIR)
        if f.endswith(".json") and not f.startswith("_")
    )
    if places_filter:
        all_files = [p for p in all_files if p in places_filter]

    completed = _load_progress() if resume else set()

    profiles  = {}
    csv_rows  = []

    for idx, place_key in enumerate(all_files, 1):
        if place_key in completed:
            if verbose:
                print(f"[{idx}/{len(all_files)}] {place_key} — skipped (done)")
            continue

        if verbose:
            print(f"[{idx}/{len(all_files)}] {place_key} ...", end=" ", flush=True)

        enriched = _process_place(place_key, extractor)

        # Save per-place enriched JSON
        out_path = os.path.join(EXTRACTION_OUTPUT_DIR, f"{place_key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        profile = _aggregate_place(place_key, enriched)
        profiles[place_key] = profile
        csv_rows.extend(_build_csv_rows(place_key, enriched))

        completed.add(place_key)
        _save_progress(completed)

        tagged = profile["tagged_counts"]
        if verbose:
            print(
                f"done  "
                f"bt={tagged['best_time']} "
                f"cl={tagged['crowd_level']} "
                f"co={tagged['cost_level']}"
            )

    # Merge profiles from previously completed places
    profiles_path = os.path.join(EXTRACTION_OUTPUT_DIR, "_normalized_profiles.json")
    if resume and os.path.exists(profiles_path):
        with open(profiles_path, encoding="utf-8") as f:
            existing = json.load(f)
        for k, v in existing.items():
            if k not in profiles:
                profiles[k] = v

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    # Build full CSV (merge old rows if resuming)
    _write_spans_csv(csv_rows)

    if verbose:
        n_tagged = sum(
            1 for p in profiles.values()
            for cat in _CATEGORIES
            if p["tagged_counts"].get(cat, 0) > 0
        )
        print(f"\nDone. {len(profiles)} places processed.")
        print(f"Profiles saved -> {profiles_path}")
        print(f"Spans CSV     -> {os.path.join(EXTRACTION_OUTPUT_DIR, '_all_spans.csv')}")
