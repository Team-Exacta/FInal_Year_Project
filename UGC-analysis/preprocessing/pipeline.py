"""Orchestrates the full preprocessing pipeline.

Reads reviews from two sources and groups them by place:

  1. Review_Collection/.../aggregated_cleaned_reviews.json
     - Already-cleaned Kaggle/TripAdvisor corpus (body, place_name, page_url)
  2. dataset/*.json
     - Raw Apify TripAdvisor output (text, placeInfo, user, url)

Both get normalized to a single Apify-like record shape, then run through
language filter -> text cleaning -> deduplication.
"""

import json
import os
import re
from collections import defaultdict

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DATASET_DIR, CLEANED_DATA_DIR, REVIEW_COLLECTION_FILE
from preprocessing.language_filter import filter_english_reviews
from preprocessing.text_cleaner import clean_review
from preprocessing.deduplicator import deduplicate_reviews


def _safe_filename(name):
    """Turn a place name into a safe filename stem."""
    stem = re.sub(r"[^\w\-]", "_", name.strip())
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "unknown"


def _normalize_rc_record(rec):
    """Convert a Review_Collection record to the Apify-like schema used downstream."""
    return {
        "title":         rec.get("title") or "",
        "rating":        rec.get("rating"),
        "travelDate":    None,
        "publishedDate": rec.get("date"),
        "text":          rec.get("body") or "",
        "url":           rec.get("page_url") or "",
        "user": {
            "name":         rec.get("reviewer") or "",
            "username":     rec.get("reviewer") or "",
            "userLocation": {"name": rec.get("reviewer_location") or "", "shortName": "", "id": ""},
            "contributions": {"totalContributions": rec.get("contributions") or 0, "helpfulVotes": 0},
        },
        "ownerResponse": None,
        "placeInfo": {"id": "", "name": rec.get("place_name") or "", "webUrl": ""},
        "_source": "review_collection",
    }


def _tag_source(review, source):
    review["_source"] = source
    return review


def load_all_reviews():
    """Load reviews from both sources, grouped by place.

    Returns dict: place_key -> list[review_dict]
    where place_key is a safe filename stem derived from the place name.
    """
    grouped = defaultdict(list)

    # --- Source 1: Review_Collection aggregated file
    if os.path.exists(REVIEW_COLLECTION_FILE):
        with open(REVIEW_COLLECTION_FILE, "r", encoding="utf-8") as f:
            rc_records = json.load(f)
        for rec in rc_records:
            if not isinstance(rec, dict):
                continue
            pname = (rec.get("place_name") or "").strip()
            if not pname:
                continue
            norm = _normalize_rc_record(rec)
            grouped[_safe_filename(pname)].append(norm)
        print(f"Review_Collection: {len(rc_records)} records across {len(grouped)} places")
    else:
        print(f"(skipped) Review_Collection file not found: {REVIEW_COLLECTION_FILE}")

    # --- Source 2: dataset/*.json (Apify)
    ds_before = sum(len(v) for v in grouped.values())
    if os.path.isdir(DATASET_DIR):
        for filename in sorted(os.listdir(DATASET_DIR)):
            if not filename.endswith(".json") or filename.startswith("_"):
                continue
            with open(os.path.join(DATASET_DIR, filename), "r", encoding="utf-8") as f:
                reviews = json.load(f)
            if not reviews:
                continue
            pname = (reviews[0].get("placeInfo", {}).get("name") or
                     filename.replace(".json", ""))
            key = _safe_filename(pname)
            for r in reviews:
                grouped[key].append(_tag_source(r, "dataset_apify"))
        ds_added = sum(len(v) for v in grouped.values()) - ds_before
        print(f"dataset/ (Apify): +{ds_added} records")

    return grouped


def _exact_body_dedup(reviews):
    """Remove reviews with identical cleaned text (case/space-insensitive)."""
    seen = set()
    kept = []
    removed = 0
    for r in reviews:
        body = (r.get("text_clean") or r.get("text") or "").strip().lower()
        body = " ".join(body.split())
        if not body:
            kept.append(r); continue
        if body in seen:
            removed += 1; continue
        seen.add(body)
        kept.append(r)
    return kept, removed


def preprocess_place(reviews, place_name):
    """Run language filter -> clean -> dedup on a single place's reviews.

    Dedup strategy depends on the source:
      - review_collection: text is already lemmatized + stopwords-stripped.
        Fuzzy TF-IDF over-matches on short cleaned text (two different
        visitors writing short reviews can look 99% similar).  Use exact
        dedup only.
      - dataset_apify: raw text, fuzzy dedup is still useful.
    """
    stats = {"place": place_name, "original_count": len(reviews)}

    english_reviews, filtered_out = filter_english_reviews(reviews)
    stats["non_english_removed"] = len(filtered_out)
    stats["after_lang_filter"] = len(english_reviews)
    if filtered_out:
        langs = {}
        for item in filtered_out:
            langs[item["detected_lang"]] = langs.get(item["detected_lang"], 0) + 1
        stats["filtered_languages"] = langs

    cleaned_reviews = [clean_review(r) for r in english_reviews]
    cleaned_reviews = [r for r in cleaned_reviews if r.get("text_clean", "").strip()]
    stats["empty_after_clean"] = stats["after_lang_filter"] - len(cleaned_reviews)

    rc_reviews  = [r for r in cleaned_reviews if r.get("_source") == "review_collection"]
    raw_reviews = [r for r in cleaned_reviews if r.get("_source") != "review_collection"]

    rc_kept,  rc_removed  = _exact_body_dedup(rc_reviews)
    raw_kept, raw_stats   = deduplicate_reviews(raw_reviews)

    merged_kept, cross_removed = _exact_body_dedup(rc_kept + raw_kept)

    stats["url_duplicates_removed"]   = raw_stats.get("url_duplicates_removed", 0)
    stats["fuzzy_duplicates_removed"] = raw_stats.get("fuzzy_duplicates_removed", 0)
    stats["exact_duplicates_removed"] = rc_removed + cross_removed
    stats["total_removed"]            = (stats["url_duplicates_removed"] +
                                         stats["fuzzy_duplicates_removed"] +
                                         stats["exact_duplicates_removed"])
    stats["final_count"] = len(merged_kept)

    return merged_kept, stats


def run_pipeline():
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    all_reviews = load_all_reviews()
    print(f"\nLoaded {len(all_reviews)} places, "
          f"{sum(len(v) for v in all_reviews.values())} reviews total")

    total_stats = {
        "total_original": 0, "total_final": 0,
        "total_non_english": 0, "total_duplicates": 0,
        "places_processed": 0,
    }
    all_place_stats = []

    for place_key, reviews in all_reviews.items():
        print(f"\nProcessing: {place_key} ({len(reviews)} reviews)")
        cleaned_reviews, stats = preprocess_place(reviews, place_key)
        all_place_stats.append(stats)

        output_path = os.path.join(CLEANED_DATA_DIR, f"{place_key}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_reviews, f, indent=2, ensure_ascii=False)

        total_stats["total_original"]    += stats["original_count"]
        total_stats["total_final"]       += stats["final_count"]
        total_stats["total_non_english"] += stats["non_english_removed"]
        total_stats["total_duplicates"]  += stats["total_removed"]
        total_stats["places_processed"]  += 1

        removed = stats["original_count"] - stats["final_count"]
        print(f"  {stats['original_count']} -> {stats['final_count']} "
              f"(removed {removed}: {stats['non_english_removed']} non-English, "
              f"{stats['total_removed']} duplicates)")

    # Consolidated CSV
    csv_path = os.path.join(CLEANED_DATA_DIR, "_all_reviews.csv")
    all_cleaned = []
    for filename in sorted(os.listdir(CLEANED_DATA_DIR)):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        with open(os.path.join(CLEANED_DATA_DIR, filename), "r", encoding="utf-8") as f:
            reviews = json.load(f)
        for r in reviews:
            if not isinstance(r, dict):
                continue
            place_info = r.get("placeInfo") if isinstance(r.get("placeInfo"), dict) else {}
            user      = r.get("user")      if isinstance(r.get("user"), dict)      else {}
            user_loc  = user.get("userLocation") if isinstance(user.get("userLocation"), dict) else {}
            all_cleaned.append({
                "place_file":     filename,
                "place_name":     place_info.get("name", "") or filename.replace(".json", ""),
                "place_id":       place_info.get("id", ""),
                "source":         r.get("_source", ""),
                "title":          r.get("title", ""),
                "rating":         r.get("rating"),
                "text":           r.get("text", ""),
                "text_clean":     r.get("text_clean", ""),
                "text_display":   r.get("text_display", ""),
                "travelDate":     r.get("travelDate", ""),
                "publishedDate":  r.get("publishedDate", ""),
                "url":            r.get("url", ""),
                "user_name":      user.get("name", ""),
                "user_location":  user_loc.get("name", ""),
                "latitude":       place_info.get("latitude"),
                "longitude":      place_info.get("longitude"),
            })

    pd.DataFrame(all_cleaned).to_csv(csv_path, index=False, encoding="utf-8")

    stats_path = os.path.join(CLEANED_DATA_DIR, "_preprocessing_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"summary": total_stats, "per_place": all_place_stats}, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Places processed: {total_stats['places_processed']}")
    print(f"Total reviews: {total_stats['total_original']} -> {total_stats['total_final']}")
    print(f"Non-English removed: {total_stats['total_non_english']}")
    print(f"Duplicates removed: {total_stats['total_duplicates']}")
    print(f"\nCleaned data saved to: {CLEANED_DATA_DIR}")
    print(f"Consolidated CSV: {csv_path}")
    print(f"Stats: {stats_path}")

    return total_stats
