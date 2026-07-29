"""Collect TripAdvisor reviews via Apify cloud scraper.

Actor used: maxcopell/tripadvisor-reviews
  - Returns one item per review (flat, not nested)
  - Each item includes a placeInfo field with full place metadata
  - English filter via language="en"

Multi-account support:
  When one account exhausts its credits ("Monthly usage hard limit exceeded"),
  the script pauses and asks for the next account's API token, then
  continues exactly where it left off — no re-scraping of completed places.

Usage:
  python run_scraper.py --apify              # scrape all remaining places
  python run_scraper.py --apify --limit 5    # test with 5 places first
"""

import json
import os
import re
import time
from datetime import datetime

from apify_client import ApifyClient

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DATASET_DIR, SCRAPE_PROGRESS_FILE,
    PLACES_TO_SCRAPE_FILE, MAX_REVIEWS_PER_PLACE,
)

ACTOR_ID  = "maxcopell/tripadvisor-reviews"
BATCH_SIZE = 10   # places per Apify run (keeps runs short & cheap)

QUOTA_ERRORS = (
    "monthly usage hard limit exceeded",
    "hard limit exceeded",
    "usage limit",
    "upgrade your subscription",
    "payment required",
)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def load_progress():
    if os.path.exists(SCRAPE_PROGRESS_FILE):
        with open(SCRAPE_PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress):
    os.makedirs(os.path.dirname(SCRAPE_PROGRESS_FILE), exist_ok=True)
    with open(SCRAPE_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def scan_dataset_for_ids():
    """Return set of place IDs already saved in the dataset folder."""
    ids = set()
    if not os.path.exists(DATASET_DIR):
        return ids
    for filename in os.listdir(DATASET_DIR):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        try:
            with open(os.path.join(DATASET_DIR, filename), encoding="utf-8") as f:
                reviews = json.load(f)
            if reviews and isinstance(reviews[0], dict):
                pid = str(reviews[0].get("placeInfo", {}).get("id", "")).strip()
                if pid:
                    ids.add(pid)
        except Exception:
            pass
    return ids


# ---------------------------------------------------------------------------
# Token management (multi-account)
# ---------------------------------------------------------------------------

def prompt_for_token(reason=""):
    """Ask the user for an Apify API token. Returns the token string."""
    print("\n" + "="*60)
    if reason:
        print(f"  {reason}")
    print("  Sign up free: https://apify.com")
    print("  Token location: Settings -> Integrations -> API token")
    print("="*60)
    token = input("  Paste Apify API token: ").strip()
    return token


def get_initial_token():
    """Get token from env var, .env file, or prompt."""
    # 1. Env var
    token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if token:
        return token

    # 2. .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.upper().startswith("APIFY_API_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if token:
                        return token

    # 3. Prompt
    token = prompt_for_token("No Apify API token found in environment.")
    return token


def is_quota_error(exc):
    msg = str(exc).lower()
    return any(kw in msg for kw in QUOTA_ERRORS)


# ---------------------------------------------------------------------------
# Apify run logic
# ---------------------------------------------------------------------------

def run_apify_batch(client, urls_batch):
    """Run one Apify actor call for a batch of place URLs.

    Returns list of raw review items (one item = one review).
    Raises the original exception on failure (caller decides whether it's quota).
    """
    print(f"\n  Sending {len(urls_batch)} URLs to Apify...")

    run_input = {
        "startUrls": [{"url": u} for u in urls_batch],
        "maxItemsPerQuery": MAX_REVIEWS_PER_PLACE,
        "language": "en",
        "scrapeReviewerInfo": True,
    }

    run = client.actor(ACTOR_ID).call(run_input=run_input, timeout_secs=600)

    items = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        items.append(item)

    print(f"  Apify returned {len(items)} items")
    return items


def process_apify_results(items, places_by_url):
    """Group flat review items by place, save each group as a JSON file.

    Returns list of place IDs that were successfully saved.
    """
    saved_ids = []

    # Group by place ID
    reviews_by_place = {}
    for item in items:
        place_info = item.get("placeInfo") or {}
        pid = str(place_info.get("id") or item.get("locationId") or "").strip()
        if not pid:
            continue
        if pid not in reviews_by_place:
            reviews_by_place[pid] = {"place_info": place_info, "reviews": []}
        reviews_by_place[pid]["reviews"].append(item)

    for pid, data in reviews_by_place.items():
        place_info = data["place_info"]
        raw_reviews  = data["reviews"]

        # Prefer name from our places list; fall back to actor data
        place_url = place_info.get("webUrl", "")
        matched = None
        for purl, place in places_by_url.items():
            if place.get("id") == pid or \
               purl.split("?")[0].rstrip("/") == place_url.split("?")[0].rstrip("/"):
                matched = place
                break
        place_name = (matched or {}).get("name") or place_info.get("name") or pid

        # Build review list (items are already in our schema format)
        reviews = []
        for item in raw_reviews:
            if not item.get("text"):
                continue
            reviews.append({
                "title":         item.get("title") or "",
                "rating":        item.get("rating"),
                "travelDate":    item.get("travelDate"),
                "publishedDate": item.get("publishedDate"),
                "text":          item.get("text") or "",
                "url":           item.get("url") or "",
                "user":          item.get("user"),
                "ownerResponse": item.get("ownerResponse"),
                "placeInfo":     place_info,
            })

        if not reviews:
            print(f"  0 valid reviews for: {place_name}")
            continue

        filename  = re.sub(r"[^\w\-_]", "", place_name.replace(" ", "_").replace("/", "_"))
        filename += ".json"
        filepath  = os.path.join(DATASET_DIR, filename)
        os.makedirs(DATASET_DIR, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(reviews)} reviews -> {filename}")
        saved_ids.append(pid)

    return saved_ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_with_apify(limit=None, places_file=None):
    """Scrape reviews for all pending places, rotating Apify accounts on quota hit.

    Args:
      limit:       only scrape the first N pending places
      places_file: path to a JSON file of places (defaults to PLACES_TO_SCRAPE_FILE)
    """
    places_path = places_file or PLACES_TO_SCRAPE_FILE
    with open(places_path, encoding="utf-8") as f:
        all_places = json.load(f)

    # Determine pending places (not yet in dataset or progress)
    progress    = load_progress()
    dataset_ids = scan_dataset_for_ids()
    done_ids    = set(progress["completed"]) | dataset_ids

    pending = [p for p in all_places if p["id"] not in done_ids]
    if limit:
        pending = pending[:limit]

    if not pending:
        print("All places already collected.")
        return

    total_places = len(all_places)
    print(f"\nDataset status: {len(done_ids)}/{total_places} places already collected")
    print(f"Places remaining: {len(pending)}")
    print(f"Reviews per place: {MAX_REVIEWS_PER_PLACE}")

    token = get_initial_token()
    if not token:
        print("No API token — cannot continue.")
        return

    client        = ApifyClient(token)
    places_by_url = {p["url"]: p for p in pending}

    total_saved = 0
    batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    batch_idx = 0   # track manually so we can retry on token swap

    while batch_idx < len(batches):
        batch = batches[batch_idx]

        print(f"\n{'='*50}")
        print(f"Batch {batch_idx+1}/{len(batches)}  ({len(batch)} places)")
        print("Places:", ", ".join(p["name"] for p in batch))

        try:
            urls   = [p["url"] for p in batch]
            items  = run_apify_batch(client, urls)
            saved  = process_apify_results(items, places_by_url)

            for pid in saved:
                if pid not in progress["completed"]:
                    progress["completed"].append(pid)

            # Any place in the batch with no saved result → mark failed for visibility
            for p in batch:
                if p["id"] not in saved and p["id"] not in progress["failed"]:
                    progress["failed"].append(p["id"])

            total_saved += len(saved)
            save_progress(progress)
            batch_idx += 1   # advance only on success

            if batch_idx < len(batches):
                time.sleep(3)

        except Exception as e:
            if is_quota_error(e):
                # -------------------------------------------------------
                # Credits exhausted — ask for next account token
                # -------------------------------------------------------
                completed_so_far = len(done_ids) + total_saved
                remaining        = total_places - completed_so_far
                print(f"\n{'!'*50}")
                print(f"  ACCOUNT CREDITS EXHAUSTED")
                print(f"  Saved so far this session: {total_saved} places")
                print(f"  Total in dataset: {completed_so_far}/{total_places}")
                print(f"  Still remaining:  {remaining} places")
                print(f"{'!'*50}")

                new_token = prompt_for_token(
                    "Provide another Apify account token to continue."
                )
                if not new_token:
                    print("No token provided. Stopping here.")
                    break

                client = ApifyClient(new_token)
                print("  New account loaded — retrying current batch...")
                # Do NOT advance batch_idx — retry the same batch

            else:
                # Non-quota error — log and skip this batch
                print(f"  Batch error: {e}")
                for p in batch:
                    if p["id"] not in progress["failed"]:
                        progress["failed"].append(p["id"])
                save_progress(progress)
                batch_idx += 1

    # Final summary
    print(f"\n{'='*50}")
    final_ids    = scan_dataset_for_ids()
    print(f"Done.")
    print(f"  Places in dataset: {len(final_ids)}/{total_places}")
    print(f"  Saved this session: {total_saved}")
    if progress["failed"]:
        non_done_failed = [f for f in progress["failed"] if f not in final_ids]
        if non_done_failed:
            print(f"  Places with no reviews: {len(non_done_failed)}")
