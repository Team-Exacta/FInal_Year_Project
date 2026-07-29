"""
scripts/build_explore_data.py
=============================
Builds the dataset behind the "Explore Sri Lanka" page.

Merges the three project data sources into a single static JSON the UI loads:

  1. MOIP        pois.csv                    -> coordinates, category, cost, visit duration
  2. UGC         _poi_profiles.json          -> best_time / crowd / cost mined from reviews
  3. LLM RAG     data/WikipediaData/*.json   -> description + article link
  4. LLM RAG     data/graph/*.csv            -> popular features / activities / facilities

Images come from the Wikipedia REST summary endpoint and are cached on disk so
re-runs are offline and instant.

Run:
    python scripts/build_explore_data.py            # use cached images
    python scripts/build_explore_data.py --refresh  # re-fetch image URLs
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

def _find_moip_data():
    """The optimiser project has been re-laid-out before; accept either shape."""
    root = os.path.join(ROOT_DIR, "Multi-Objective-Itinerary-Planning")
    candidates = [
        os.path.join(root, "data"),                                              # current layout
        os.path.join(root, "Multi-Objective-Itinerary-Planning--master", "data"),  # previous layout
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "pois.csv")):
            return os.path.normpath(path)
    raise SystemExit(
        "Could not find the MOIP data folder (pois.csv). Looked in:\n  "
        + "\n  ".join(os.path.normpath(p) for p in candidates)
    )


MOIP_DATA = _find_moip_data()
UGC_PROFILES = os.path.join(
    ROOT_DIR, "UGC-analysis", "output", "aggregation", "_poi_profiles.json")
FALLBACK_PROFILES = os.path.join(BASE_DIR, "suggestionData", "_poi_profiles.json")
WIKI_DIR = os.path.join(BASE_DIR, "data", "WikipediaData")
GRAPH_DIR = os.path.join(BASE_DIR, "data", "graph")

OUT_PATH = os.path.join(BASE_DIR, "ui", "data", "explore_places.json")
IMAGE_CACHE = os.path.join(BASE_DIR, "data", "wiki_image_cache.json")

USER_AGENT = "LankaTravelAI-ExploreBuilder/1.0 (academic project; contact via repo)"


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

# Some scraped profile keys carry the TripAdvisor "claim this listing" boilerplate.
JUNK_SUFFIX = re.compile(
    r"(Unclaimed|Claimed)?If_?you_?own_?this_?business.*$", re.IGNORECASE)


def clean_name(raw: str) -> str:
    """Strip scraper boilerplate and underscores from a profile key/name."""
    name = JUNK_SUFFIX.sub("", raw)
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def norm_key(name: str) -> str:
    """Loose key so 'Baker s Falls' matches \"Baker's Falls\"."""
    return re.sub(r"[^a-z0-9]", "", clean_name(name).lower())


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------------------------------------------------------------------------
# Region: Sri Lanka has no district column filled in the source data, so derive
# a coarse travel region from coordinates. Ordered - first match wins.
# ---------------------------------------------------------------------------

REGIONS = [
    ("Northern",       lambda la, lo: la >= 8.55),
    ("East Coast",     lambda la, lo: lo >= 81.30 and la >= 6.60),
    ("Cultural Triangle", lambda la, lo: 7.55 <= la < 8.55 and lo < 81.30),
    ("Hill Country",   lambda la, lo: 6.55 <= la < 7.55 and 80.35 <= lo < 81.30),
    ("West Coast",     lambda la, lo: la >= 6.60 and lo < 80.35),
    ("South Coast",    lambda la, lo: la < 6.60),
]


def region_for(lat, lon):
    if lat is None or lon is None:
        return "Sri Lanka"
    for name, test in REGIONS:
        try:
            if test(lat, lon):
                return name
        except TypeError:
            continue
    return "Sri Lanka"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pois():
    path = os.path.join(MOIP_DATA, "pois.csv")
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["name"].strip()

            def num(key, cast=float):
                try:
                    return cast(row[key])
                except (TypeError, ValueError, KeyError):
                    return None

            out[norm_key(name)] = {
                "name": name,
                "category": (row.get("category") or "").strip() or "Attraction",
                "satisfaction": num("satisfaction_score"),
                "planner_cost": num("cost"),
                "duration_min": num("duration_time_min", int),
                "lat": num("latitude"),
                "lon": num("longitude"),
            }
    return out


def load_profiles():
    path = UGC_PROFILES if os.path.exists(UGC_PROFILES) else FALLBACK_PROFILES
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, prof in raw.items():
        name = clean_name(prof.get("place_name") or key)
        prof = dict(prof)
        prof["place_name"] = name
        out[norm_key(name)] = prof
    print(f"  profiles source: {os.path.relpath(path, ROOT_DIR)}")
    return out


def load_wikipedia():
    out = {}
    if not os.path.isdir(WIKI_DIR):
        return out
    for fname in os.listdir(WIKI_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(WIKI_DIR, fname), encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        name = (d.get("poi_name") or "").strip()
        if name:
            out[norm_key(name)] = d
    return out


def load_graph_csv(fname, value_col):
    """popular_*_by_place.csv -> {norm_key: [{name, pct, sentiment}, ...]}"""
    path = os.path.join(GRAPH_DIR, fname)
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = norm_key(row.get("place_name", ""))
            if not key:
                continue
            try:
                pct = round(float(row.get("review_percentage") or 0), 1)
            except ValueError:
                pct = 0.0
            out.setdefault(key, []).append({
                "name": (row.get(value_col) or "").strip(),
                "pct": pct,
                "sentiment": (row.get("dominant_sentiment") or "neutral").strip(),
            })
    for key in out:
        out[key].sort(key=lambda x: x["pct"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Wikipedia images
# ---------------------------------------------------------------------------

SKIP_EXT = (".svg", ".pdf", ".ogv", ".webm", ".tif", ".tiff", ".gif")

WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def api_get(base, params, attempts=3):
    """GET with retry - the media APIs throttle, and a silent {} costs us images."""
    params = dict(params, format="json")
    url = base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.load(resp)
        except Exception:
            if attempt == attempts - 1:
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


def _pages(payload):
    return list((payload.get("query", {}).get("pages") or {}).values())


def batch_pageimages(titles):
    """Resolve thumbnails for up to 50 article titles in one request."""
    out = {}
    if not titles:
        return out
    data = api_get(WIKI_API, {
        "action": "query",
        "titles": "|".join(titles),
        "redirects": 1,
        "prop": "pageimages",
        "piprop": "thumbnail|original",
        "pithumbsize": 800,
    })
    # Map any redirect/normalisation back to the title we asked for.
    alias = {}
    for group in ("normalized", "redirects"):
        for item in data.get("query", {}).get(group, []) or []:
            alias[item["to"]] = item["from"]

    for page in _pages(data):
        thumb = (page.get("thumbnail") or {}).get("source")
        if not thumb:
            continue
        title = page.get("title", "")
        original = (page.get("original") or {}).get("source")
        # unwind alias chains (normalized -> redirect -> final)
        key = title
        for _ in range(3):
            key = alias.get(key, key)
        out[key] = {"thumb": thumb, "full": original or thumb, "credit": title}
    return out


def _tokens(name):
    stop = {"the", "of", "sri", "lanka", "temple", "beach", "falls", "national",
            "park", "museum", "fort", "tower", "point", "view", "rock", "city"}
    return {t for t in re.findall(r"[a-z]+", name.lower()) if len(t) > 3} - stop


def search_image(name):
    """Full-text search fallback, guarded so we don't grab an unrelated article."""
    data = api_get(WIKI_API, {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{name} Sri Lanka",
        "gsrlimit": 3,
        "prop": "pageimages",
        "piprop": "thumbnail|original",
        "pithumbsize": 800,
    })
    want = _tokens(name)
    for page in sorted(_pages(data), key=lambda p: p.get("index", 99)):
        thumb = (page.get("thumbnail") or {}).get("source")
        if not thumb:
            continue
        title = page.get("title", "")
        # Only accept if the article name actually overlaps the place name.
        if want and not (want & _tokens(title)):
            continue
        original = (page.get("original") or {}).get("source")
        return {"thumb": thumb, "full": original or thumb, "credit": title}
    return None


def geo_image(lat, lon, radius=2000):
    """Commons photos taken near the coordinates - always geographically right."""
    if lat is None or lon is None:
        return None
    data = api_get(COMMONS_API, {
        "action": "query",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}",
        "ggsradius": radius,
        "ggslimit": 12,
        "ggsnamespace": 6,
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 800,
    })
    for page in _pages(data):
        info = (page.get("imageinfo") or [{}])[0]
        thumb = info.get("thumburl")
        title = page.get("title", "")
        if not thumb or title.lower().endswith(SKIP_EXT):
            continue
        return {"thumb": thumb, "full": info.get("url") or thumb,
                "credit": title.replace("File:", "")}
    return None


def resolve_images(places, refresh=False):
    cache = {}
    if os.path.exists(IMAGE_CACHE) and not refresh:
        try:
            with open(IMAGE_CACHE, encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    todo = [p for p in places if p["id"] not in cache or not cache[p["id"]].get("thumb")]
    fetched = 0

    # --- pass 1: batched title lookup -------------------------------------
    titled = [p for p in todo if p.get("wikipedia_title")]
    for i in range(0, len(titled), 50):
        chunk = titled[i:i + 50]
        found = batch_pageimages([p["wikipedia_title"] for p in chunk])
        for place in chunk:
            hit = found.get(place["wikipedia_title"])
            if hit:
                cache[place["id"]] = hit
                fetched += 1
        time.sleep(0.15)
    print(f"    pass 1 (titles): {fetched} resolved")

    # --- pass 2: guarded search -------------------------------------------
    before = fetched
    for place in todo:
        if cache.get(place["id"], {}).get("thumb"):
            continue
        hit = search_image(place["name"])
        if hit:
            cache[place["id"]] = hit
            fetched += 1
        time.sleep(0.3)
    print(f"    pass 2 (search): {fetched - before} resolved")

    # --- pass 3: Commons geosearch ----------------------------------------
    before = fetched
    for place in todo:
        if cache.get(place["id"], {}).get("thumb"):
            continue
        hit = geo_image(place["lat"], place["lon"], 2000) \
            or geo_image(place["lat"], place["lon"], 6000)
        cache[place["id"]] = hit or {"thumb": None, "full": None, "credit": None}
        if hit:
            fetched += 1
        time.sleep(0.12)
    print(f"    pass 3 (geosearch): {fetched - before} resolved")

    for place in places:
        entry = cache.get(place["id"]) or {}
        place["image"] = entry.get("thumb")
        place["image_full"] = entry.get("full")
        place["image_credit"] = entry.get("credit")

    os.makedirs(os.path.dirname(IMAGE_CACHE), exist_ok=True)
    with open(IMAGE_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, ensure_ascii=False)
    return fetched


# ---------------------------------------------------------------------------
# Profile trimming - keep what the UI shows, drop the bulk of the evidence
# ---------------------------------------------------------------------------

def trim_evidence(items, limit=3):
    out = []
    for ev in (items or [])[:limit]:
        sentence = (ev.get("sentence") or "").strip()
        if not sentence:
            continue
        out.append({
            "text": sentence,
            "rating": ev.get("rating"),
            "date": ev.get("date"),
        })
    return out


def build_ugc(prof):
    if not prof:
        return None

    ugc = {"total_reviews": prof.get("total_reviews") or 0}

    bt = prof.get("best_time")
    if bt:
        ugc["best_time"] = {
            "time_of_day": bt.get("time_of_day"),
            "season": bt.get("season"),
            "months": bt.get("months") or [],
            "avoid": bt.get("avoid") or [],
            "confidence": bt.get("confidence"),
            "based_on": bt.get("based_on"),
            "evidence": trim_evidence(bt.get("evidence")),
        }

    cr = prof.get("crowd")
    if cr:
        ugc["crowd"] = {
            "level": cr.get("level"),
            "label": cr.get("label"),
            "avg_level": cr.get("avg_level"),
            "busiest_period": cr.get("busiest_period") or [],
            "confidence": cr.get("confidence"),
            "based_on": cr.get("based_on"),
            "evidence": trim_evidence(cr.get("evidence")),
        }

    co = prof.get("cost")
    if co:
        ugc["cost"] = {
            "level": co.get("level"),
            "amount_level": co.get("amount_level"),
            "sentiment_level": co.get("sentiment_level"),
            "median_lkr": co.get("median_lkr"),
            "fee_type": co.get("fee_type"),
            "confidence": co.get("confidence"),
            "based_on": co.get("based_on"),
            "evidence": trim_evidence(co.get("evidence")),
        }

    return ugc


def extract_description(wiki):
    """Short tagline + a couple of readable paragraphs from the RAG text."""
    if not wiki:
        return "", ""
    tagline = (wiki.get("wikipedia_desc") or "").strip()

    body = ""
    rag = wiki.get("rag_text") or ""
    if "SOURCE:" in rag:
        after = rag.split("SOURCE:", 1)[1]
        parts = after.split("\n\n", 1)
        body = parts[1].strip() if len(parts) > 1 else ""
    paragraphs = [p.strip() for p in body.split("\n") if len(p.strip()) > 60]
    body = " ".join(paragraphs[:2])
    if len(body) > 900:
        cut = body[:900].rsplit(". ", 1)[0]
        body = cut + "." if cut else body[:900] + "..."
    return tagline, body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the image cache and re-fetch from Wikipedia")
    ap.add_argument("--no-images", action="store_true",
                    help="skip image resolution entirely")
    args = ap.parse_args()

    print("Building Explore dataset...")
    pois = load_pois()
    profiles = load_profiles()
    wiki = load_wikipedia()
    features = load_graph_csv("popular_features_by_place.csv", "feature")
    activities = load_graph_csv("popular_activities_by_place.csv", "activity")
    facilities = load_graph_csv("popular_facilities_by_place.csv", "facility")

    print(f"  pois={len(pois)} profiles={len(profiles)} wikipedia={len(wiki)}")

    places = []
    for key in sorted(set(pois) | set(profiles)):
        poi = pois.get(key, {})
        prof = profiles.get(key)
        wk = wiki.get(key, {})

        name = poi.get("name") or (prof or {}).get("place_name") or ""
        if not name:
            continue

        lat = poi.get("lat")
        lon = poi.get("lon")
        if lat is None and wk:
            try:
                lat = float(wk.get("latitude"))
                lon = float(wk.get("longitude"))
            except (TypeError, ValueError):
                lat = lon = None

        tagline, body = extract_description(wk)
        category = poi.get("category") or (wk.get("category") or "").strip() or "Attraction"

        places.append({
            "id": slugify(name),
            "name": name,
            "category": category,
            "region": region_for(lat, lon),
            "lat": lat,
            "lon": lon,
            "satisfaction": poi.get("satisfaction"),
            "duration_min": poi.get("duration_min"),
            "planner_cost": poi.get("planner_cost"),
            "tagline": tagline,
            "description": body,
            "wikipedia_title": wk.get("wikipedia_title"),
            "wikipedia_url": wk.get("wikipedia_url"),
            "features": features.get(key, [])[:6],
            "activities": activities.get(key, [])[:6],
            "facilities": facilities.get(key, [])[:6],
            "ugc": build_ugc(prof),
        })

    mapped = [p for p in places if p["lat"] is not None]
    print(f"  merged {len(places)} places ({len(mapped)} with coordinates)")

    if not args.no_images:
        print("  resolving images...")
        fetched = resolve_images(places, refresh=args.refresh)
        print(f"  images: {fetched} fetched, "
              f"{sum(1 for p in places if p.get('image'))}/{len(places)} resolved")

    payload = {
        "generated_from": {
            "pois": "Multi-Objective-Itinerary-Planning/data/pois.csv",
            "ugc": "UGC-analysis/output/aggregation/_poi_profiles.json",
            "wikipedia": "LLM RAG/data/WikipediaData",
            "graph": "LLM RAG/data/graph",
        },
        "counts": {
            "places": len(places),
            "with_coords": len(mapped),
            "with_ugc": sum(1 for p in places if p["ugc"]),
            "with_best_time": sum(1 for p in places if (p["ugc"] or {}).get("best_time")),
            "with_crowd": sum(1 for p in places if (p["ugc"] or {}).get("crowd")),
            "with_cost": sum(1 for p in places if (p["ugc"] or {}).get("cost")),
        },
        "places": places,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nWrote {os.path.relpath(OUT_PATH, BASE_DIR)} ({size_kb:.0f} KB)")
    print(f"  {payload['counts']}")


if __name__ == "__main__":
    sys.exit(main())
