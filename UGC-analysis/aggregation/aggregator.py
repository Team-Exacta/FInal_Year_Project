"""Step 6 — POI Aggregation.

Reads per-review enriched JSONs from output/extraction/ and builds one
planner-ready profile per place with confidence scores.
"""

import json
import os
import re
from collections import Counter

from config.settings import EXTRACTION_OUTPUT_DIR

AGGREGATION_OUTPUT_DIR = os.path.join(
    os.path.dirname(EXTRACTION_OUTPUT_DIR), "aggregation"
)
os.makedirs(AGGREGATION_OUTPUT_DIR, exist_ok=True)

POI_PROFILES_FILE = os.path.join(AGGREGATION_OUTPUT_DIR, "_poi_profiles.json")

_CATEGORIES = ["best_time", "crowd_level", "cost_level"]

_CROWD_LABELS = {1: "EMPTY", 2: "QUIET", 3: "MODERATE", 4: "BUSY", 5: "PACKED"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _place_display_name(key: str) -> str:
    """'Sigiriya_Lion_Rock' → 'Sigiriya Lion Rock' (strips TripAdvisor suffixes)."""
    # Remove TripAdvisor claim/manage suffixes
    key = re.sub(r"UnclaimedIf_you_own.*", "", key)
    key = re.sub(r"Someone_from_this_business.*", "", key)
    return key.replace("_", " ").strip()


def _confidence(top_count: int, total: int) -> float:
    """What fraction of reviews agree on the dominant value."""
    if total == 0:
        return 0.0
    return round(top_count / total, 2)


def _dominant(counter: Counter):
    """Return (value, count) of the most common element, or (None, 0)."""
    if not counter:
        return None, 0
    val, cnt = counter.most_common(1)[0]
    return val, cnt


# ---------------------------------------------------------------------------
# Evidence collection — picks the strongest example sentences that produced
# each aggregated value, so the profile is auditable.
# ---------------------------------------------------------------------------

MAX_EVIDENCE_PER_ASPECT = 10


def _norm_summary(norm: dict, cat: str) -> dict:
    """Compact summary of the normalized fields for a single review's evidence."""
    if cat == "best_time":
        return {k: norm.get(k) for k in
                ("time_of_day", "season", "months", "recommend_day", "avoid")
                if norm.get(k)}
    if cat == "crowd_level":
        return {k: norm.get(k) for k in ("crowd_level", "crowd_label", "crowd_when")
                if norm.get(k) is not None and norm.get(k) != []}
    if cat == "cost_level":
        return {k: norm.get(k) for k in
                ("cost_level", "amount_lkr", "fee_type", "evaluation",
                 "amount_level", "sentiment_level")
                if norm.get(k) is not None}
    return {}


def _collect_evidence(reviews: list, cat: str,
                      match_key: str = None, match_val=None,
                      max_examples: int = MAX_EVIDENCE_PER_ASPECT) -> list:
    """Pick up to N example sentences that produced this aggregated value.

    Ranking (descending):
      1. Reviews whose normalized[cat][match_key] equals match_val
         (i.e. they actually voted for the dominant aggregated value).
      2. Within tied rank, higher analysis_scores[cat] (stronger SBERT match).
      3. Deduplicate sentences and skip empty ones.
    """
    tagged = [r for r in reviews if r.get("analysis_tags", {}).get(cat)]
    if not tagged:
        return []

    def relevance(r):
        norm = (r.get("normalized") or {}).get(cat) or {}
        agrees = 1 if (match_key and norm.get(match_key) == match_val) else 0
        score  = float((r.get("analysis_scores") or {}).get(cat, 0.0))
        return (agrees, score)

    tagged_sorted = sorted(tagged, key=relevance, reverse=True)

    out = []
    seen = set()
    for r in tagged_sorted:
        sents = (r.get("analysis_sentences") or {}).get(cat) or []
        if not sents:
            continue
        sentence = sents[0].strip()
        key = sentence.lower()[:120]
        if not sentence or key in seen:
            continue
        seen.add(key)
        norm = (r.get("normalized") or {}).get(cat) or {}
        out.append({
            "sentence":  sentence,
            "extracted": _norm_summary(norm, cat),
            "rating":    r.get("rating"),
            "date":      r.get("travelDate") or r.get("publishedDate"),
            "score":     round(float((r.get("analysis_scores") or {}).get(cat, 0.0)), 4),
        })
        if len(out) >= max_examples:
            break
    return out


# ---------------------------------------------------------------------------
# Per-category aggregation
# ---------------------------------------------------------------------------

def _agg_best_time(reviews: list) -> dict:
    tagged = [r for r in reviews if r.get("analysis_tags", {}).get("best_time")]
    if not tagged:
        return None

    tod_counter    = Counter()
    season_counter = Counter()
    month_counter  = Counter()
    avoid_counter  = Counter()
    rec_day_counter = Counter()

    for r in tagged:
        norm = r.get("normalized", {}).get("best_time", {})
        if not norm:
            continue
        if norm.get("time_of_day"):
            tod_counter[norm["time_of_day"]] += 1
        if norm.get("season"):
            season_counter[norm["season"]] += 1
        for m in norm.get("months", []):
            month_counter[m] += 1
        for a in norm.get("avoid", []):
            avoid_counter[a] += 1
        if norm.get("recommend_day"):
            rec_day_counter[norm["recommend_day"]] += 1

    tod_val, tod_cnt = _dominant(tod_counter)
    season_val, _   = _dominant(season_counter)

    # reviews that actually produced a time_of_day value
    tod_total = sum(tod_counter.values())

    # Top months by mention frequency (keep up to 6)
    months = [m for m, _ in month_counter.most_common(6)]

    # Avoid list — deduplicated, sorted by frequency
    avoid = [a for a, _ in avoid_counter.most_common()]

    return {
        "time_of_day":  tod_val,
        "season":       season_val,
        "months":       months,
        "avoid":        avoid,
        "confidence":   _confidence(tod_cnt, tod_total),
        "based_on":     len(tagged),
        "evidence":     _collect_evidence(reviews, "best_time",
                                          match_key="time_of_day",
                                          match_val=tod_val),
    }


def _agg_crowd(reviews: list) -> dict:
    tagged = [r for r in reviews if r.get("analysis_tags", {}).get("crowd_level")]
    if not tagged:
        return None

    scores     = []
    when_counter = Counter()

    for r in tagged:
        norm = r.get("normalized", {}).get("crowd_level", {})
        if not norm:
            continue
        if norm.get("crowd_level") is not None:
            scores.append(norm["crowd_level"])
        for w in norm.get("crowd_when", []):
            when_counter[w] += 1

    if not scores:
        return None

    avg_level  = round(sum(scores) / len(scores), 1)
    peak_level = max(scores)
    peak_label = _CROWD_LABELS[peak_level]

    # Confidence: fraction of reviews that scored 4 or 5 (busy/packed)
    level_counter = Counter(scores)
    top_val, top_cnt = _dominant(level_counter)

    busiest = [w for w, _ in when_counter.most_common(3)]

    return {
        "level":          peak_level,
        "label":          peak_label,
        "avg_level":      avg_level,
        "busiest_period": busiest,
        "confidence":     _confidence(top_cnt, len(scores)),
        "based_on":       len(tagged),
        "evidence":       _collect_evidence(reviews, "crowd_level",
                                            match_key="crowd_level",
                                            match_val=top_val),
    }


def _agg_cost(reviews: list) -> dict:
    tagged = [r for r in reviews if r.get("analysis_tags", {}).get("cost_level")]
    if not tagged:
        return None

    # Track each signal source separately (fixes the HIGH+100LKR inconsistency)
    amount_level_counter    = Counter()
    sentiment_level_counter = Counter()
    primary_level_counter   = Counter()
    amounts                 = []
    fee_counter             = Counter()

    for r in tagged:
        norm = r.get("normalized", {}).get("cost_level", {})
        if not norm:
            continue
        if norm.get("cost_level"):
            primary_level_counter[norm["cost_level"]] += 1
        if norm.get("amount_level"):
            amount_level_counter[norm["amount_level"]] += 1
        if norm.get("sentiment_level"):
            sentiment_level_counter[norm["sentiment_level"]] += 1
        if norm.get("amount_lkr") is not None:
            amounts.append(norm["amount_lkr"])
        if norm.get("fee_type"):
            fee_counter[norm["fee_type"]] += 1

    if not primary_level_counter:
        return None

    dom_level, dom_cnt = _dominant(primary_level_counter)
    amt_level, _       = _dominant(amount_level_counter)
    sent_level, _      = _dominant(sentiment_level_counter)
    fee_type, _        = _dominant(fee_counter)

    median_lkr = None
    if amounts:
        s = sorted(amounts)
        median_lkr = s[len(s) // 2]

    return {
        "level":           dom_level,    # primary (amount > sentiment)
        "amount_level":    amt_level,    # from numeric LKR amounts only
        "sentiment_level": sent_level,   # from opinion words only
        "median_lkr":      median_lkr,
        "fee_type":        fee_type,
        "confidence":      _confidence(dom_cnt, sum(primary_level_counter.values())),
        "based_on":        len(tagged),
        "evidence":        _collect_evidence(reviews, "cost_level",
                                             match_key="cost_level",
                                             match_val=dom_level),
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_poi_profile(place_key: str, reviews: list) -> dict:
    """Return one planner-ready profile dict for a place."""
    profile = {
        "place_name":    _place_display_name(place_key),
        "total_reviews": len(reviews),
        "best_time":     _agg_best_time(reviews),
        "crowd":         _agg_crowd(reviews),
        "cost":          _agg_cost(reviews),
    }
    return profile


def run_aggregation(verbose: bool = True) -> dict:
    """Load all extraction JSONs, build profiles, save to _poi_profiles.json."""
    place_files = sorted(
        f[:-5] for f in os.listdir(EXTRACTION_OUTPUT_DIR)
        if f.endswith(".json") and not f.startswith("_")
    )

    profiles = {}
    for idx, place_key in enumerate(place_files, 1):
        src = os.path.join(EXTRACTION_OUTPUT_DIR, f"{place_key}.json")
        with open(src, encoding="utf-8") as f:
            reviews = json.load(f)

        profile = build_poi_profile(place_key, reviews)
        profiles[place_key] = profile

        if verbose:
            bt = "yes" if profile["best_time"] else "no"
            cr = "yes" if profile["crowd"] else "no"
            co = "yes" if profile["cost"] else "no"
            print(f"[{idx}/{len(place_files)}] {profile['place_name']:<45} "
                  f"bt={bt} crowd={cr} cost={co}")

    with open(POI_PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nDone. {len(profiles)} profiles saved to {POI_PROFILES_FILE}")

    return profiles
