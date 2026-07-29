"""
scripts/ingest_poi_profiles.py
==============================
Ingests POI profiles from suggestionData/_poi_profiles.json into Neo4j.

For each place it adds THREE new node types linked to the existing Place node:

  (Place)-[:HAS_BEST_TIME]    -> (TimeProfile)
  (Place)-[:HAS_CROWD_PROFILE]-> (CrowdProfile)
  (Place)-[:HAS_COST]         -> (CostProfile)

Uses MERGE on every node/relationship — safe to re-run without duplicates.

Usage:
    python scripts/ingest_poi_profiles.py
"""

import sys, os, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.neo4j_client import Neo4jClient
from src.core.logger import get_logger

logger = get_logger("ingest_poi_profiles")

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "suggestionData", "_poi_profiles.json"
)

# ─────────────────────────────────────────────────────────────────────────────
# Cypher templates
# ─────────────────────────────────────────────────────────────────────────────

# Check if a Place node already exists (match by name, case-insensitive)
MATCH_PLACE_CYPHER = """
MATCH (p:Place)
WHERE toLower(p.name) = toLower($place_name)
RETURN p.name AS name
LIMIT 1
"""

# ---------- TimeProfile ----------
TIME_CYPHER = """
MATCH (p:Place) WHERE toLower(p.name) = toLower($place_name)
MERGE (t:TimeProfile {place_name: $place_name})
  ON CREATE SET
    t.time_of_day  = $time_of_day,
    t.season       = $season,
    t.months       = $months,
    t.avoid        = $avoid,
    t.confidence   = $confidence,
    t.based_on     = $based_on
  ON MATCH SET
    t.time_of_day  = $time_of_day,
    t.season       = $season,
    t.months       = $months,
    t.avoid        = $avoid,
    t.confidence   = $confidence,
    t.based_on     = $based_on
MERGE (p)-[:HAS_BEST_TIME]->(t)
"""

# ---------- CrowdProfile ----------
CROWD_CYPHER = """
MATCH (p:Place) WHERE toLower(p.name) = toLower($place_name)
MERGE (cr:CrowdProfile {place_name: $place_name})
  ON CREATE SET
    cr.level          = $level,
    cr.label          = $label,
    cr.avg_level      = $avg_level,
    cr.busiest_period = $busiest_period,
    cr.confidence     = $confidence,
    cr.based_on       = $based_on
  ON MATCH SET
    cr.level          = $level,
    cr.label          = $label,
    cr.avg_level      = $avg_level,
    cr.busiest_period = $busiest_period,
    cr.confidence     = $confidence,
    cr.based_on       = $based_on
MERGE (p)-[:HAS_CROWD_PROFILE]->(cr)
"""

# ---------- CostProfile ----------
COST_CYPHER = """
MATCH (p:Place) WHERE toLower(p.name) = toLower($place_name)
MERGE (co:CostProfile {place_name: $place_name})
  ON CREATE SET
    co.level           = $level,
    co.amount_level    = $amount_level,
    co.sentiment_level = $sentiment_level,
    co.median_lkr      = $median_lkr,
    co.fee_type        = $fee_type,
    co.confidence      = $confidence,
    co.based_on        = $based_on
  ON MATCH SET
    co.level           = $level,
    co.amount_level    = $amount_level,
    co.sentiment_level = $sentiment_level,
    co.median_lkr      = $median_lkr,
    co.fee_type        = $fee_type,
    co.confidence      = $confidence,
    co.based_on        = $based_on
MERGE (p)-[:HAS_COST]->(co)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Write transaction functions
# ─────────────────────────────────────────────────────────────────────────────

def _write_time(tx, params):
    tx.run(TIME_CYPHER, **params)

def _write_crowd(tx, params):
    tx.run(CROWD_CYPHER, **params)

def _write_cost(tx, params):
    tx.run(COST_CYPHER, **params)


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion logic
# ─────────────────────────────────────────────────────────────────────────────

def ingest(client: Neo4jClient):
    logger.info(f"Loading data from: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    total         = len(profiles)
    matched       = 0   # Place node found in Neo4j
    skipped       = 0   # Place node NOT found — skipped
    time_added    = 0
    crowd_added   = 0
    cost_added    = 0

    logger.info(f"Found {total} POI entries. Starting ingestion...")

    for key, data in profiles.items():
        place_name = data.get("place_name", key.replace("_", " "))

        # ── Check the Place node exists ──────────────────────────────────
        existing = client.query(MATCH_PLACE_CYPHER, {"place_name": place_name})
        if not existing:
            logger.warning(f"  [SKIP] '{place_name}' — no matching Place node in Neo4j.")
            skipped += 1
            continue

        matched += 1
        logger.info(f"  [MATCH] '{place_name}'")

        # ── 1. HAS_BEST_TIME ─────────────────────────────────────────────
        best_time = data.get("best_time")
        if best_time:
            params = {
                "place_name": place_name,
                "time_of_day": best_time.get("time_of_day"),
                "season":      best_time.get("season"),
                "months":      best_time.get("months") or [],
                "avoid":       best_time.get("avoid") or [],
                "confidence":  best_time.get("confidence", 0.0),
                "based_on":    best_time.get("based_on", 0),
            }
            client.execute_write(_write_time, params)
            time_added += 1

        # ── 2. HAS_CROWD_PROFILE ─────────────────────────────────────────
        crowd = data.get("crowd")
        if crowd:
            params = {
                "place_name":    place_name,
                "level":         crowd.get("level"),
                "label":         crowd.get("label"),
                "avg_level":     crowd.get("avg_level", 0.0),
                "busiest_period":crowd.get("busiest_period") or [],
                "confidence":    crowd.get("confidence", 0.0),
                "based_on":      crowd.get("based_on", 0),
            }
            client.execute_write(_write_crowd, params)
            crowd_added += 1

        # ── 3. HAS_COST ──────────────────────────────────────────────────
        cost = data.get("cost")
        if cost:
            params = {
                "place_name":      place_name,
                "level":           cost.get("level"),
                "amount_level":    cost.get("amount_level"),
                "sentiment_level": cost.get("sentiment_level"),
                "median_lkr":      cost.get("median_lkr"),
                "fee_type":        cost.get("fee_type"),
                "confidence":      cost.get("confidence", 0.0),
                "based_on":        cost.get("based_on", 0),
            }
            client.execute_write(_write_cost, params)
            cost_added += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  POI Profile Ingestion — Summary")
    print("=" * 55)
    print(f"  Total profiles in JSON   : {total}")
    print(f"  Matched Place nodes      : {matched}")
    print(f"  Skipped (no Place node)  : {skipped}")
    print(f"  TimeProfile nodes added  : {time_added}")
    print(f"  CrowdProfile nodes added : {crowd_added}")
    print(f"  CostProfile nodes added  : {cost_added}")
    print("=" * 55)
    print("  Done! New relationships added to the KG:")
    print("    (Place)-[:HAS_BEST_TIME]    ->(TimeProfile)")
    print("    (Place)-[:HAS_CROWD_PROFILE]->(CrowdProfile)")
    print("    (Place)-[:HAS_COST]         ->(CostProfile)")
    print("=" * 55 + "\n")


def main():
    client = Neo4jClient()
    client.connect()
    try:
        ingest(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
