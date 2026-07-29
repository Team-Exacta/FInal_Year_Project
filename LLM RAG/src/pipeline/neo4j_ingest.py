import os
import pandas as pd
from src.core.config import settings
from src.core.logger import get_logger
from src.database.neo4j_client import Neo4jClient

logger = get_logger(__name__)

POI_FILENAME = "poi_data.csv"
FEATURES_FILENAME = "popular_features_by_place.csv"
ACTIVITIES_FILENAME = "popular_activities_by_place.csv"
FACILITIES_FILENAME = "popular_facilities_by_place.csv"


def clear_database(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def create_place_node(tx, place, category, district, province):
    query = """
    MERGE (p:Place {name: $place})
    SET p.category = $category
    MERGE (d:District {name: $district})
    MERGE (pr:Province {name: $province})
    MERGE (c:Category {name: $category})
    MERGE (p)-[:LOCATED_IN]->(d)
    MERGE (d)-[:PART_OF]->(pr)
    MERGE (p)-[:HAS_CATEGORY]->(c)
    """
    tx.run(query, place=place, category=category, district=district, province=province)


def create_feature_node(tx, place, feature, sentiment, percentage):
    query = """
    MATCH (p:Place {name: $place})
    MERGE (f:Feature {name: $feature})
    MERGE (p)-[r:HAS_FEATURE]->(f)
    SET r.sentiment = $sentiment, r.percentage = $percentage
    """
    tx.run(query, place=place, feature=feature, sentiment=sentiment, percentage=percentage)


def create_activity_node(tx, place, activity, sentiment, percentage):
    query = """
    MATCH (p:Place {name: $place})
    MERGE (a:Activity {name: $activity})
    MERGE (p)-[r:HAS_ACTIVITY]->(a)
    SET r.sentiment = $sentiment, r.percentage = $percentage
    """
    tx.run(query, place=place, activity=activity, sentiment=sentiment, percentage=percentage)


def create_facility_node(tx, place, facility, sentiment, percentage):
    query = """
    MATCH (p:Place {name: $place})
    MERGE (fa:Facility {name: $facility})
    MERGE (p)-[r:HAS_FACILITY]->(fa)
    SET r.sentiment = $sentiment, r.percentage = $percentage
    """
    tx.run(query, place=place, facility=facility, sentiment=sentiment, percentage=percentage)


def run_ingestion():
    poi_path = os.path.join(settings.data_dir, "graph", POI_FILENAME)
    features_path = os.path.join(settings.data_dir, "graph", FEATURES_FILENAME)
    activities_path = os.path.join(settings.data_dir, "graph", ACTIVITIES_FILENAME)
    facilities_path = os.path.join(settings.data_dir, "graph", FACILITIES_FILENAME)

    # Validate files exist
    for path in [poi_path, features_path, activities_path, facilities_path]:
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            logger.error("Make sure your data files are in the configured data directory.")
            return

    df_poi = pd.read_csv(poi_path)
    df_features = pd.read_csv(features_path)
    df_activities = pd.read_csv(activities_path)
    df_facilities = pd.read_csv(facilities_path)
    logger.info(f"Loaded {len(df_poi)} places, {len(df_features)} features, {len(df_activities)} activities, and {len(df_facilities)} facilities.")

    client = Neo4jClient()

    # Always start fresh – this makes future data updates safe
    logger.info("Clearing existing Neo4j graph (fresh rebuild)...")
    client.execute_write(clear_database)

    # Ingest Places
    logger.info("Ingesting Places, Districts, Provinces and Categories...")
    for _, row in df_poi.iterrows():
        place    = str(row.get("poi_name")    or row.get("Place",    "")).strip()
        category = str(row.get("category")    or row.get("Category", "")).strip()
        district = str(row.get("district")    or row.get("District", "")).strip()
        province = str(row.get("province")    or row.get("Province", "")).strip()

        if not place or place == "nan":
            continue
        district = district if district and district != "nan" else "Unknown"
        province = province if province and province != "nan" else "Unknown"
        category = category if category and category != "nan" else "Uncategorized"

        client.execute_write(create_place_node, place, category, district, province)

    logger.info(f"Ingested {len(df_poi)} place nodes.")

    # Ingest Features
    logger.info("Ingesting Features and relationships...")
    skipped_features = 0
    for _, row in df_features.iterrows():
        place     = str(row.get("place_name")         or row.get("Place",     "")).strip()
        feature   = str(row.get("feature")            or row.get("Feature",   "")).strip()
        sentiment = str(row.get("dominant_sentiment") or row.get("Sentiment", "positive")).strip()
        raw_pct   = row.get("review_percentage") or row.get("Percentage", 0)

        try:
            percentage = float(raw_pct)
        except (ValueError, TypeError):
            percentage = 0.0

        if place and place != "nan" and feature and feature != "nan":
            client.execute_write(create_feature_node, place, feature, sentiment, percentage)
        else:
            skipped_features += 1

    logger.info(f"Ingested features. Skipped {skipped_features} rows with missing data.")

    # Ingest Activities
    logger.info("Ingesting Activities and relationships...")
    skipped_activities = 0
    for _, row in df_activities.iterrows():
        place     = str(row.get("place_name")         or row.get("Place",     "")).strip()
        activity  = str(row.get("activity")           or row.get("Activity",  "")).strip()
        sentiment = str(row.get("dominant_sentiment") or row.get("Sentiment", "positive")).strip()
        raw_pct   = row.get("review_percentage") or row.get("Percentage", 0)

        try:
            percentage = float(raw_pct)
        except (ValueError, TypeError):
            percentage = 0.0

        if place and place != "nan" and activity and activity != "nan":
            client.execute_write(create_activity_node, place, activity, sentiment, percentage)
        else:
            skipped_activities += 1

    logger.info(f"Ingested activities. Skipped {skipped_activities} rows with missing data.")
    
    # Ingest Facilities
    logger.info("Ingesting Facilities and relationships...")
    skipped_facilities = 0
    for _, row in df_facilities.iterrows():
        place     = str(row.get("place_name")         or row.get("Place",     "")).strip()
        facility  = str(row.get("facility")           or row.get("Facility",  "")).strip()
        sentiment = str(row.get("dominant_sentiment") or row.get("Sentiment", "positive")).strip()
        raw_pct   = row.get("review_percentage") or row.get("Percentage", 0)

        try:
            percentage = float(raw_pct)
        except (ValueError, TypeError):
            percentage = 0.0

        if place and place != "nan" and facility and facility != "nan":
            client.execute_write(create_facility_node, place, facility, sentiment, percentage)
        else:
            skipped_facilities += 1

    logger.info(f"Ingested facilities. Skipped {skipped_facilities} rows with missing data.")
    
    client.close()
    logger.info("Neo4j ingestion complete. Knowledge Graph is ready.")


if __name__ == "__main__":
    run_ingestion()
