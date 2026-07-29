"""
src/agents/v2/retrieval_router.py
==================================
Reusable helper functions for intent-based retrieval routing.

Research Value:
    Separating retrieval logic into a dedicated router module makes the pipeline
    more explainable and modular. Each function corresponds to a specific retrieval
    concern, making it easy to audit, extend, or replace individual strategies.
    Intent-specific retrieval reduces irrelevant context passed to the writer,
    which directly improves recommendation quality and reduces hallucination.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# JSON Utilities
# ──────────────────────────────────────────────────────────────────────────────

def safe_json_parse(text: str) -> Optional[dict]:
    """
    Safely parses a JSON string. Strips markdown code blocks if present.
    Returns None if parsing fails.
    """
    if not text:
        return None
    try:
        # Strip markdown code fences if LLM disobeyed instructions
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Remove first line (```json or ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[RetrievalRouter] JSON parse failed: {e}")
        return None


def build_fallback_intent() -> dict:
    """
    Returns a safe general_query fallback intent when classification fails.
    The pipeline continues with hybrid retrieval instead of crashing.
    """
    return {
        "intent_type": "general_query",
        "secondary_intents": [],
        "entities": {
            "places": [],
            "activities": [],
            "features": [],
            "facilities": [],
            "locations": [],
            "categories": []
        },
        "constraints": {
            "positive": [],
            "negative": [],
            "traveller_type": None,
            "time_or_season": None,
            "budget": None,
            "safety_or_accessibility": []
        },
        "retrieval_strategy": {
            "use_knowledge_graph": True,
            "use_vector_reviews": True,
            "use_evidence_table": True,
            "preferred_graph_relations": [],
            "vector_search_focus": "general tourism recommendation evidence Sri Lanka",
            "ranking_focus": ["overall evidence quality", "positive sentiment", "review frequency"]
        },
        "reason_for_strategy": "Fallback general query because intent classification failed.",
        "confidence": 0.5
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entity Extraction Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_primary_activity(query_intent: dict) -> str:
    """Returns the first activity mentioned in the intent, or empty string."""
    try:
        activities = query_intent.get("entities", {}).get("activities", [])
        return activities[0] if activities else ""
    except Exception:
        return ""


def extract_primary_feature(query_intent: dict) -> str:
    """Returns the first feature mentioned in the intent, or empty string."""
    try:
        features = query_intent.get("entities", {}).get("features", [])
        return features[0] if features else ""
    except Exception:
        return ""


def extract_primary_facility(query_intent: dict) -> str:
    """Returns the first facility mentioned in the intent, or empty string."""
    try:
        facilities = query_intent.get("entities", {}).get("facilities", [])
        return facilities[0] if facilities else ""
    except Exception:
        return ""


def extract_locations(query_intent: dict) -> List[str]:
    """Returns all location names extracted from the intent."""
    try:
        locations = query_intent.get("entities", {}).get("locations", [])
        return [loc for loc in locations if loc.lower() not in ("sri lanka", "srilanka")]
    except Exception:
        return []


def extract_places(query_intent: dict) -> List[str]:
    """Returns all place names extracted from the intent."""
    try:
        return query_intent.get("entities", {}).get("places", [])
    except Exception:
        return []


def extract_all_activities(query_intent: dict) -> List[str]:
    """Returns all activity names extracted from the intent."""
    try:
        return query_intent.get("entities", {}).get("activities", [])
    except Exception:
        return []


def extract_all_features(query_intent: dict) -> List[str]:
    """Returns all feature names extracted from the intent."""
    try:
        return query_intent.get("entities", {}).get("features", [])
    except Exception:
        return []


def extract_all_facilities(query_intent: dict) -> List[str]:
    """Returns all facility names extracted from the intent."""
    try:
        return query_intent.get("entities", {}).get("facilities", [])
    except Exception:
        return []


def extract_positive_constraints(query_intent: dict) -> List[str]:
    """Returns positive constraint terms."""
    try:
        return query_intent.get("constraints", {}).get("positive", [])
    except Exception:
        return []


def extract_negative_constraints(query_intent: dict) -> List[str]:
    """Returns negative constraint terms."""
    try:
        return query_intent.get("constraints", {}).get("negative", [])
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Vector Query Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_vector_query(question: str, query_intent: dict) -> str:
    """
    Returns the user's actual question as the ChromaDB query string.

    The database contains only Sri Lankan tourism data, so the raw question
    gives the most accurate semantic match — no keyword-template noise needed.

    Falls back to the original question on any error.
    """
    try:
        strategy = query_intent.get("retrieval_strategy", {})
        focus = strategy.get("vector_search_focus", "").strip()

        # If the Intent Classifier provided an explicit focus, use it
        if focus:
            return focus

        # Otherwise use the actual question directly
        return question

    except Exception as e:
        logger.warning(f"[RetrievalRouter] build_vector_query failed: {e}")
        return question


# ──────────────────────────────────────────────────────────────────────────────
# Result Formatters
# ──────────────────────────────────────────────────────────────────────────────

def format_graph_results(results: List[dict], query_intent: dict = None, place_docs: dict = None, question: str = "") -> str:
    """
    Formats Neo4j result dicts into a readable numbered list for the research_data string.
    If place_docs is provided (a dict mapping place_name -> list of doc objects), relevant
    review/Wikipedia chunks are embedded directly under each graph fact so the Writer
    sees structured KG data and its supporting evidence together in one block.
    Handles missing keys gracefully.
    """
    if not results:
        return "No graph results returned."

    fallback_place = "Unknown Place"
    if query_intent:
        extracted_places = query_intent.get("entities", {}).get("places", [])
        if extracted_places:
            fallback_place = extracted_places[0]

    lines = []
    for i, r in enumerate(results, start=1):
        parts = [f"Knowledge Graph Fact {i}."]
        # Try common key patterns
        place = r.get("place") or r.get("Place") or r.get("p.name") or r.get("name") or fallback_place
        parts.append(f"Place: {place}")

        activity = r.get("activity") or r.get("Activity") or r.get("a.name")
        if activity:
            parts.append(f"Activity: {activity}")

        feature = r.get("feature") or r.get("Feature") or r.get("f.name")
        if feature:
            parts.append(f"Feature: {feature}")

        facility = r.get("facility") or r.get("Facility") or r.get("fa.name")
        if facility:
            parts.append(f"Facility: {facility}")

        sentiment = r.get("sentiment") or r.get("Sentiment") or r.get("ActivitySentiment") or r.get("FeatureSentiment") or r.get("r.sentiment")
        if sentiment:
            parts.append(f"Sentiment: {sentiment}")

        percentage = r.get("percentage") or r.get("Percentage") or r.get("ActivityPercentage") or r.get("FeaturePercentage") or r.get("r.percentage")
        if percentage and any(w in question.lower() for w in ["percent", "score", "statistic", "rating"]):
            parts.append(f"Percentage: {percentage}%")

        district = r.get("district") or r.get("District") or r.get("d.name")
        if district:
            parts.append(f"District: {district}")

        distance_km = r.get("distance_km")
        if distance_km is not None:
            parts.append(f"Road Distance: {distance_km} km")

        province = r.get("province") or r.get("pr.name")
        if province:
            parts.append(f"Province: {province}")

        category = r.get("category") or r.get("c.name")
        if category:
            parts.append(f"Category: {category}")

        # Handle nested lists (e.g. from comparison query)
        activities_list = r.get("activities") or r.get("top_activities")
        if isinstance(activities_list, list) and activities_list:
            activity_strs = []
            for a in activities_list[:5]:
                if isinstance(a, dict):
                    a_name = a.get("activity") or a.get("name") or str(a)
                    activity_strs.append(str(a_name))
                else:
                    activity_strs.append(str(a))
            parts.append(f"Activities: {', '.join(activity_strs)}")

        features_list = r.get("features") or r.get("top_features")
        if isinstance(features_list, list) and features_list:
            feature_strs = []
            for f in features_list[:5]:
                if isinstance(f, dict):
                    f_name = f.get("feature") or f.get("name") or str(f)
                    feature_strs.append(str(f_name))
                else:
                    feature_strs.append(str(f))
            parts.append(f"Features: {', '.join(feature_strs)}")

        facilities_list = r.get("facilities") or r.get("top_facilities")
        if isinstance(facilities_list, list) and facilities_list:
            facility_strs = []
            for fa in facilities_list[:5]:
                if isinstance(fa, dict):
                    fa_name = fa.get("facility") or fa.get("name") or str(fa)
                    facility_strs.append(str(fa_name))
                else:
                    facility_strs.append(str(fa))
            parts.append(f"Facilities: {', '.join(facility_strs)}")

        # ── New: TimeProfile fields ──────────────────────────────────────────────
        time_of_day = r.get("time_of_day") or r.get("t.time_of_day")
        if time_of_day:
            parts.append(f"Best Time of Day: {time_of_day}")

        season = r.get("season") or r.get("t.season")
        if season:
            parts.append(f"Best Season: {season}")

        months = r.get("months") or r.get("t.months")
        if months and isinstance(months, list) and months:
            parts.append(f"Best Months: {', '.join(months)}")

        avoid = r.get("avoid") or r.get("t.avoid")
        if avoid and isinstance(avoid, list) and avoid:
            parts.append(f"Avoid: {', '.join(avoid)}")

        # ── New: CrowdProfile fields ──────────────────────────────────────────────
        crowd_label = r.get("crowd_label") or r.get("label") or r.get("cr.label")
        if crowd_label:
            parts.append(f"Crowd Level: {crowd_label}")

        crowd_level = r.get("crowd_level") or r.get("level") or r.get("cr.level")
        if crowd_level is not None and not time_of_day:  # avoid duplicate if timing result
            parts.append(f"Crowd Score: {crowd_level}/5")

        busiest = r.get("busiest_period") or r.get("cr.busiest_period")
        if busiest and isinstance(busiest, list) and busiest:
            parts.append(f"Busiest Period: {', '.join(busiest)}")

        # ── New: CostProfile fields ──────────────────────────────────────────────
        cost_level = r.get("cost_level") or r.get("co.level")
        if cost_level:
            parts.append(f"Cost Level: {cost_level}")

        median_lkr = r.get("median_lkr") or r.get("co.median_lkr")
        if median_lkr is not None:
            parts.append(f"Median Entry Fee: LKR {median_lkr}")

        fee_type = r.get("fee_type") or r.get("co.fee_type")
        if fee_type:
            parts.append(f"Fee Type: {fee_type}")

        fact_line = "   ".join(parts)
        lines.append(fact_line)

        # ── Inline supporting review/Wikipedia chunks for this place ────────
        if place_docs:
            docs_for_place = place_docs.get(place, [])
            if docs_for_place:
                lines.append("   Supporting Evidence:")
                for j, doc in enumerate(docs_for_place, start=1):
                    content = getattr(doc, "page_content", str(doc)).strip()
                    lines.append(f"   [{j}] {content}")
        lines.append("")  # blank line between facts for readability

    return "\n".join(lines)


def format_vector_results(docs: list) -> str:
    """
    Formats ChromaDB Document objects into a numbered review evidence list.
    """
    if not docs:
        return "No review evidence returned."

    lines = []
    for i, doc in enumerate(docs, start=1):
        content = getattr(doc, "page_content", str(doc)).strip()
        lines.append(f"{i}. {content}")
    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Research Data Merger
# ──────────────────────────────────────────────────────────────────────────────

def merge_research_data(graph_text: str, vector_text: str, query_intent: dict, quality_metrics: dict = None) -> str:
    """
    Combines graph facts and vector review evidence into a single structured
    research_data string that can be read by the writer node.

    The format is designed to be human-readable so the LLM writer can easily
    extract and cite evidence.
    """
    intent_type = query_intent.get("intent_type", "general_query")
    strategy = query_intent.get("retrieval_strategy", {})
    ranking_focus = strategy.get("ranking_focus", [])
    reason = query_intent.get("reason_for_strategy", "")

    # Build intent summary header
    entities = query_intent.get("entities") or {}
    entity_parts = []
    for key, vals in entities.items():
        if vals:
            entity_parts.append(f"{key}: {', '.join(vals)}")
    entity_str = " | ".join(entity_parts) if entity_parts else "none"

    constraints = query_intent.get("constraints") or {}
    pos = constraints.get("positive", [])
    neg = constraints.get("negative", [])
    constraint_parts = []
    if pos:
        constraint_parts.append(f"positive: {', '.join(pos)}")
    if neg:
        constraint_parts.append(f"negative (avoid): {', '.join(neg)}")
    constraint_str = " | ".join(constraint_parts) if constraint_parts else "none"

    ranking_str = ", ".join(ranking_focus) if ranking_focus else "evidence quality, sentiment"
    
    vector_first = False
    quality_str = ""
    if quality_metrics:
        g_w = quality_metrics.get("graph_weight", 0.5)
        v_w = quality_metrics.get("vector_weight", 0.5)
        
        # Force Graph Facts to always be at the top (Rank 1) to maximize precision
        # unless there are absolutely no Graph Facts available.
        if v_w > g_w and ("No graph results" in graph_text or not graph_text.strip()):
            vector_first = True
        g_s = quality_metrics.get("graph_score", 0.0)
        v_s = quality_metrics.get("vector_score", 0.0)
        
        if intent_type == "place_attribute_query":
            weight_instruction = "The user is asking for specific features or activities of a place. You MUST prioritize the graph facts listed below and mention them in your answer."
        elif g_w > v_w:
            weight_instruction = f"The Knowledge Graph has higher quality data ({g_s:.2f}). Rely primarily on graph facts and use review evidence only for color."
        else:
            weight_instruction = f"The Vector Search has higher quality data ({v_s:.2f}). Rely primarily on the specific reviews and use graph facts only for background context."
            
        quality_str = f"""
RETRIEVAL QUALITY & WEIGHTS:
- Graph Weight: {g_w:.2f} (Score: {g_s:.2f})
- Vector Weight: {v_w:.2f} (Score: {v_s:.2f})
- INSTRUCTION: {weight_instruction}
"""

    # ── Distance Matrix Section Removed ──────────

    # Dynamically order the context blocks based on weights
    if vector_first:
        data_blocks = f"""REVIEW EVIDENCE (ChromaDB):
{vector_text}

GRAPH FACTS (Neo4j Knowledge Graph):
{graph_text}"""
    else:
        data_blocks = f"""GRAPH FACTS (Neo4j Knowledge Graph):
{graph_text}

REVIEW EVIDENCE (ChromaDB):
{vector_text}"""

    intent_str = build_intent_summary(query_intent) if query_intent else "Intent: general_query"

    research_data = f"""QUERY INTENT:
{intent_str}
{quality_str}
{data_blocks}

RETRIEVAL NOTES:
- Ranked by: {ranking_str}
- CONTRADICTION RULE: If Review Evidence contradicts the Knowledge Graph facts, you MUST treat the Knowledge Graph as the absolute ground truth.
"""
    return research_data


# ──────────────────────────────────────────────────────────────────────────────
# Intent Summary for Writer Prompt
# ──────────────────────────────────────────────────────────────────────────────

def build_intent_summary(query_intent: dict) -> str:
    """
    Builds a concise intent summary string to inject into the writer prompt.
    """
    try:
        intent_type = query_intent.get("intent_type", "general_query")
        secondary = query_intent.get("secondary_intents", [])
        entities = query_intent.get("entities") or {}
        constraints = query_intent.get("constraints") or {}
        confidence = query_intent.get("confidence", 0.5)
        reason = query_intent.get("reason_for_strategy", "")

        lines = [
            f"Primary Intent: {intent_type}",
            f"Confidence: {confidence:.2f}",
        ]
        if secondary:
            lines.append(f"Secondary Intents: {', '.join(secondary)}")

        # Entities
        for key, vals in entities.items():
            if vals:
                lines.append(f"  {key.capitalize()}: {', '.join(vals)}")

        # Constraints
        pos = constraints.get("positive", [])
        neg = constraints.get("negative", [])
        traveller = constraints.get("traveller_type")
        if pos:
            lines.append(f"  Positive conditions: {', '.join(pos)}")
        if neg:
            lines.append(f"  Negative conditions (avoid): {', '.join(neg)}")
        if traveller:
            lines.append(f"  Traveller type: {traveller}")

        if reason:
            lines.append(f"Strategy reason: {reason}")

        return "\n".join(lines)
    except Exception:
        return "Intent: general_query (summary unavailable)"
