"""
scripts/test_query_intent.py
============================
Test script for the Query Intent Classifier node (v2).

Tests 7 query types and prints:
  - question
  - detected intent type
  - extracted entities
  - retrieval strategy
  - confidence score

Usage:
    cd "c:\\LLM RAG\\LLMRag-LangChain"
    python scripts/test_query_intent.py

Note: This script ONLY runs the intent classifier node. It does NOT
run the full pipeline (no Neo4j / ChromaDB calls are made).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.v2.nodes import query_intent_classifier_node
from src.agents.v2.retrieval_router import build_fallback_intent

# ── Test Cases ──────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": 1,
        "question": "Where can I surf in Sri Lanka?",
        "expected_intent": "activity_query",
    },
    {
        "id": 2,
        "question": "Find quiet beaches in Sri Lanka",
        "expected_intent": "feature_query or constraint_query",
    },
    {
        "id": 3,
        "question": "Places near Kandy",
        "expected_intent": "location_query",
    },
    {
        "id": 4,
        "question": "Compare Mirissa and Hikkaduwa for surfing",
        "expected_intent": "comparison_query (secondary: activity_query)",
    },
    {
        "id": 5,
        "question": "Family friendly beaches that are not crowded",
        "expected_intent": "constraint_query",
    },
    {
        "id": 6,
        "question": "Why did you recommend Ella?",
        "expected_intent": "explanation_query",
    },
    {
        "id": 7,
        "question": "Best places to visit in Sri Lanka",
        "expected_intent": "general_query",
    },
    {
        "id": 8,
        "question": "Places with wifi and parking in Sri Lanka",
        "expected_intent": "facility_query",
    },
    {
        "id": 9,
        "question": "Which hotels have a swimming pool?",
        "expected_intent": "facility_query",
    },
]


def print_separator(char="-", width=65):
    print(char * width)


def print_intent_result(case: dict, intent: dict):
    print_separator()
    print(f"Test #{case['id']}")
    print(f"  Question        : {case['question']}")
    print(f"  Expected Intent : {case['expected_intent']}")
    print(f"  Detected Intent : {intent.get('intent_type', 'N/A')}")
    print(f"  Confidence      : {intent.get('confidence', 0.0):.2f}")

    secondary = intent.get("secondary_intents", [])
    if secondary:
        print(f"  Secondary       : {', '.join(secondary)}")

    entities = intent.get("entities", {})
    entity_parts = []
    for key, vals in entities.items():
        if vals:
            entity_parts.append(f"{key}={vals}")
    if entity_parts:
        print(f"  Entities        : {' | '.join(entity_parts)}")

    constraints = intent.get("constraints", {})
    pos = constraints.get("positive", [])
    neg = constraints.get("negative", [])
    traveller = constraints.get("traveller_type")
    if pos:
        print(f"  Positive cond.  : {pos}")
    if neg:
        print(f"  Negative cond.  : {neg}")
    if traveller:
        print(f"  Traveller type  : {traveller}")

    strategy = intent.get("retrieval_strategy", {})
    kg = strategy.get("use_knowledge_graph", False)
    vec = strategy.get("use_vector_reviews", False)
    relations = strategy.get("preferred_graph_relations", [])
    focus = strategy.get("vector_search_focus", "")
    ranking = strategy.get("ranking_focus", [])
    print(f"  Knowledge Graph : {'YES' if kg else 'NO'}")
    print(f"  Vector Reviews  : {'YES' if vec else 'NO'}")
    if relations:
        print(f"  Graph Relations : {', '.join(relations)}")
    if focus:
        print(f"  Vector Focus    : {focus}")
    if ranking:
        print(f"  Ranking Focus   : {', '.join(ranking)}")

    reason = intent.get("reason_for_strategy", "")
    if reason:
        print(f"  Strategy Reason : {reason}")

    # Pass/fail indicator
    detected = intent.get("intent_type", "")
    expected_raw = case["expected_intent"].lower()
    passed = detected.lower() in expected_raw
    print(f"  Result          : {'✓ PASS' if passed else '✗ UNEXPECTED (check manually)'}")


def run_tests():
    print_separator("=")
    print("  Query Intent Classifier — Test Suite (v2)")
    print_separator("=")
    print(f"  Running {len(TEST_CASES)} test cases...\n")

    passed = 0
    failed = 0

    for case in TEST_CASES:
        # Build minimal state with just the question
        state = {
            "question": case["question"],
            "query_intent": None,
            "research_data": "",
            "draft": "",
            "feedback": "",
            "iterations": 0,
            "is_approved": False,
            "structured_facts": [],
            "evidenced_response": None,
        }

        try:
            result = query_intent_classifier_node(state)
            intent = result.get("query_intent") or build_fallback_intent()
        except Exception as e:
            print(f"\n[ERROR] Test #{case['id']} crashed: {e}")
            intent = build_fallback_intent()
            intent["intent_type"] = f"ERROR: {e}"

        print_intent_result(case, intent)

        detected = intent.get("intent_type", "")
        if detected.lower() in case["expected_intent"].lower():
            passed += 1
        else:
            failed += 1

    print_separator("=")
    print(f"  Results: {passed} passed, {failed} unexpected out of {len(TEST_CASES)} cases")
    print("  (Some 'unexpected' results may still be correct — verify manually)")
    print_separator("=")


if __name__ == "__main__":
    run_tests()
