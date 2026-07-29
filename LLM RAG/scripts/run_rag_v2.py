"""
scripts/run_rag_v2.py
=====================
Entry point for the v2 pipeline with Query Intent Classification.

Provides the same process_query() API as run_rag.py so it can be
swapped into the UI server without changing the server code.

Usage (interactive CLI):
    python scripts/run_rag_v2.py

Usage (import):
    from scripts.run_rag_v2 import process_query
    result = process_query("Where can I surf in Sri Lanka?")
    print(result["text"])
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.v2.graph import app
from src.core.logger import get_logger

logger = get_logger("run_rag_v2")


def process_query(question: str) -> dict:
    """
    Runs the v2 multi-agent RAG pipeline for a given question.

    Returns:
        {
            "text":             str   — final evidenced response (or plain draft fallback)
            "structured_facts": list  — raw graph facts for UI evidence display
            "query_intent":     dict  — detected intent classification
        }
    """
    # Initialise state — query_intent starts as None (set by classifier node)
    inputs = {
        "question": question,
        "query_intent": None,           # v2: populated by query_intent_classifier_node
        "research_data": "",
        "draft": "",
        "feedback": "",
        "iterations": 0,
        "is_approved": False,
        "structured_facts": [],
        "evidenced_response": None,
    }
    current_state = inputs.copy()

    logger.info(f"[v2 Pipeline] Processing query: {question}")

    for output in app.stream(inputs):
        for node_name, state_update in output.items():
            logger.info(f"[STEP COMPLETE] {node_name.upper()} returned updates")
            if state_update is not None:
                for k, v in state_update.items():
                    val_str = str(v)[:120] + "..." if len(str(v)) > 120 else str(v)
                    logger.debug(f"  - {k}: {val_str}")
                current_state.update(state_update)

    text = (
        current_state.get("evidenced_response")
        or current_state.get("draft", "No response generated.")
    )
    facts = current_state.get("structured_facts") or []
    intent = current_state.get("query_intent") or {}

    return {
        "text": text,
        "structured_facts": facts,
        "query_intent": intent,
        "research_data": current_state.get("research_data", "")
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Sri Lanka Tourism RAG v2 — Intent-Based Retrieval Pipeline")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("User Question: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            result = process_query(question)

            print("\n" + "=" * 60)
            intent = result.get("query_intent", {})
            print(f"[Intent]  {intent.get('intent_type', 'unknown')} "
                  f"(confidence: {intent.get('confidence', 0):.2f})")
            print("\nFINAL ANSWER:")
            print(result["text"])
            print("=" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(f"Error processing query: {e}")
