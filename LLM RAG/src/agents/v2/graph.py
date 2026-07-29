"""
src/agents/v2/graph.py
======================
LangGraph workflow for the v2 pipeline with Query Intent Classification.

New flow:
    START
      ↓
    query_intent_classifier
      ↓
    intent_based_researcher
      ↓
    writer
      ↓
    fact_checker
      ↓ approved?
         yes → evidence_formatter → END
         no  → writer (up to 3 iterations)

Research Value:
    Adding the intent classifier before the researcher makes the RAG pipeline
    adaptive. Different query types now use different retrieval strategies,
    improving precision and reducing irrelevant context in the writer's input.
"""

from langgraph.graph import StateGraph, END
from src.core.config import settings
from src.agents.v2.state import GraphState
from src.agents.v2.nodes import (
    query_intent_classifier_node,
    intent_based_researcher_node,
    writer_node,
    fact_checker_node,
    evidence_formatter_node,
    kg_translator_node,
    itinerary_formatter_node,
)


def routing_decision(state: GraphState) -> str:
    """Routes fact_checker output: approved → evidence_formatter, rejected → writer."""
    if state.get("is_approved", False):
        return "format_evidence"
    return "rewrite"


def intent_routing_decision(state: GraphState) -> str:
    """Routes after intent classification."""
    intent = state.get("query_intent", {}).get("intent_type", "")
    
    if intent == "itinerary_planning":
        if state.get("moip_result"):
            return "itinerary_formatter"
        else:
            return "end"
            
    return "research"


# ── Build Graph ──────────────────────────────────────────────────
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("query_intent_classifier", query_intent_classifier_node)
workflow.add_node("intent_based_researcher", intent_based_researcher_node)
workflow.add_node("kg_translator", kg_translator_node)
workflow.add_node("writer", writer_node)
workflow.add_node("fact_checker", fact_checker_node)
workflow.add_node("evidence_formatter", evidence_formatter_node)
workflow.add_node("itinerary_formatter", itinerary_formatter_node)

# Intent Classifier Bypass
if settings.use_intent_classifier:
    workflow.set_entry_point("query_intent_classifier")
    workflow.add_conditional_edges(
        "query_intent_classifier",
        intent_routing_decision,
        {
            "research": "intent_based_researcher",
            "itinerary_formatter": "itinerary_formatter",
            "end": END,
        }
    )
else:
    def bypass_intent_node(state: GraphState):
        if not state.get("query_intent"):
            state["query_intent"] = {"intent_type": "general_query", "entities": {}, "constraints": {}}
        return state
        
    workflow.add_node("bypass_intent", bypass_intent_node)
    workflow.set_entry_point("bypass_intent")
    workflow.add_conditional_edges(
        "bypass_intent",
        intent_routing_decision,
        {
            "research": "intent_based_researcher",
            "itinerary_formatter": "itinerary_formatter",
            "end": END,
        }
    )

workflow.add_edge("intent_based_researcher", "kg_translator")
workflow.add_edge("kg_translator", "writer")

# Fact Checker Bypass
if settings.use_fact_checker:
    workflow.add_edge("writer", "fact_checker")
    workflow.add_conditional_edges(
        "fact_checker",
        routing_decision,
        {
            "format_evidence": "evidence_formatter",
            "rewrite": "writer",
        },
    )
else:
    workflow.add_edge("writer", "evidence_formatter")

# Final edge
workflow.add_edge("evidence_formatter", END)
workflow.add_edge("itinerary_formatter", END)

# Compile
app = workflow.compile()
