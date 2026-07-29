"""
src/agents/v2/state.py
======================
Extended GraphState and Pydantic models for the v2 Query Intent Classification pipeline.

Research Value:
    Adding structured intent models makes the RAG pipeline adaptive. Instead of using
    the same retrieval strategy for every question, the system first classifies the
    user query and then selects the best retrieval path. This improves retrieval
    precision, reduces irrelevant context, and helps reduce hallucination because the
    writer receives more targeted evidence.
"""

from typing import TypedDict, List, Optional, Any
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Intent Sub-Models
# ──────────────────────────────────────────────────────────────────────────────

class QueryEntities(BaseModel):
    """Named entities extracted from the user question."""
    places: List[str] = Field(default_factory=list, description="Named places mentioned in the question.")
    activities: List[str] = Field(default_factory=list, description="Activities mentioned (e.g. surfing, hiking).")
    features: List[str] = Field(default_factory=list, description="Features or qualities mentioned (e.g. quiet, scenic).")
    facilities: List[str] = Field(default_factory=list, description="Facilities mentioned (e.g. restaurant, accommodation).")
    locations: List[str] = Field(default_factory=list, description="Geographic locations such as districts or provinces.")
    categories: List[str] = Field(default_factory=list, description="Place categories mentioned (e.g. beach, temple).")


class QueryConstraints(BaseModel):
    """Positive and negative conditions extracted from the user question."""
    positive: List[str] = Field(default_factory=list, description="Desired conditions (e.g. family friendly, clean).")
    negative: List[str] = Field(default_factory=list, description="Unwanted conditions (e.g. not crowded, avoid noisy).")
    traveller_type: Optional[str] = Field(default=None, description="Type of traveller if mentioned (e.g. family, solo, couple).")
    time_or_season: Optional[str] = Field(default=None, description="Season or time constraint if mentioned.")
    budget: Optional[str] = Field(default=None, description="Budget constraint if mentioned (e.g. budget, luxury).")
    safety_or_accessibility: List[str] = Field(default_factory=list, description="Safety or accessibility requirements.")


class RetrievalStrategy(BaseModel):
    """Recommended retrieval strategy for this query intent."""
    use_knowledge_graph: bool = Field(default=True, description="Whether to query Neo4j Knowledge Graph.")
    use_vector_reviews: bool = Field(default=True, description="Whether to query ChromaDB vector store.")
    use_evidence_table: bool = Field(default=True, description="Whether to use evidence scoring from graph relations.")
    preferred_graph_relations: List[str] = Field(
        default_factory=list,
        description="Which graph relations to prioritize (e.g. HAS_ACTIVITY, HAS_FEATURE, HAS_FACILITY, LOCATED_IN, PART_OF)."
    )
    vector_search_focus: str = Field(default="", description="The focused query string for ChromaDB retrieval.")
    ranking_focus: List[str] = Field(default_factory=list, description="Criteria to rank results by.")
    suggested_retrieval_limit: int = Field(default=8, ge=1, le=10, description="Recommended number of items to retrieve based on the query scope.")


class QueryIntent(BaseModel):
    """
    Full structured intent output produced by the Query Intent Classifier.

    Allowed intent_type values:
        activity_query    — user asks for places suitable for an activity
        feature_query     — user asks for places with a quality or characteristic
        facility_query    — user asks for places with a facility available
        location_query    — user asks for places in/near a specific location
        comparison_query  — user compares two or more places
        constraint_query  — user gives multiple conditions or restrictions
        explanation_query — user asks why a place was recommended or asks for evidence
        place_attribute_query — user asks for details/attributes of a specific place
        timing_query      — user asks best time of day/season/month to visit a place
        crowd_query       — user asks about crowd levels or busyness at places
        cost_query        — user asks about entry fees, pricing, or budget options
        nearby_query      — user asks for places near/around a specific named Place (e.g. "places near Mirissa", "what to visit around Ella")
        itinerary_planning — user asks to plan a trip or itinerary
        general_query     — broad query that does not clearly match the above types
    """
    intent_type: str = Field(description="Primary intent classification.")
    secondary_intents: List[str] = Field(default_factory=list, description="Secondary intents if the query has multiple.")
    entities: QueryEntities = Field(default_factory=QueryEntities, description="Entities extracted from the question.")
    constraints: QueryConstraints = Field(default_factory=QueryConstraints, description="Constraints extracted from the question.")
    retrieval_strategy: RetrievalStrategy = Field(default_factory=RetrievalStrategy, description="Recommended retrieval strategy.")
    reason_for_strategy: str = Field(default="", description="Explanation of why this strategy was selected.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score between 0 and 1.")


# ──────────────────────────────────────────────────────────────────────────────
# Fact Check Models (identical to original for compatibility)
# ──────────────────────────────────────────────────────────────────────────────

class ClaimVerification(BaseModel):
    claim: str = Field(description="An atomic claim extracted from the draft.")
    is_supported: bool = Field(description="True if the claim is supported by research data, False otherwise.")
    evidence: str = Field(description="The specific fact or text that supports or refutes the claim.")


class FactCheckOutput(BaseModel):
    is_approved: bool = Field(description="True if ALL claims are supported, False otherwise.")
    feedback: str = Field(description="Summary of feedback for the writer if not approved.")
    claims: List[ClaimVerification] = Field(description="List of all claims extracted and verified.")


# ──────────────────────────────────────────────────────────────────────────────
# Extended GraphState (v2)
# ──────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """
    Extended graph state for the v2 pipeline.

    New fields vs v1:
        query_intent: Structured intent produced by the classifier node (stored as dict).
        moip_result: Result dictionary from the Multi-Objective Itinerary Planning system.

    Backward-compatible fields retained from v1:
        question, research_data, draft, feedback, iterations,
        is_approved, structured_facts, evidenced_response.
    """
    question: str
    query_intent: Optional[dict]                  # NEW: structured intent from classifier
    moip_result: Optional[dict]                   # NEW: ACO simulation result for itinerary
    research_data: str
    draft: str
    feedback: str
    iterations: int
    is_approved: bool
    structured_facts: Optional[List[dict]]        # raw graph facts for evidence scoring
    evidenced_response: Optional[str]             # final formatted answer with evidence badges
