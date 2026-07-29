"""
src/agents/v2/nodes.py
======================
Five LangGraph node functions for the v2 pipeline:
  1. query_intent_classifier_node  — classifies query intent before retrieval
  2. intent_based_researcher_node  — routes retrieval based on intent type
  3. writer_node                   — intent-aware writer
  4. fact_checker_node             — intent-aware fact checker
  5. evidence_formatter_node       — intent-aware evidence formatter

Research Value:
    Intent-based retrieval means each query type gets the most relevant graph
    relations and vector search focus, reducing irrelevant context and improving
    recommendation quality and explainability.
"""

from langchain_core.messages import HumanMessage
from src.core.config import get_llm, settings
from src.core.logger import get_logger
from src.database.neo4j_client import Neo4jClient
from src.database.chroma_client import ChromaClient

from src.agents.v2.state import GraphState, FactCheckOutput, QueryIntent
from src.agents.v2.prompts import (
    QUERY_INTENT_CLASSIFIER_PROMPT,
    _KNOWN_ACTIVITIES,
    _KNOWN_FEATURES,
    _KNOWN_FACILITIES,
    WRITER_PROMPT_TEMPLATE,
    WRITER_FEEDBACK_TEMPLATE,
    FACT_CHECKER_PROMPT_TEMPLATE,
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    GUIDED_CYPHER_PROMPT,
    KG_TRANSLATOR_PROMPT_TEMPLATE,
)
from src.agents.v2.retrieval_router import (
    safe_json_parse,
    build_fallback_intent,
    extract_primary_activity,
    extract_primary_feature,
    extract_primary_facility,
    extract_locations,
    extract_places,
    extract_all_activities,
    extract_all_features,
    extract_all_facilities,
    extract_positive_constraints,
    extract_negative_constraints,
    build_vector_query,
    format_graph_results,
    format_vector_results,
    merge_research_data,
    build_intent_summary,
)
from src.agents.v2.idf_utils import get_dynamic_threshold
from src.agents.v2.graph_vocabulary import (
    resolve_activity_terms,
    resolve_feature_terms,
    resolve_facility_terms,
    resolve_constraint_features,
    get_search_terms_for_activity,
    get_search_terms_for_feature,
    get_search_terms_for_facility,
)

logger = get_logger(__name__)

# Shared singletons
llm = get_llm()
neo4j = Neo4jClient()
chroma = ChromaClient()
retriever = chroma.get_retriever(k=settings.vector_retrieval_limit)


# ─────────────────────────────────────────────────────────────────
# Helper: run ChromaDB retrieval safely
# ─────────────────────────────────────────────────────────────────

def _vector_search(query: str, k: int = None) -> list:
    if k is None: k = settings.vector_retrieval_limit
    """Run ChromaDB retrieval; returns list of Document objects."""
    try:
        if retriever:
            docs = retriever.invoke(query)
            docs = docs[:k]
            logger.info(f"[Researcher] ChromaDB returned {len(docs)} chunks for: '{query[:60]}'")
            return docs
        logger.warning("[Researcher] ChromaDB retriever unavailable.")
        return []
    except Exception as e:
        logger.error(f"[Researcher] ChromaDB retrieval error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# Helper: run Neo4j query safely
# ─────────────────────────────────────────────────────────────────

def _graph_query(cypher: str, params: dict = None) -> list:
    """Executes a Cypher query against Neo4j and returns a list of dictionaries."""
    if not settings.use_graph_retrieval:
        logger.info("[Researcher] Graph retrieval bypassed by config.")
        return []
        
    try:
        results = neo4j.query(cypher, params or {})
        logger.info(f"[Researcher] Neo4j returned {len(results)} graph facts.")
        return results or []
    except Exception as e:
        logger.error(f"[Researcher] Neo4j query error: {e}")
        return []



def _ground_entities(entities: dict) -> dict:
    """
    Queries Neo4j to find exact node names for extracted activities, features, and locations.
    Returns a new entities dict with grounded names.
    """
    grounded = {
        "places": [],
        "activities": [],
        "features": [],
        "facilities": [],
        "locations": [],
        "categories": []
    }
    
    # Helper to ground a term
    def ground_term(term: str, label: str = None) -> str:
        if not term:
            return term
        # Query Neo4j for closest match
        label_clause = f":{label}" if label else ""
        cypher = f"""
MATCH (n{label_clause})
WHERE toLower(n.name) CONTAINS toLower('{term}')
RETURN n.name AS name
ORDER BY CASE WHEN toLower(n.name) = toLower('{term}') THEN 0 ELSE 1 END, size(n.name) ASC
LIMIT 1
"""
        results = _graph_query(cypher)
        if results:
            return results[0]['name']
        return term # Fallback to original term if no match
        
    # Ground Places
    for p in entities.get("places", []):
        grounded["places"].append(ground_term(p, "Place"))
        
    # Ground Activities
    for a in entities.get("activities", []):
        grounded["activities"].append(ground_term(a, "Activity"))
        
    # Ground Features
    for f in entities.get("features", []):
        grounded["features"].append(ground_term(f, "Feature"))
        
    # Ground Facilities
    for fa in entities.get("facilities", []):
        grounded["facilities"].append(ground_term(fa, "Facility"))
        
    # Ground Locations (could be District or Province)
    for l in entities.get("locations", []):
        # Try District first
        grounded_loc = ground_term(l, "District")
        if grounded_loc == l: # If no match or same, try Province
            grounded_loc = ground_term(l, "Province")
        grounded["locations"].append(grounded_loc)
        
    # Categories
    for c in entities.get("categories", []):
        grounded["categories"].append(ground_term(c, "Category"))
        
    logger.info(f"[Researcher] Grounded entities: {grounded}")
    return grounded


# ─────────────────────────────────────────────────────────────────
# NODE 1 — Query Intent Classifier
# ─────────────────────────────────────────────────────────────────

def query_intent_classifier_node(state: GraphState) -> dict:
    """
    Classifies the user query intent BEFORE retrieval.
    Injects the known graph vocabulary into the prompt so the LLM extracts
    graph-compatible activity/feature terms instead of free-form language.

    Reads:  state["question"]
    Writes: state["query_intent"]

    Falls back to general_query if LLM output is invalid or missing.
    Never crashes the pipeline.
    """
    logger.info("Executing QUERY INTENT CLASSIFIER node")
    question = state.get("question", "")

    try:
        # Inject the graph vocabulary so the LLM extracts graph-compatible terms
        prompt = QUERY_INTENT_CLASSIFIER_PROMPT.format(
            question=question,
            known_activities=_KNOWN_ACTIVITIES.strip(),
            known_features=_KNOWN_FEATURES.strip(),
            known_facilities=_KNOWN_FACILITIES.strip(),
        )
        response = llm.invoke([HumanMessage(content=prompt)])

        # Extract text from LangChain response
        content = response.content
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            ).strip()
        else:
            content = str(content).strip()

        parsed = safe_json_parse(content)

        if parsed is None:
            logger.warning("[IntentClassifier] LLM returned invalid JSON. Using fallback intent.")
            return {"query_intent": build_fallback_intent()}

        # Validate with Pydantic
        try:
            intent_obj = QueryIntent(**parsed)
            if intent_obj.confidence is None or intent_obj.confidence == 0.0:
                intent_obj.confidence = 0.5
            intent_dict = intent_obj.model_dump()
        except Exception as ve:
            logger.warning(f"[IntentClassifier] Pydantic validation failed: {ve}. Using raw parsed dict.")
            intent_dict = parsed
            intent_dict.setdefault("confidence", 0.5)

        # ── Post-process: resolve activity/feature terms through vocabulary map ──
        # This fixes any remaining natural-language terms the LLM didn't map.
        entities = intent_dict.get("entities", {})
        raw_activities = entities.get("activities", [])
        raw_features = entities.get("features", [])
        raw_facilities = entities.get("facilities", [])
        if raw_activities:
            intent_dict["entities"]["activities"] = resolve_activity_terms(raw_activities)
        if raw_features:
            intent_dict["entities"]["features"] = resolve_feature_terms(raw_features)
        if raw_facilities:
            intent_dict["entities"]["facilities"] = resolve_facility_terms(raw_facilities)

        # Also resolve constraint terms
        constraints = intent_dict.get("constraints", {})
        raw_pos = constraints.get("positive", [])
        raw_neg = constraints.get("negative", [])
        if raw_pos or raw_neg:
            resolved_pos, resolved_neg = resolve_constraint_features(raw_pos, raw_neg)
            intent_dict["constraints"]["positive"] = resolved_pos
            intent_dict["constraints"]["negative"] = resolved_neg

        logger.info(
            f"[IntentClassifier] Query classified as '{intent_dict.get('intent_type')}' "
            f"with confidence {intent_dict.get('confidence', 0.5):.2f}"
        )
        if intent_dict["entities"].get("activities"):
            logger.info(f"[IntentClassifier] Resolved activities: {intent_dict['entities']['activities']}")
        if intent_dict["entities"].get("features"):
            logger.info(f"[IntentClassifier] Resolved features: {intent_dict['entities']['features']}")
        if intent_dict["entities"].get("facilities"):
            logger.info(f"[IntentClassifier] Resolved facilities: {intent_dict['entities']['facilities']}")

        return {"query_intent": intent_dict}

    except Exception as e:
        logger.error(f"[IntentClassifier] Unexpected error: {e}. Using fallback intent.")
        return {"query_intent": build_fallback_intent()}


# ─────────────────────────────────────────────────────────────────
# NODE 2 — Intent-Based Researcher
# ─────────────────────────────────────────────────────────────────

def intent_based_researcher_node(state: GraphState) -> dict:
    """
    Routes retrieval to the correct strategy based on classified intent.

    Reads:  state["question"], state["query_intent"]
    Writes: state["research_data"], state["structured_facts"]
    """
    logger.info("Executing INTENT-BASED RESEARCHER node")
    question = state.get("question", "")
    query_intent = state.get("query_intent") or build_fallback_intent()
    intent_type = query_intent.get("intent_type", "general_query")

    logger.info(f"[RetrievalRouter] Intent type: {intent_type}")

    graph_results = []
    vector_docs = []

    # ── Dispatch to intent-specific retrieval strategy ──────────────
    # Ground entities to exact graph node names before retrieval
    entities = query_intent.get("entities", {})
    grounded_entities = _ground_entities(entities)
    query_intent["entities"] = grounded_entities
    
    # Use grounded entities for the rest of the logic
    activities = grounded_entities.get("activities", [])
    features = grounded_entities.get("features", [])
    facilities = grounded_entities.get("facilities", [])
    locations = grounded_entities.get("locations", [])

    # Check nearby / proximity queries FIRST so 'near Kandy' or 'around Ella' never get trapped by activity or location queries
    if intent_type == "nearby_query" or any(w in question.lower().split() for w in ["near", "nearby", "around", "closest", "surrounding"]):
        graph_results, vector_docs = _retrieve_nearby(question, query_intent)

    # Comparison queries must be checked BEFORE the multi-faceted check
    # because comparison queries naturally have both activities and features,
    # which would incorrectly trigger _retrieve_multi_hop instead of _retrieve_comparison.
    elif intent_type == "comparison_query":
        graph_results, vector_docs = _retrieve_comparison(question, query_intent)

    # Multi-faceted query check: if query has both activity and feature, or activity and location,
    # or facility and activity/feature/location, we use multi-hop traversal to enrich context.
    elif (activities and features) or (activities and locations and len(locations) > 0) or (facilities and (activities or features or locations)):
        logger.info("[Researcher] Multi-faceted query detected. Using multi-hop traversal retrieval.")
        graph_results, vector_docs = _retrieve_multi_hop(question, query_intent)
    elif intent_type == "activity_query":
        graph_results, vector_docs = _retrieve_activity(question, query_intent)

    elif intent_type == "feature_query":
        graph_results, vector_docs = _retrieve_feature(question, query_intent)

    elif intent_type == "facility_query":
        graph_results, vector_docs = _retrieve_facility(question, query_intent)

    elif intent_type == "place_attribute_query":
        graph_results, vector_docs = _retrieve_place_attributes(question, query_intent)

    elif intent_type == "location_query":
        graph_results, vector_docs = _retrieve_location(question, query_intent)

    elif intent_type == "constraint_query":
        graph_results, vector_docs = _retrieve_constraint(question, query_intent)

    elif intent_type == "explanation_query":
        graph_results, vector_docs = _retrieve_explanation(question, query_intent)

    elif intent_type == "timing_query":
        graph_results, vector_docs = _retrieve_timing(question, query_intent)

    elif intent_type == "crowd_query":
        graph_results, vector_docs = _retrieve_crowd(question, query_intent)

    elif intent_type == "cost_query":
        graph_results, vector_docs = _retrieve_cost(question, query_intent)

    else:  # general_query or unknown
        # Use guided cypher as the default fallback as it is more powerful than general text-to-cypher
        graph_results, vector_docs = _retrieve_guided_cypher(question, query_intent)

    # ── Compensate for Graph Shortfall ───────
    suggested_limit = int(query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8)
    graph_shortfall = max(0, suggested_limit - len(graph_results))
    
    if graph_shortfall > 0:
        logger.info(f"[Researcher] Graph shortfall of {graph_shortfall}. Retrieving additional vector chunks.")
        vec_query = build_vector_query(question, query_intent)
        extra_docs = _vector_search(vec_query, k=graph_shortfall)
        
        existing_content = {getattr(d, 'page_content', str(d)).strip() for d in vector_docs}
        for d in extra_docs:
            c = getattr(d, 'page_content', str(d)).strip()
            if c not in existing_content:
                vector_docs.append(d)
                existing_content.add(c)

    # ── Targeted per-place vector search + group docs by place ───────
    # For each place found in the graph, run a focused vector search and
    # store results grouped by place name so they can be embedded inline
    # with their corresponding KG fact in the formatted output.
    place_to_docs: dict = {}   # place_name -> list of docs
    if graph_results and settings.use_graph_retrieval:
        place_names_in_graph = list({
            r.get("place") for r in graph_results if r.get("place")
        })
        logger.info(
            f"[Researcher] Performing targeted vector search for "
            f"{len(place_names_in_graph)} graph places: {place_names_in_graph}"
        )
        existing_content = {getattr(d, 'page_content', str(d)).strip() for d in vector_docs}
        
        for place_name in place_names_in_graph[:5]:  # cap at 5 places
            targeted_query = f"{place_name} {question}"
            place_docs_found = _vector_search(targeted_query, k=1)
            for doc in place_docs_found:
                content = getattr(doc, 'page_content', str(doc)).strip()
                if content not in existing_content:
                    vector_docs.append(doc)
                    existing_content.add(content)
                # Check doc.metadata for place name first, fallback to exact string matching
                meta = getattr(doc, 'metadata', {}) or {}
                doc_place = (meta.get('place_name') or meta.get('Place') or meta.get('place') or '').strip()
                if doc_place and (doc_place.lower() == place_name.lower() or doc_place.lower() in place_name.lower() or place_name.lower() in doc_place.lower()):
                    place_to_docs.setdefault(place_name, []).append(doc)
                elif not doc_place and place_name.lower() in content.lower():
                    place_to_docs.setdefault(place_name, []).append(doc)

    # Also group any general vector docs that match a graph place by metadata or exact name
    graph_place_names = {r.get("place", "") for r in graph_results if r.get("place")}
    for doc in vector_docs:
        content = getattr(doc, 'page_content', str(doc)).strip()
        meta = getattr(doc, 'metadata', {}) or {}
        doc_place = (meta.get('place_name') or meta.get('Place') or meta.get('place') or '').strip()
        for pname in graph_place_names:
            if not pname:
                continue
            matched = False
            if doc_place and (doc_place.lower() == pname.lower() or doc_place.lower() in pname.lower() or pname.lower() in doc_place.lower()):
                matched = True
            elif not doc_place and pname.lower() in content.lower():
                matched = True
                
            if matched:
                existing = [getattr(d, 'page_content', str(d)).strip() for d in place_to_docs.get(pname, [])]
                if content not in existing:
                    place_to_docs.setdefault(pname, []).append(doc)
                break

    # ── Calculate Quality Scores ─────────────────────────────────────
    graph_density = len(graph_results)
    avg_percentage = 0.0
    if graph_density > 0:
        percentages = [r.get("percentage", 0) for r in graph_results if r.get("percentage") is not None]
        avg_percentage = sum(percentages) / len(percentages) if percentages else 0.0
        
    graph_quality_score = min(1.0, (graph_density / 10.0) * (avg_percentage / 100.0))
    vector_quality_score = min(1.0, len(vector_docs) / 5.0)
    
    total_score = graph_quality_score + vector_quality_score
    if total_score > 0:
        graph_weight = graph_quality_score / total_score
        vector_weight = vector_quality_score / total_score
    else:
        graph_weight, vector_weight = 0.5, 0.5
        
    quality_metrics = {
        "graph_score": graph_quality_score,
        "vector_score": vector_quality_score,
        "graph_weight": graph_weight,
        "vector_weight": vector_weight
    }

    # ── Format and merge ─────────────────────────────────────────────
    # Pass place_to_docs so each KG fact gets its supporting reviews inline.
    # Any unmatched general chunks still go in the vector_text section.
    graph_text = format_graph_results(graph_results, query_intent, place_docs=place_to_docs, question=question)
    unmatched_docs = [
        d for d in vector_docs
        if not any(d in docs for docs in place_to_docs.values())
    ]
    vector_text = format_vector_results(unmatched_docs) if unmatched_docs else ""
    research_data = merge_research_data(graph_text, vector_text, query_intent, quality_metrics)

    logger.info(
        f"[Researcher] Retrieved {len(graph_results)} graph facts "
        f"and {len(vector_docs)} review chunks ({len(unmatched_docs)} unmatched general)."
    )
    return {"research_data": research_data, "structured_facts": graph_results}





# ─────────────────────────────────────────────────────────────────
# Retrieval Strategies
# ─────────────────────────────────────────────────────────────────

def _retrieve_activity(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    A. activity_query — places with HAS_ACTIVITY relation.
    Uses vocabulary-resolved activity terms and multi-term WHERE clauses
    so variants like "Hiking" and "Trekking" both match for a "hike" query.
    Uses: HAS_ACTIVITY, LOCATED_IN, HAS_CATEGORY
    Ranked by: activity evidence percentage DESC
    """
    logger.info("[RetrievalRouter] Using HAS_ACTIVITY + vector review retrieval")
    if not settings.use_graph_retrieval:
        logger.info("[RetrievalRouter] Graph retrieval bypassed by config. Using Vector Fallback.")
        graph_results = []
        vec_query = build_vector_query(question, query_intent)
        vector_docs = _vector_search(vec_query, k=min(suggested_limit, 5))
        return graph_results, vector_docs

    raw_activities = extract_all_activities(query_intent)
    places = query_intent.get("entities", {}).get("places", [])
    locations = extract_locations(query_intent)
    
    # Vocabulary already resolved by classifier; expand each to synonyms for broader Cypher match
    all_search_terms = []
    for act in raw_activities:
        terms = get_search_terms_for_activity(act)
        all_search_terms.extend(terms)
    # Deduplicate while preserving order
    seen = set()
    search_terms = [t for t in all_search_terms if not (t in seen or seen.add(t))]

    dynamic_threshold = get_dynamic_threshold(search_terms)

    graph_results = []
    
    if places and not search_terms:
        # User asks for activities at a specific place
        place_name = places[0]
        logger.info(f"[RetrievalRouter] Querying activities for specific place: {place_name}")
        cypher = f"""
MATCH (p:Place)-[r:HAS_ACTIVITY]->(a:Activity)
WHERE toLower(p.name) CONTAINS toLower('{place_name}') AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
RETURN DISTINCT p.name AS place, a.name AS activity,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category,
       collect(DISTINCT f.name)[0..3] AS top_features
ORDER BY r.percentage DESC
LIMIT {graph_limit}
"""
        graph_results = _graph_query(cypher)
        
    elif search_terms:
        # Build a multi-term CONTAINS clause: any term in list matches
        where_clauses = " OR ".join(
            [f"toLower(a.name) CONTAINS toLower('{t}')" for t in search_terms[:5]]
        )
        
        place_filter = ""
        if places:
            place_filter = f" AND toLower(p.name) CONTAINS toLower('{places[0]}')"
            
        if locations:
            loc_clauses = []
            for loc in locations:
                loc_clauses.append(f"toLower(d.name) CONTAINS toLower('{loc}') OR (pr IS NOT NULL AND toLower(pr.name) CONTAINS toLower('{loc}'))")
            location_filter = f" AND ({' OR '.join(loc_clauses)})"
            cypher = f"""
MATCH (p:Place)-[r:HAS_ACTIVITY]->(a:Activity)
MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (d)-[:PART_OF]->(pr:Province)
WITH p, r, a, d, pr
WHERE ({where_clauses}){place_filter}{location_filter} AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
RETURN DISTINCT p.name AS place, a.name AS activity,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category,
       collect(DISTINCT f.name)[0..3] AS top_features
ORDER BY r.percentage DESC
LIMIT {graph_limit}
"""
        else:
            cypher = f"""
MATCH (p:Place)-[r:HAS_ACTIVITY]->(a:Activity)
WHERE ({where_clauses}){place_filter} AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
RETURN DISTINCT p.name AS place, a.name AS activity,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category,
       collect(DISTINCT f.name)[0..3] AS top_features
ORDER BY r.percentage DESC
LIMIT {graph_limit}
"""
        graph_results = _graph_query(cypher)
        logger.info(f"[RetrievalRouter] Activity search terms: {search_terms}")

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)

    if len(graph_results) == 0 and dynamic_threshold < 5.0:
        logger.warning(f"[RetrievalRouter] Graph returned 0 results for rare terms {search_terms}. Triggering Vector Fallback.")
        fallback_query = " ".join(search_terms) + " Sri Lanka"
        vector_docs += _vector_search(fallback_query, k=5)

    return graph_results, vector_docs


def _retrieve_feature(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    B. feature_query — places with HAS_FEATURE relation.
    Uses vocabulary-resolved feature terms and multi-term WHERE clauses.
    Uses: HAS_FEATURE, LOCATED_IN, HAS_CATEGORY
    Ranked by: feature evidence percentage DESC
    """
    logger.info("[RetrievalRouter] Using HAS_FEATURE + vector review retrieval")
    if not settings.use_graph_retrieval:
        logger.info("[RetrievalRouter] Graph retrieval bypassed by config. Using Vector Fallback.")
        graph_results = []
        vec_query = build_vector_query(question, query_intent)
        vector_docs = _vector_search(vec_query, k=min(suggested_limit, 5))
        return graph_results, vector_docs

    raw_features = extract_all_features(query_intent)
    locations = extract_locations(query_intent)
    
    # Expand each resolved feature to possible synonyms for broader Cypher match
    all_search_terms = []
    for feat in raw_features:
        terms = get_search_terms_for_feature(feat)
        all_search_terms.extend(terms)
    seen = set()
    search_terms = [t for t in all_search_terms if not (t in seen or seen.add(t))]

    dynamic_threshold = get_dynamic_threshold(search_terms)

    # Decouple Peacefulness and Crowdedness conflict while acknowledging they are related
    is_uncrowded_query = False
    question_lower = question.lower()
    uncrowded_indicators = ["uncrowded", "crowdless", "not crowded", "less crowded", "avoid crowds", "not busy", "crowdlessplaces", "crowdlessplaceswhy"]
    if any(ind in question_lower for ind in uncrowded_indicators):
        is_uncrowded_query = True

    if "Crowdedness" in search_terms and is_uncrowded_query:
        # Acknowledge peacefulness is related with crowdedness: add Peacefulness as related feature
        search_terms = [t for t in search_terms if t != "Crowdedness"]
        if "Peacefulness" not in search_terms:
            search_terms.append("Peacefulness")

    graph_results = []
    if search_terms:
        where_clauses = " OR ".join(
            [f"toLower(f.name) CONTAINS toLower('{t}')" for t in search_terms[:5]]
        )
        if locations:
            loc_clauses = []
            for loc in locations:
                loc_clauses.append(f"toLower(d.name) CONTAINS toLower('{loc}') OR (pr IS NOT NULL AND toLower(pr.name) CONTAINS toLower('{loc}'))")
            location_filter = f" AND ({' OR '.join(loc_clauses)})"
            cypher = f"""
MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature)
MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (d)-[:PART_OF]->(pr:Province)
WITH p, r, f, d, pr
WHERE ({where_clauses}){location_filter} AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
RETURN DISTINCT p.name AS place, f.name AS feature,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..3] AS top_activities
ORDER BY r.percentage DESC
LIMIT {graph_limit}
"""
        else:
            cypher = f"""
MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature)
WHERE ({where_clauses}) AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
RETURN DISTINCT p.name AS place, f.name AS feature,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..3] AS top_activities
ORDER BY r.percentage DESC
LIMIT {graph_limit}
"""
        graph_results = _graph_query(cypher)
        logger.info(f"[RetrievalRouter] Feature search terms: {search_terms}")

    # Database-wide filter out of crowded places in _retrieve_feature if it's an uncrowded query
    if is_uncrowded_query:
        crowded_cypher = """
        MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature {name: 'Crowdedness'})
        WHERE r.sentiment = 'negative' AND r.percentage > 20.0
        RETURN p.name AS place
        """
        crowded_places = {r['place'] for r in _graph_query(crowded_cypher)}
        if crowded_places:
            graph_results = [
                fact for fact in graph_results
                if fact.get("place", fact.get("Place")) not in crowded_places
            ]

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)

    if len(graph_results) == 0 and dynamic_threshold < 5.0:
        logger.warning(f"[RetrievalRouter] Graph returned 0 results for rare terms {search_terms}. Triggering Vector Fallback.")
        fallback_query = " ".join(search_terms) + " Sri Lanka"
        vector_docs += _vector_search(fallback_query, k=5)

    return graph_results, vector_docs


def _retrieve_facility(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    C. facility_query — places with HAS_FACILITY relation.
    Uses vocabulary-resolved facility terms and multi-term WHERE clauses.
    Uses: HAS_FACILITY, LOCATED_IN, HAS_CATEGORY
    Ranked by: facility evidence percentage DESC
    """
    logger.info("[RetrievalRouter] Using HAS_FACILITY + vector review retrieval")
    if not settings.use_graph_retrieval:
        logger.info("[RetrievalRouter] Graph retrieval bypassed by config. Using Vector Fallback.")
        graph_results = []
        vec_query = build_vector_query(question, query_intent)
        vector_docs = _vector_search(vec_query, k=min(suggested_limit, 5))
        return graph_results, vector_docs

    raw_facilities = extract_all_facilities(query_intent)
    locations = extract_locations(query_intent)
    
    # Expand each resolved facility to possible synonyms for broader Cypher match
    all_search_terms = []
    for fac in raw_facilities:
        terms = get_search_terms_for_facility(fac)
        all_search_terms.extend(terms)
    seen = set()
    search_terms = [t for t in all_search_terms if not (t in seen or seen.add(t))]

    dynamic_threshold = get_dynamic_threshold(search_terms)

    graph_results = []
    if search_terms:
        where_clauses = " OR ".join(
            [f"toLower(fa.name) CONTAINS toLower('{t}')" for t in search_terms[:5]]
        )
        if locations:
            loc_clauses = []
            for loc in locations:
                loc_clauses.append(f"toLower(d.name) CONTAINS toLower('{loc}') OR (pr IS NOT NULL AND toLower(pr.name) CONTAINS toLower('{loc}'))")
            location_filter = f" AND ({' OR '.join(loc_clauses)})"
            cypher = f"""
MATCH (p:Place)-[r:HAS_FACILITY]->(fa:Facility)
MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (d)-[:PART_OF]->(pr:Province)
WITH p, r, fa, d, pr
WHERE ({where_clauses}){location_filter} AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
RETURN DISTINCT p.name AS place, fa.name AS facility,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..2] AS top_activities,
       collect(DISTINCT f.name)[0..2] AS top_features
ORDER BY r.percentage DESC
LIMIT {graph_limit}
"""
        else:
            cypher = f"""
MATCH (p:Place)-[r:HAS_FACILITY]->(fa:Facility)
WHERE ({where_clauses}) AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
RETURN DISTINCT p.name AS place, fa.name AS facility,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..2] AS top_activities,
       collect(DISTINCT f.name)[0..2] AS top_features
ORDER BY r.percentage DESC
LIMIT {graph_limit}
"""
        graph_results = _graph_query(cypher)
        logger.info(f"[RetrievalRouter] Facility search terms: {search_terms}")

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)

    if len(graph_results) == 0 and dynamic_threshold < 5.0:
        logger.warning(f"[RetrievalRouter] Graph returned 0 results for rare terms {search_terms}. Triggering Vector Fallback.")
        fallback_query = " ".join(search_terms) + " Sri Lanka"
        vector_docs += _vector_search(fallback_query, k=5)

    return graph_results, vector_docs


def _retrieve_place_attributes(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    D. place_attribute_query — features and activities of a specific place.
    Uses: HAS_FEATURE, HAS_ACTIVITY
    """
    logger.info("[RetrievalRouter] Using Place Attribute Retrieval")
    places = query_intent.get("entities", {}).get("places", [])
    
    if not places:
        # Fallback to general search if no place extracted
        return _retrieve_general(question, query_intent)
        
    place_name = places[0]
    
    # Cypher query to get features, activities, and facilities
    cypher = """
    MATCH (p:Place)
    WHERE toLower(p.name) CONTAINS toLower($place_name)
    OPTIONAL MATCH (p)-[r:HAS_FEATURE]->(f:Feature)
    OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
    OPTIONAL MATCH (p)-[rfa:HAS_FACILITY]->(fa:Facility)
    RETURN p.name AS place, 
           collect(distinct {name: f.name, sentiment: r.sentiment, percentage: r.percentage}) AS features,
           collect(distinct {name: a.name, sentiment: ra.sentiment, percentage: ra.percentage}) AS activities,
           collect(distinct {name: fa.name, sentiment: rfa.sentiment, percentage: rfa.percentage}) AS facilities
    """
    
    graph_results = _graph_query(cypher, {"place_name": place_name})
    
    # Also do a vector search for reviews about this place
    vec_query = f"reviews and features of {place_name}"
    vector_docs = _vector_search(vec_query)
    
    return graph_results, vector_docs


def _retrieve_dynamic_cypher(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    E. Dynamic Text-to-Cypher retrieval.
    Calls LLM to generate Cypher query based on question and schema.
    """
    logger.info("[RetrievalRouter] Using Dynamic Text-to-Cypher Retrieval")
    
    from src.agents.v2.prompts import CYPHER_GENERATOR_PROMPT
    from langchain_core.messages import HumanMessage
    
    try:
        response = llm.invoke([HumanMessage(content=CYPHER_GENERATOR_PROMPT + f"\n\nQuestion: {question}\nCypher:")])
        
        # Extract text from LangChain response
        content = response.content
        if isinstance(content, list):
            cypher_query = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            ).strip()
        else:
            cypher_query = str(content).strip()
        
        # Clean up code blocks if LLM ignored instructions
        if cypher_query.startswith("```"):
            parts = cypher_query.split("```")
            if len(parts) > 1:
                cypher_query = parts[1]
                if cypher_query.startswith("cypher"):
                    cypher_query = cypher_query[6:]
            cypher_query = cypher_query.strip()
            
        logger.info(f"[Researcher] Generated Cypher: {cypher_query}")
        
        graph_results = _graph_query(cypher_query)
        
    except Exception as e:
        logger.warning(f"[Researcher] Dynamic Cypher failed: {e}. Falling back to general search.")
        return _retrieve_general(question, query_intent)
        
    # Also do a vector search
    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    
    return graph_results, vector_docs


def _retrieve_location(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    C. location_query — places LOCATED_IN district / PART_OF province.
    Uses: LOCATED_IN, PART_OF, HAS_ACTIVITY, HAS_FEATURE, HAS_CATEGORY
    """
    logger.info("[RetrievalRouter] Using LOCATED_IN/PART_OF + vector review retrieval")
    locations = extract_locations(query_intent)
    primary = locations[0] if locations else ""

    graph_results = []
    if primary:
        # Try district match first
        cypher_district = f"""
MATCH (p:Place)-[:LOCATED_IN]->(d:District)
WHERE toLower(d.name) CONTAINS toLower($location)
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place, d.name AS district, c.name AS category,
       collect(DISTINCT a.name) AS activities,
       collect(DISTINCT f.name) AS features
LIMIT {graph_limit}
"""
        graph_results = _graph_query(cypher_district, {"location": primary})

        if not graph_results:
            # Fallback: try province match
            cypher_province = f"""
MATCH (p:Place)-[:LOCATED_IN]->(d:District)-[:PART_OF]->(pr:Province)
WHERE toLower(pr.name) CONTAINS toLower($location)
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place, d.name AS district, pr.name AS province,
       c.name AS category, collect(DISTINCT a.name) AS activities
LIMIT {graph_limit}
"""
            graph_results = _graph_query(cypher_province, {"location": primary})

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    return graph_results, vector_docs


def _retrieve_comparison(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    D. comparison_query — same evidence for all compared places.
    Uses: HAS_ACTIVITY, HAS_FEATURE, LOCATED_IN
    """
    logger.info("[RetrievalRouter] Using comparison retrieval for multiple places")
    places = extract_places(query_intent)

    graph_results = []
    if places:
        cypher = """
MATCH (p:Place)
WHERE any(placeName IN $places WHERE toLower(p.name) CONTAINS toLower(placeName))
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place,
       collect(DISTINCT {activity: a.name, sentiment: ra.sentiment, percentage: ra.percentage}) AS activities,
       collect(DISTINCT {feature: f.name, sentiment: rf.sentiment, percentage: rf.percentage}) AS features,
       d.name AS district, c.name AS category
"""
        graph_results = _graph_query(cypher, {"places": places})

    # Search reviews for each place
    vector_docs = []
    activities = extract_all_activities(query_intent)
    for place in places[:3]:
        q = f"reviews {place} Sri Lanka tourism"
        if activities:
            q += f" {activities[0]}"
        vector_docs += _vector_search(q, k=3)

    return graph_results, vector_docs





def _retrieve_constraint(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    E. constraint_query — places matching positive features; filter/penalise negatives.
    Uses vocabulary-resolved constraint terms for both positive and negative conditions.
    Uses: HAS_FEATURE, HAS_ACTIVITY, LOCATED_IN
    """
    logger.info("[RetrievalRouter] Using constraint-based retrieval")
    pos_raw = extract_positive_constraints(query_intent)
    neg_raw = extract_negative_constraints(query_intent)
    features = extract_all_features(query_intent)
    activities = extract_all_activities(query_intent)
    locations = extract_locations(query_intent)

    # Resolve through vocabulary
    pos_resolved, neg_resolved = resolve_constraint_features(pos_raw, neg_raw)
    all_positive_features = list(dict.fromkeys(features + pos_resolved))  # deduplicate

    dynamic_threshold = get_dynamic_threshold(all_positive_features)

    # Decouple Peacefulness and Crowdedness conflict while acknowledging they are related
    is_uncrowded = False
    question_lower = question.lower()
    uncrowded_indicators = ["uncrowded", "crowdless", "not crowded", "less crowded", "avoid crowds", "not busy", "crowdlessplaces", "crowdlessplaceswhy"]
    if any(ind in question_lower for ind in uncrowded_indicators):
        is_uncrowded = True

    if "Crowdedness" in all_positive_features and is_uncrowded:
        all_positive_features = [f for f in all_positive_features if f != "Crowdedness"]
        if "Crowdedness" not in neg_resolved:
            neg_resolved.append("Crowdedness")
        if "Peacefulness" not in all_positive_features:
            all_positive_features.append("Peacefulness")

    graph_results = []

    # Retrieve places matching positive features (multi-term)
    if all_positive_features:
        where_clauses = " OR ".join(
            [f"toLower(f.name) CONTAINS toLower('{t}')" for t in all_positive_features[:5]]
        )
        if locations:
            loc_clauses = []
            for loc in locations:
                loc_clauses.append(f"toLower(d.name) CONTAINS toLower('{loc}') OR (pr IS NOT NULL AND toLower(pr.name) CONTAINS toLower('{loc}'))")
            location_filter = f" AND ({' OR '.join(loc_clauses)})"
            cypher = f"""
MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature)
MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (d)-[:PART_OF]->(pr:Province)
WITH p, r, f, d, pr
WHERE ({where_clauses})
  AND r.sentiment = 'positive'{location_filter}
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place, f.name AS feature,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category
ORDER BY r.percentage DESC
LIMIT 12
"""
        else:
            cypher = f"""
MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature)
WHERE ({where_clauses})
  AND r.sentiment = 'positive'
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place, f.name AS feature,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district, c.name AS category
ORDER BY r.percentage DESC
LIMIT 12
"""
        graph_results += _graph_query(cypher)
        logger.info(f"[RetrievalRouter] Constraint positive features: {all_positive_features}")

    # Retrieve places matching positive activities
    for act in activities[:2]:
        terms = get_search_terms_for_activity(act)
        where_clauses = " OR ".join(
            [f"toLower(a.name) CONTAINS toLower('{t}')" for t in terms[:3]]
        )
        if locations:
            loc_clauses = []
            for loc in locations:
                loc_clauses.append(f"toLower(d.name) CONTAINS toLower('{loc}') OR (pr IS NOT NULL AND toLower(pr.name) CONTAINS toLower('{loc}'))")
            location_filter = f" AND ({' OR '.join(loc_clauses)})"
            cypher = f"""
MATCH (p:Place)-[r:HAS_ACTIVITY]->(a:Activity)
MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (d)-[:PART_OF]->(pr:Province)
WITH p, r, a, d, pr
WHERE ({where_clauses}){location_filter} AND r.sentiment <> 'negative'
RETURN p.name AS place, a.name AS activity,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district
ORDER BY r.percentage DESC
LIMIT 8
"""
        else:
            cypher = f"""
MATCH (p:Place)-[r:HAS_ACTIVITY]->(a:Activity)
WHERE ({where_clauses}) AND r.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN p.name AS place, a.name AS activity,
       r.sentiment AS sentiment, r.percentage AS percentage,
       d.name AS district
ORDER BY r.percentage DESC
LIMIT 8
"""
        graph_results += _graph_query(cypher)

    # Location filter
    for loc in locations[:1]:
        cypher = """
MATCH (p:Place)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (d)-[:PART_OF]->(pr:Province)
WITH p, d, pr
WHERE toLower(d.name) CONTAINS toLower($location) OR (pr IS NOT NULL AND toLower(pr.name) CONTAINS toLower($location))
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
RETURN p.name AS place, d.name AS district,
       f.name AS feature, rf.sentiment AS sentiment, rf.percentage AS percentage
LIMIT {graph_limit}
"""
        graph_results += _graph_query(cypher, {"location": loc})

    # Build ChromaDB query combining positive + negative context
    pos_str = " ".join(pos_raw[:3]) if pos_raw else ""
    neg_str = " not ".join(neg_raw[:2]) if neg_raw else ""
    vec_query = f"{pos_str} {neg_str} Sri Lanka tourist place reviews".strip()
    if not pos_str and not neg_str:
        vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)

    if len(graph_results) == 0 and dynamic_threshold < 5.0:
        logger.warning(f"[RetrievalRouter] Graph returned 0 results for rare constraints {all_positive_features}. Triggering Vector Fallback.")
        fallback_query = " ".join(all_positive_features) + " Sri Lanka"
        vector_docs += _vector_search(fallback_query, k=5)


    # Filter out places that violate negative constraints
    if neg_resolved:
        violated_places = set()
        
        # 1. Query Neo4j for ALL places violating the negative constraints
        neg_clauses = " OR ".join([f"toLower(f.name) CONTAINS toLower('{neg}')" for neg in neg_resolved])
        cypher_neg = f"""
        MATCH (p:Place)-[r:HAS_FEATURE]->(f:Feature)
        WHERE ({neg_clauses}) AND r.sentiment = 'negative'
        RETURN p.name AS place, r.percentage AS percentage
        """
        neg_results = _graph_query(cypher_neg)
        for r in neg_results:
            p_name = r.get("place")
            pct = r.get("percentage")
            # Filter if percentage is None (manual/implicit) or greater than 20%
            if p_name and (pct is None or pct > 20.0):
                violated_places.add(p_name)
                
        logger.info(f"[ConstraintFilter] Identified violated places violating {neg_resolved} (threshold > 20%): {violated_places}")
        
        # 2. Filter graph_results
        if violated_places:
            graph_results = [
                fact for fact in graph_results
                if fact.get("place", fact.get("Place")) not in violated_places
            ]
            
            # 3. Filter vector_docs to remove any chunks mentioning violated places
            filtered_docs = []
            for doc in vector_docs:
                content = getattr(doc, "page_content", str(doc))
                if not any(violated.lower() in content.lower() for violated in violated_places):
                    filtered_docs.append(doc)
                else:
                    logger.info(f"[ConstraintFilter] Filtering out vector doc mentioning violated place: {getattr(doc, 'metadata', {})}")
            vector_docs = filtered_docs

    return graph_results, vector_docs


def _retrieve_explanation(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    F. explanation_query — evidence for a mentioned place.
    Focus: why a place was recommended, structured facts + review evidence.
    """
    logger.info("[RetrievalRouter] Using explanation/evidence retrieval")
    places = extract_places(query_intent)
    activities = extract_all_activities(query_intent)
    features = extract_all_features(query_intent)
    primary_place = places[0] if places else ""

    graph_results = []
    if primary_place:
        cypher = """
MATCH (p:Place)
WHERE toLower(p.name) CONTAINS toLower($place)
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place,
       collect(DISTINCT {activity: a.name, sentiment: ra.sentiment, percentage: ra.percentage}) AS activities,
       collect(DISTINCT {feature: f.name, sentiment: rf.sentiment, percentage: rf.percentage}) AS features,
       d.name AS district, c.name AS category
"""
        graph_results = _graph_query(cypher, {"place": primary_place})

    # Search reviews focused on this place
    vec_parts = [primary_place] if primary_place else []
    vec_parts += activities[:1] + features[:1]
    vec_query = " ".join(vec_parts) + " Sri Lanka tourism review evidence"
    vector_docs = _vector_search(vec_query.strip())

    return graph_results, vector_docs


def _retrieve_general(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    G. general_query — hybrid: text-to-Cypher + general vector search.
    Falls back to the v1 researcher approach for full coverage.
    """
    logger.info("[RetrievalRouter] Using general hybrid retrieval (text-to-Cypher + vector)")
    cypher_prompt = f"""
You are an expert Neo4j Cypher query generator.
Generate a Cypher query to answer the user's question based on the schema below.

SCHEMA:
- Nodes: Place (name, category), District (name), Province (name),
         Category (name), Feature (name), Activity (name)
- Relationships:
  - (p:Place)-[:LOCATED_IN]->(d:District)
  - (d:District)-[:PART_OF]->(pr:Province)
  - (p:Place)-[:HAS_CATEGORY]->(c:Category)
  - (p:Place)-[:HAS_FEATURE]->(f:Feature)  [properties: sentiment, percentage]
  - (p:Place)-[:HAS_ACTIVITY]->(a:Activity) [properties: sentiment, percentage]

RULES:
1. Return ONLY an executable Cypher query. No markdown, no explanations.
2. If not answerable from this schema, return 'None'.
3. For multi-hop similarity, use shared nodes.

Question: {question}
"""
    graph_results = []
    try:
        resp = llm.invoke([HumanMessage(content=cypher_prompt)])
        cypher = resp.content
        if isinstance(cypher, list):
            cypher = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in cypher).strip()
        else:
            cypher = str(cypher).strip()

        # Strip markdown fences
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
            cypher = "\n".join(lines).strip()

        if cypher and cypher.lower() != "none" and "MATCH" in cypher.upper():
            logger.info(f"[Researcher] General Cypher: {cypher[:100]}")
            graph_results = _graph_query(cypher)
    except Exception as e:
        logger.error(f"[Researcher] General text-to-Cypher failed: {e}")

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    return graph_results, vector_docs


def _retrieve_guided_cypher(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    H. guided_cypher — uses LLM to generate Cypher guided by extracted entities.
    Supports multi-hop retrieval and complex combinations.
    """
    logger.info("[RetrievalRouter] Using guided Cypher retrieval")
    
    entities = query_intent.get("entities", {})
    activities = entities.get("activities", [])
    features = entities.get("features", [])
    facilities = entities.get("facilities", [])
    locations = entities.get("locations", [])
    
    # Format entities for the prompt
    activities_str = ", ".join([f"'{a}'" for a in activities]) if activities else "None"
    features_str = ", ".join([f"'{f}'" for f in features]) if features else "None"
    facilities_str = ", ".join([f"'{fa}'" for fa in facilities]) if facilities else "None"
    locations_str = ", ".join([f"'{l}'" for l in locations]) if locations else "None"
    
    prompt = GUIDED_CYPHER_PROMPT.format(
        question=question,
        activities=activities_str,
        features=features_str,
        facilities=facilities_str,
        locations=locations_str,
    )
    
    graph_results = []
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        cypher = resp.content
        if isinstance(cypher, list):
            cypher = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in cypher).strip()
        else:
            cypher = str(cypher).strip()

        # Strip markdown fences
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
            cypher = "\n".join(lines).strip()

        if cypher and cypher.lower() != "none" and "MATCH" in cypher.upper():
            logger.info(f"[Researcher] Guided Cypher: {cypher[:100]}")
            graph_results = _graph_query(cypher)
    except Exception as e:
        logger.error(f"[Researcher] Guided Cypher generation failed: {e}")

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    return graph_results, vector_docs


def _retrieve_multi_hop(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    I. multi_hop — performs multi-hop traversal to enrich context.
    Finds places matching criteria, then fetches ALL their activities and features.
    """
    logger.info("[RetrievalRouter] Using multi-hop traversal retrieval")
    
    entities = query_intent.get("entities", {})
    activities = entities.get("activities", [])
    features = entities.get("features", [])
    facilities = entities.get("facilities", [])
    locations = entities.get("locations", [])
    
    # Build match clauses for criteria
    match_clauses = []
    where_clauses = []
    
    if activities:
        match_clauses.append("(p:Place)-[:HAS_ACTIVITY]->(a:Activity)")
        where_clauses.append(" OR ".join([f"toLower(a.name) CONTAINS toLower('{act}')" for act in activities]))
        
    if features:
        match_clauses.append("(p:Place)-[:HAS_FEATURE]->(f:Feature)")
        where_clauses.append(" OR ".join([f"toLower(f.name) CONTAINS toLower('{feat}')" for feat in features]))
        
    if facilities:
        match_clauses.append("(p:Place)-[:HAS_FACILITY]->(fa:Facility)")
        where_clauses.append(" OR ".join([f"toLower(fa.name) CONTAINS toLower('{fac}')" for fac in facilities]))
        
    if locations:
        match_clauses.append("(p:Place)-[:LOCATED_IN]->(d:District)")
        where_clauses.append(" OR ".join([f"toLower(d.name) CONTAINS toLower('{loc}')" for loc in locations]))
        
    if not match_clauses:
        # Fallback to guided cypher if no entities extracted
        return _retrieve_guided_cypher(question, query_intent)
        
    # Combine clauses
    match_str = " MATCH ".join(match_clauses)
    where_str = " AND ".join([f"({wc})" for wc in where_clauses])
    
    cypher = f"""
MATCH {match_str}
WHERE {where_str}
WITH p
LIMIT {graph_limit}
OPTIONAL MATCH (p)-[r:HAS_ACTIVITY]->(all_a:Activity)
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(all_f:Feature)
OPTIONAL MATCH (p)-[rfa:HAS_FACILITY]->(all_fa:Facility)
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN p.name AS place,
       collect(DISTINCT {{activity: all_a.name, sentiment: r.sentiment, percentage: r.percentage}}) AS activities,
       collect(DISTINCT {{feature: all_f.name, sentiment: rf.sentiment, percentage: rf.percentage}}) AS features,
       collect(DISTINCT {{facility: all_fa.name, sentiment: rfa.sentiment, percentage: rfa.percentage}}) AS facilities,
       d.name AS district
"""
    
    logger.info(f"[Researcher] Multi-hop Cypher: {cypher}")
    graph_results = _graph_query(cypher)
    
    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    return graph_results, vector_docs


import json

# ─────────────────────────────────────────────────────────────────
# NODE 2.5 — KG Translator
# ─────────────────────────────────────────────────────────────────

def kg_translator_node(state: GraphState) -> dict:
    """Translates raw KG facts into natural language sentences."""
    logger.info("Executing KG TRANSLATOR node")
    
    structured_facts = state.get("structured_facts")
    if not structured_facts:
        return {"research_data": state.get("research_data", "")}
        
    prompt = KG_TRANSLATOR_PROMPT_TEMPLATE.format(
        question=state.get("question", ""),
        raw_facts=json.dumps(structured_facts, indent=2)
    )
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ).strip()
    else:
        content = str(content).strip()
        
    translated_sentences = safe_json_parse(content)
    if not translated_sentences or not isinstance(translated_sentences, list):
        logger.warning("[KG Translator] Failed to parse translated sentences as JSON array. Falling back.")
        return {"research_data": state.get("research_data", "")}

    import re
    research_data = state.get("research_data", "")

    for i, s in enumerate(translated_sentences, start=1):
        pattern = f"(Knowledge Graph Fact {i}\\..*?)(?=\\n   Supporting Evidence:|\\nKnowledge Graph Fact|\\nRETRIEVAL NOTES:|$)"
        research_data = re.sub(pattern, f"Knowledge Graph Fact {i}. {s}", research_data, flags=re.DOTALL)

    return {"research_data": research_data}


def _retrieve_nearby(question: str, query_intent: dict):
    """
    J. nearby_query — finds tourist places near a named anchor place.
    """
    from src.core.distance_matrix import DistanceMatrix

    suggested_limit = (
        query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8)
        if isinstance(query_intent, dict) else 8
    )
    top_n = min(int(suggested_limit), 10)

    places = (query_intent.get("entities") or {}).get("places", [])
    anchor = places[0] if places else ""

    # Extract attribute filters (e.g. "swimming places near X" or "scenic view near X")
    entities = query_intent.get("entities") or {}
    filter_activities = entities.get("activities", []) or []
    filter_features   = entities.get("features", [])   or []
    filter_categories = entities.get("categories", []) or []

    if not anchor:
        logger.warning("[NearbyRetrieval] No anchor place found in intent. Falling back to guided cypher.")
        return _retrieve_guided_cypher(question, query_intent)

    has_filter = bool(filter_activities or filter_features or filter_categories)
    dm = DistanceMatrix()

    if has_filter:
        logger.info(
            f"[NearbyRetrieval] Filter-First flow active for: "
            f"activities={filter_activities}, features={filter_features}, categories={filter_categories}"
        )
        # Step 1: Get anchor coordinates (lat, lon)
        anchor_coords = None
        exact_anchor = dm.get_exact_name(anchor)
        if exact_anchor and exact_anchor in dm.poi_coords:
            anchor_coords = dm.poi_coords[exact_anchor]
        else:
            anchor_coords = dm.geocode_place(anchor)

        if anchor_coords:
            anchor_lat, anchor_lon = anchor_coords
            logger.info(f"[NearbyRetrieval] Anchor '{anchor}' coords: ({anchor_lat:.4f}, {anchor_lon:.4f})")

            # Step 2: Query Neo4j for ALL places in Sri Lanka matching the requested feature/activity/category
            if filter_activities:
                acts = " OR ".join([f"toLower(a.name) CONTAINS toLower('{a}')" for a in filter_activities[:3]])
                cypher = f"""
MATCH (p:Place)-[ra:HAS_ACTIVITY]->(a:Activity)
WHERE ra.sentiment <> 'negative' AND ({acts})
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature) WHERE rf.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN DISTINCT p.name AS place, d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..3] AS activities, collect(DISTINCT f.name)[0..3] AS features
"""
            elif filter_features:
                feats = " OR ".join([f"toLower(f.name) CONTAINS toLower('{f}')" for f in filter_features[:3]])
                cypher = f"""
MATCH (p:Place)-[rf:HAS_FEATURE]->(f:Feature)
WHERE rf.sentiment <> 'negative' AND ({feats})
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity) WHERE ra.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN DISTINCT p.name AS place, d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..3] AS activities, collect(DISTINCT f.name)[0..3] AS features
"""
            else:
                cats = " OR ".join([f"toLower(c.name) CONTAINS toLower('{c}') OR toLower(p.name) CONTAINS toLower('{c}')" for c in filter_categories[:3]])
                cypher = f"""
MATCH (p:Place)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
WITH p, c
WHERE {cats}
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity) WHERE ra.sentiment <> 'negative'
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature) WHERE rf.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN DISTINCT p.name AS place, d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..3] AS activities, collect(DISTINCT f.name)[0..3] AS features
"""
            matching_places = _graph_query(cypher, {"name": anchor})

            # Step 3: Find nearest matching places from anchor coords using Haversine distance
            haversine_matches = []
            for row in matching_places:
                pname = row.get("place")
                if not pname or pname.lower() == anchor.lower():
                    continue
                poi_coords = dm.poi_coords.get(dm.get_exact_name(pname) or pname)
                if not poi_coords:
                    poi_coords = dm.poi_coords.get(pname)
                if poi_coords:
                    plat, plon = poi_coords
                    dist_km = dm._haversine(anchor_lat, anchor_lon, plat, plon)
                    haversine_matches.append((row, dist_km))

            haversine_matches.sort(key=lambda x: x[1])
            candidates = haversine_matches[:top_n]
            logger.info(f"[NearbyRetrieval] Top {len(candidates)} closest matching places by Haversine: {[c[0]['place'] for c in candidates]}")

            # Step 4: Get actual driving distances (matrix or OSRM) for those closest matching places
            graph_results = []
            for row, straight_km in candidates:
                pname = row["place"]
                exact_anchor_name = dm.get_exact_name(anchor)
                exact_poi_name = dm.get_exact_name(pname)
                road_km = None
                if exact_anchor_name and exact_poi_name and dm.df is not None:
                    try:
                        val = dm.df.loc[exact_anchor_name, exact_poi_name]
                        if val > 0:
                            road_km = float(val)
                    except Exception:
                        pass
                if road_km is None:
                    road_km = dm.get_dynamic_distance(anchor, pname)

                final_km = road_km if road_km is not None else straight_km * 1.2
                row["distance_km"] = round(final_km, 1)
                graph_results.append(row)

            graph_results.sort(key=lambda x: x.get("distance_km", 999.0))
            attr_str = " ".join(filter_activities + filter_features + filter_categories)
            vector_docs = _vector_search(f"{attr_str} near {anchor} Sri Lanka tourism".strip())
            return graph_results, vector_docs

    # ── Non-Filtered Proximity Search (e.g. "places near Matara") ────────────
    logger.info(f"[NearbyRetrieval] Finding places near: '{anchor}' (no attribute filter)")

    # ── Tier 1: Static CSV matrix
    nearby_pairs = dm.get_nearby_places(anchor, top_n=top_n)
    tier_used = "CSV matrix"

    # ── Tier 2: Nominatim + Haversine + OSRM (anchor not in CSV)
    if not nearby_pairs:
        logger.info(f"[NearbyRetrieval] '{anchor}' not in static CSV. Trying dynamic geocode + OSRM...")
        nearby_pairs = dm.get_nearby_for_unknown_place(
            anchor,
            top_n=top_n,
            haversine_candidates=15,
        )
        tier_used = "Nominatim+Haversine+OSRM"

    # ── Tier 3: Neo4j LOCATED_IN fallback
    if not nearby_pairs:
        logger.info(f"[NearbyRetrieval] Dynamic routing also failed. Falling back to Neo4j LOCATED_IN.")
        tier_used = "Neo4j LOCATED_IN"
        cypher = """
MATCH (anchor:Place)-[:LOCATED_IN]->(d:District)<-[:LOCATED_IN]-(nearby:Place)
WHERE toLower(anchor.name) CONTAINS toLower($anchor)
  AND toLower(nearby.name) <> toLower($anchor)
OPTIONAL MATCH (nearby)-[ra:HAS_ACTIVITY]->(a:Activity)
OPTIONAL MATCH (nearby)-[rf:HAS_FEATURE]->(f:Feature)
OPTIONAL MATCH (nearby)-[:HAS_CATEGORY]->(c:Category)
RETURN DISTINCT nearby.name AS place,
       d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..3] AS activities,
       collect(DISTINCT f.name)[0..3] AS features
ORDER BY nearby.name
LIMIT $limit
"""
        fallback_results = _graph_query(cypher, {"anchor": anchor, "limit": top_n})
        vec_query = f"places to visit near {anchor} Sri Lanka tourism"
        vector_docs = _vector_search(vec_query)
        logger.info(
            f"[NearbyRetrieval] LOCATED_IN fallback returned {len(fallback_results)} co-district places."
        )
        return fallback_results, vector_docs

    logger.info(
        f"[NearbyRetrieval] Found {len(nearby_pairs)} nearby places via {tier_used}: "
        f"{[p for p, _ in nearby_pairs]}"
    )

    # ── Enrich nearby places with Neo4j graph data in ONE query ──
    place_names = [p for p, _ in nearby_pairs[:top_n]]
    dist_map = {p: round(d, 1) for p, d in nearby_pairs[:top_n]}

    cypher = """
UNWIND $names AS name
OPTIONAL MATCH (p:Place)
WHERE toLower(p.name) CONTAINS toLower(name)
WITH name, collect(p)[0] AS p
WHERE p IS NOT NULL
OPTIONAL MATCH (p)-[ra:HAS_ACTIVITY]->(a:Activity) WHERE ra.sentiment <> 'negative'
OPTIONAL MATCH (p)-[rf:HAS_FEATURE]->(f:Feature)   WHERE rf.sentiment <> 'negative'
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN name AS query_name, p.name AS place,
       d.name AS district, c.name AS category,
       collect(DISTINCT a.name)[0..3] AS activities,
       collect(DISTINCT f.name)[0..3] AS features
"""
    results = _graph_query(cypher, {"names": place_names})
    
    enriched_map = {}
    for r in results:
        qname = r.pop("query_name")
        if qname not in enriched_map:
            enriched_map[qname] = r
            
    graph_results = []
    for p in place_names:
        res = enriched_map.get(p, {"place": p})
        res["distance_km"] = dist_map[p]
        graph_results.append(res)

    vec_query = f"places to visit near {anchor} Sri Lanka tourism"
    vector_docs = _vector_search(vec_query)

    return graph_results, vector_docs


# ─────────────────────────────────────────────────────────────────
# NODE 3 — Writer
# ─────────────────────────────────────────────────────────────────


def writer_node(state: GraphState) -> dict:
    """Intent-aware writer. Uses query_intent to match answer style to intent type."""
    logger.info("Executing WRITER node")
    question = state.get("question", "")
    research_data = state.get("research_data", "")
    feedback = state.get("feedback", "")
    query_intent = state.get("query_intent") or build_fallback_intent()

    intent_summary = build_intent_summary(query_intent)

    wants_percentage = any(w in question.lower() for w in ["percent", "score", "statistic", "rating"])
    if wants_percentage:
        rule_percentage = "- CRITICAL EXCEPTION: The user specifically asked for percentages or statistics. You MUST include exact percentages (e.g., '71.2%') from the Raw Data when describing places."
    else:
        rule_percentage = "- CRITICAL: NEVER, UNDER ANY CIRCUMSTANCES, OUTPUT PERCENTAGES. DO NOT USE THE '%' SYMBOL.\n- CRITICAL: Do NOT say phrases like 'positively mentioned' or 'with positive sentiment' for every single place. Only mention sentiment specifically when needed (e.g., contrasting viewpoints or specific review ratings). Otherwise, describe each place directly and naturally using its specific features, activities, and reasons from the evidence without appending sentiment phrases."

    prompt = WRITER_PROMPT_TEMPLATE.format(
        question=question,
        research_data=research_data,
        intent_summary=intent_summary,
        percentage_rule=rule_percentage
    )
    if feedback:
        prompt += WRITER_FEEDBACK_TEMPLATE.format(feedback=feedback)

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        ).strip()
    else:
        content = str(content).strip()
    return {"draft": content}


# ─────────────────────────────────────────────────────────────────
# NODE 4 — Retrieval-Augmented Fact Checker
# ─────────────────────────────────────────────────────────────────

def fact_checker_node(state: GraphState) -> dict:
    """
    Monolithic Fact Checker.
    Takes the entire draft and entire research data and checks for hallucinations
    in a single LLM call.
    """
    logger.info("Executing FACT CHECKER node")
    draft = state.get("draft", "")
    research_data = state.get("research_data", "")
    query_intent = state.get("query_intent") or build_fallback_intent()
    iterations = state.get("iterations", 0) + 1
    intent_type = query_intent.get("intent_type", "general_query")
    question = state.get("question", "")

    prompt = FACT_CHECKER_PROMPT_TEMPLATE.format(
        research_data=research_data,
        draft=draft,
        intent_type=intent_type,
        question=question,
    )
    structured_llm = llm.with_structured_output(FactCheckOutput)
    result = structured_llm.invoke([HumanMessage(content=prompt)])

    if iterations >= 2:
        logger.warning("Max iterations reached. Forcing approval.")
        result.is_approved = True

    if result.is_approved:
        logger.info("Draft APPROVED by Fact Checker.")
    else:
        logger.warning(f"Draft REJECTED. Feedback: {result.feedback}")

    return {
        "feedback": result.feedback,
        "iterations": iterations,
        "is_approved": result.is_approved,
    }


# ─────────────────────────────────────────────────────────────────
# NODE 5 — Evidence Formatter
# ─────────────────────────────────────────────────────────────────

def evidence_formatter_node(state: GraphState) -> dict:
    """
    Intent-aware evidence formatter. Formats the approved draft with
    evidence badges and caution notes based on intent type.

    - activity_query    : shows activity evidence, sentiment, percentage
    - feature_query     : shows feature evidence, positive/negative note
    - location_query    : shows location relationship and category
    - comparison_query  : side-by-side evidence per place
    - constraint_query  : supporting + cautionary evidence
    - explanation_query : focuses on why recommendations are supported
    - general_query     : standard evidence badge formatting
    """
    logger.info("Executing EVIDENCE FORMATTER node")
    draft = state.get("draft", "")
    query_intent = state.get("query_intent") or {}
    intent_type = query_intent.get("intent_type", "general_query")
    structured_facts = state.get("structured_facts", [])

    try:
        intent_instructions = {
            "activity_query": (
                "Ensure the text mentions the activity and supporting details naturally in the prose. "
                "Do NOT use bracketed tags."
            ),
            "feature_query": (
                "Ensure the text mentions the feature naturally. "
                "Do NOT use bracketed tags."
            ),
            "location_query": (
                "Ensure the text naturally mentions the district and category. "
                "Do NOT use bracketed tags or append raw data at the end of the sentence."
            ),
            "comparison_query": (
                "Ensure the text clearly compares the places using the evidence. "
                "Do NOT use bracketed tags."
            ),
            "constraint_query": (
                "Ensure the text naturally mentions supporting evidence and cautionary notes for negative constraints. "
                "Do NOT use bracketed tags."
            ),
            "explanation_query": (
                "Focus on WHY the recommendation is supported by naturally mentioning the graph relation and relevant details. "
                "Do NOT use bracketed tags."
            ),
            "general_query": (
                "Ensure the text accurately reflects the structured facts. "
                "Do NOT use bracketed tags."
            ),
        }

        intent_instruction = intent_instructions.get(
            intent_type, intent_instructions["general_query"]
        )
        
        question = state.get("question", "")
        wants_percentage = any(w in question.lower() for w in ["percent", "score", "statistic", "rating"])
        
        if wants_percentage:
            rule_percentage = "- Append or weave the exact percentage facts naturally into the sentences (e.g., 'with 71.2% positive sentiment')."
        else:
            rule_percentage = "- CRITICAL: NEVER, UNDER ANY CIRCUMSTANCES, OUTPUT PERCENTAGES (e.g., '71.2%'). DO NOT USE THE '%' SYMBOL.\n- CRITICAL: Do NOT say phrases like 'positively mentioned' or 'positive sentiment' for every single place. Only mention sentiment specifically when needed (such as contrasting review opinions or specific ratings). Otherwise, keep the place description natural and direct without appending sentiment phrases."

        evidence_prompt = f"""You are an evidence annotator for a tourism recommendation system.

I will give you an approved answer draft and structured graph facts.
Your task is to rewrite the draft to naturally include specific supporting details provided in the structured facts for each place.

Intent Type: {intent_type}
Formatting Instruction: {intent_instruction}

Rules:
- Weave supporting details naturally into the sentences without robotically adding 'positively mentioned' to every item.
- ONLY use facts mentioned in the STRUCTURED FACTS or existing draft. Do NOT add irrelevant adjectives unless they are specifically part of the requested activity/feature.
- CRITICAL GROUNDING RULE: Never invent descriptive adjectives, summaries, or assumptions that are not explicitly written in the evidence. Do not call a trail "challenging", "easy", or "scenic" unless the evidence explicitly uses those exact words. Describe ONLY what is strictly and literally stated in the retrieved facts and visitor reviews.
{rule_percentage}
- Do NOT add any raw bracketed tags like [[LOCATION:...]] or [[ACTIVITY EVIDENCE:...]].
- CRITICAL: You MUST preserve all specific details, tips, place names, and actionable advice from the original draft (e.g., "take a tuk tuk", "Main Point"). Do NOT summarize or delete these details.
- STRICT MARKDOWN FORMATTING: Output clean GitHub-Flavored Markdown. IF the user explicitly asks for a table, output a Markdown table. OTHERWISE: For MULTI-PLACE recommendations, use `- **Place Name** — reason.` format. For SINGLE-PLACE queries (place_attribute_query), do NOT repeat the place name on every bullet — just use simple `- description` bullets. Do NOT use plain Unicode bullets (`•`).
- If no facts match a sentence, leave that sentence unchanged.
- Make the output user-friendly, conversational, and easy to read.

STRUCTURED FACTS:
{structured_facts}

DRAFT TO ANNOTATE:
{draft}
"""
        response = llm.invoke([HumanMessage(content=evidence_prompt)])
        content = response.content

        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            ).strip()
        else:
            content = str(content).strip()

        logger.info("Evidence formatting complete.")
        return {"evidenced_response": content}

    except Exception as e:
        logger.error(f"Evidence formatting failed: {e}. Falling back to plain draft.")
        return {"evidenced_response": draft}


# ─────────────────────────────────────────────────────────────────
# POI Profile Retrieval Strategies (New nodes: TimeProfile, CrowdProfile, CostProfile)
# ─────────────────────────────────────────────────────────────────

def _retrieve_timing(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    timing_query — queries TimeProfile nodes linked via HAS_BEST_TIME.
    Returns best time of day, season, and months to visit.
    Filters by specific place name if provided; otherwise returns top confident entries.
    """
    from src.agents.v2.graph_vocabulary import resolve_time_of_day, resolve_season
    logger.info("[RetrievalRouter] Using HAS_BEST_TIME + vector review retrieval")

    places = query_intent.get("entities", {}).get("places", [])
    locations = query_intent.get("entities", {}).get("locations", [])
    constraints = query_intent.get("constraints", {})
    time_hint = constraints.get("time_or_season", "")

    graph_results = []

    if places:
        # Specific place(s) asked — return their timing profiles
        place_filter = " OR ".join(
            [f"toLower(p.name) CONTAINS toLower('{pl}')" for pl in places[:3]]
        )
        cypher = f"""
MATCH (p:Place)-[:HAS_BEST_TIME]->(t:TimeProfile)
WHERE ({place_filter}) AND t.time_of_day IS NOT NULL
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN p.name AS place,
       t.time_of_day AS time_of_day,
       t.season AS season,
       t.months AS months,
       t.avoid AS avoid,
       t.confidence AS confidence,
       t.based_on AS based_on,
       d.name AS district
ORDER BY t.confidence DESC
LIMIT {graph_limit}
"""
    elif locations:
        loc_filter = " OR ".join(
            [f"toLower(d.name) CONTAINS toLower('{loc}')" for loc in locations[:2]]
        )
        cypher = f"""
MATCH (p:Place)-[:HAS_BEST_TIME]->(t:TimeProfile)
MATCH (p)-[:LOCATED_IN]->(d:District)
WHERE ({loc_filter}) AND t.time_of_day IS NOT NULL AND t.confidence >= 0.5
RETURN p.name AS place,
       t.time_of_day AS time_of_day,
       t.season AS season,
       t.months AS months,
       t.confidence AS confidence,
       d.name AS district
ORDER BY t.confidence DESC
LIMIT {graph_limit}
"""
    else:
        # General — return high-confidence timing profiles
        cypher = """
MATCH (p:Place)-[:HAS_BEST_TIME]->(t:TimeProfile)
WHERE t.time_of_day IS NOT NULL AND t.confidence >= 0.7
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN p.name AS place,
       t.time_of_day AS time_of_day,
       t.season AS season,
       t.months AS months,
       t.confidence AS confidence,
       d.name AS district
ORDER BY t.confidence DESC
LIMIT 12
"""
    graph_results = _graph_query(cypher)

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    return graph_results, vector_docs


def _retrieve_crowd(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    crowd_query — queries CrowdProfile nodes linked via HAS_CROWD_PROFILE.
    Detects whether the user wants quiet (EMPTY/QUIET) or lively (BUSY/PACKED) places
    and filters accordingly.
    """
    from src.agents.v2.graph_vocabulary import resolve_crowd_label, get_crowd_level_range
    logger.info("[RetrievalRouter] Using HAS_CROWD_PROFILE + vector review retrieval")

    places = query_intent.get("entities", {}).get("places", [])
    locations = query_intent.get("entities", {}).get("locations", [])
    constraints = query_intent.get("constraints", {})
    neg_constraints = constraints.get("negative", [])
    pos_constraints = constraints.get("positive", [])

    question_lower = question.lower()
    wants_quiet = any(w in question_lower for w in [
        "not crowded", "uncrowded", "quiet", "peaceful", "avoid crowds",
        "less busy", "not busy", "empty", "serene", "tranquil"
    ])
    wants_busy = any(w in question_lower for w in [
        "busy", "packed", "popular", "lively", "crowded", "vibrant"
    ])

    # Default: show quiet places if not explicitly asking for busy
    if wants_busy and not wants_quiet:
        crowd_filter = "cr.level >= 4"  # BUSY or PACKED
        order = "cr.level DESC"
    else:
        crowd_filter = "cr.level <= 2"  # EMPTY or QUIET
        order = "cr.level ASC"

    graph_results = []

    if places:
        place_filter = " OR ".join(
            [f"toLower(p.name) CONTAINS toLower('{pl}')" for pl in places[:3]]
        )
        cypher = f"""
MATCH (p:Place)-[:HAS_CROWD_PROFILE]->(cr:CrowdProfile)
WHERE ({place_filter})
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN p.name AS place,
       cr.level AS level,
       cr.label AS crowd_label,
       cr.avg_level AS avg_level,
       cr.busiest_period AS busiest_period,
       cr.confidence AS confidence,
       d.name AS district
ORDER BY {order}
LIMIT {graph_limit}
"""
    elif locations:
        loc_filter = " OR ".join(
            [f"toLower(d.name) CONTAINS toLower('{loc}')" for loc in locations[:2]]
        )
        cypher = f"""
MATCH (p:Place)-[:HAS_CROWD_PROFILE]->(cr:CrowdProfile)
MATCH (p)-[:LOCATED_IN]->(d:District)
WHERE ({loc_filter}) AND {crowd_filter}
RETURN p.name AS place,
       cr.level AS level,
       cr.label AS crowd_label,
       cr.busiest_period AS busiest_period,
       cr.confidence AS confidence,
       d.name AS district
ORDER BY {order}
LIMIT {graph_limit}
"""
    else:
        cypher = f"""
MATCH (p:Place)-[:HAS_CROWD_PROFILE]->(cr:CrowdProfile)
WHERE {crowd_filter} AND cr.confidence >= 0.5
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place,
       cr.level AS level,
       cr.label AS crowd_label,
       cr.busiest_period AS busiest_period,
       cr.confidence AS confidence,
       d.name AS district,
       c.name AS category
ORDER BY {order}
LIMIT 12
"""
    graph_results = _graph_query(cypher)

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    return graph_results, vector_docs


def _retrieve_cost(question: str, query_intent: dict):
    suggested_limit = query_intent.get("retrieval_strategy", {}).get("suggested_retrieval_limit", 8) if isinstance(query_intent, dict) else 8
    graph_limit = min(int(suggested_limit), 10)
    """
    cost_query — queries CostProfile nodes linked via HAS_COST.
    Detects whether user wants cheap/free/expensive and filters cost level.
    Returns median_lkr, fee_type, and cost level.
    """
    from src.agents.v2.graph_vocabulary import resolve_cost_level
    logger.info("[RetrievalRouter] Using HAS_COST + vector review retrieval")

    places = query_intent.get("entities", {}).get("places", [])
    locations = query_intent.get("entities", {}).get("locations", [])
    constraints = query_intent.get("constraints", {})
    budget = constraints.get("budget", "")

    question_lower = question.lower()

    # Determine cost filter from question keywords + budget constraint
    resolved_level = None
    if budget:
        resolved_level = resolve_cost_level(budget)
    if not resolved_level:
        if any(w in question_lower for w in ["free", "no fee", "no entry fee", "no charge"]):
            resolved_level = "FREE"
        elif any(w in question_lower for w in ["cheap", "budget", "affordable", "inexpensive", "low cost"]):
            resolved_level = "LOW"
        elif any(w in question_lower for w in ["expensive", "luxury", "high end"]):
            resolved_level = "HIGH"

    graph_results = []

    if places:
        place_filter = " OR ".join(
            [f"toLower(p.name) CONTAINS toLower('{pl}')" for pl in places[:3]]
        )
        cypher = f"""
MATCH (p:Place)-[:HAS_COST]->(co:CostProfile)
WHERE ({place_filter})
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
RETURN p.name AS place,
       co.level AS cost_level,
       co.median_lkr AS median_lkr,
       co.fee_type AS fee_type,
       co.confidence AS confidence,
       d.name AS district
ORDER BY co.median_lkr ASC
LIMIT {graph_limit}
"""
    elif resolved_level:
        loc_clause = ""
        if locations:
            loc_parts = [f"toLower(d.name) CONTAINS toLower('{loc}')" for loc in locations[:2]]
            loc_clause = f" AND ({' OR '.join(loc_parts)})"
            cypher = f"""
MATCH (p:Place)-[:HAS_COST]->(co:CostProfile)
MATCH (p)-[:LOCATED_IN]->(d:District)
WHERE co.level = '{resolved_level}'{loc_clause}
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place,
       co.level AS cost_level,
       co.median_lkr AS median_lkr,
       co.fee_type AS fee_type,
       co.confidence AS confidence,
       d.name AS district,
       c.name AS category
ORDER BY co.median_lkr ASC
LIMIT 12
"""
        else:
            cypher = f"""
MATCH (p:Place)-[:HAS_COST]->(co:CostProfile)
WHERE co.level = '{resolved_level}' AND co.confidence >= 0.3
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place,
       co.level AS cost_level,
       co.median_lkr AS median_lkr,
       co.fee_type AS fee_type,
       co.confidence AS confidence,
       d.name AS district,
       c.name AS category
ORDER BY co.median_lkr ASC
LIMIT 12
"""
        graph_results = _graph_query(cypher)
    else:
        # General — return all with cost info ordered cheapest first
        cypher = """
MATCH (p:Place)-[:HAS_COST]->(co:CostProfile)
WHERE co.confidence >= 0.3
OPTIONAL MATCH (p)-[:LOCATED_IN]->(d:District)
OPTIONAL MATCH (p)-[:HAS_CATEGORY]->(c:Category)
RETURN p.name AS place,
       co.level AS cost_level,
       co.median_lkr AS median_lkr,
       co.fee_type AS fee_type,
       co.confidence AS confidence,
       d.name AS district,
       c.name AS category
ORDER BY co.median_lkr ASC
LIMIT 12
"""
        graph_results = _graph_query(cypher)

    vec_query = build_vector_query(question, query_intent)
    vector_docs = _vector_search(vec_query)
    return graph_results, vector_docs


# ─────────────────────────────────────────────────────────────────
# NODE 6 — Itinerary Formatter (MOIP)
# ─────────────────────────────────────────────────────────────────

def itinerary_formatter_node(state: GraphState) -> dict:
    """
    Takes the raw ACO multi-objective itinerary planning result
    and formats it nicely into a conversational response.
    """
    logger.info("Executing ITINERARY FORMATTER node")
    moip_result = state.get("moip_result", {})
    question = state.get("question", "")

    if not moip_result:
        return {"evidenced_response": "I'm sorry, I couldn't generate an itinerary for you based on the inputs."}

    best_route = moip_result.get("best", {})
    itinerary = moip_result.get("itinerary", [])
    
    if not best_route or not itinerary:
        return {"evidenced_response": "I'm sorry, no feasible routes were found for those constraints."}

    try:
        # 1. Fetch rich Neo4j data for the places
        all_places = []
        for day in itinerary:
            if "detailed_places" in day:
                all_places.extend([p["name"] for p in day["detailed_places"]])
            else:
                all_places.extend(day.get("places", []))
                
        all_places = list(dict.fromkeys(all_places))
        
        place_details = {}
        if all_places:
            place_filter = " OR ".join([f"toLower(p.name) CONTAINS toLower('{pl}')" for pl in all_places])
            cypher = f"""
            MATCH (p:Place)
            WHERE {place_filter}
            OPTIONAL MATCH (p)-[:HAS_FEATURE]->(f:Feature)
            OPTIONAL MATCH (p)-[:HAS_ACTIVITY]->(a:Activity)
            RETURN p.name AS place,
                   p.description AS description,
                   collect(DISTINCT f.name) AS features,
                   collect(DISTINCT a.name) AS activities
            """
            graph_results = _graph_query(cypher)
            
            # Map results by place name
            for r in graph_results:
                place_name = r.get("place")
                if place_name:
                    place_details[place_name.lower()] = {
                        "description": r.get("description") or "",
                        "features": r.get("features") or [],
                        "activities": r.get("activities") or []
                    }
                    
        # 2. Enrich the frontend moip_result object and build the prompt
        prompt = f"""You are an enthusiastic and professional travel planner for Sri Lanka.
I will provide you with the raw output of a highly optimized multi-objective itinerary planning algorithm.
Your task is to present this itinerary to the user in a very friendly, readable, and exciting way.

User Question/Context:
{question}

Algorithm Output (Best Route Details):
- Total Places: {len(best_route.get("route", []))}
- Total Time (mins): {best_route.get("total_time_min", 0)}
- Total Distance (km): {best_route.get("total_distance_km", 0)}
- Estimated Cost (LKR): {best_route.get("total_cost", 0)}
- Satisfaction Score: {best_route.get("total_satisfaction", 0)}
- Reason for this route: {best_route.get("explanation", "")}

Day-by-Day Breakdown:
"""
        for day in itinerary:
            places = " -> ".join(day.get("places", []))
            prompt += f"\nDay {day.get('day')}: {places} (Est. Time: {round(day.get('time_min', 0)/60, 1)} hrs, Distance: {round(day.get('distance_km', 0), 1)} km)\n"
            
            # Also attach features and activities into moip_result for the frontend UI map
            if "detailed_places" in day:
                for dp in day["detailed_places"]:
                    name_lower = dp["name"].lower()
                    matched_key = next((k for k in place_details.keys() if name_lower in k or k in name_lower), None)
                    if matched_key:
                        dp["description"] = place_details[matched_key]["description"]
                        dp["features"] = place_details[matched_key]["features"]
                        dp["activities"] = place_details[matched_key]["activities"]
                        
                        # Add snippet to prompt so LLM writes about it
                        if dp["features"] or dp["activities"]:
                            feats = ", ".join(dp["features"][:3])
                            acts = ", ".join(dp["activities"][:3])
                            prompt += f"  * {dp['name']}: {dp['description'][:100]}... Features: {feats}. Activities: {acts}.\n"

        prompt += """
Please draft a highly engaging and visually appealing conversational response explaining this itinerary to the user.

FORMATTING RULES (CRITICAL):
1. Use lots of emojis to make it lively (e.g. 🌴, 🚗, 🏖️, ⛰️).
2. Use Markdown heavily (bolding for place names, italics for descriptions).
3. Use short paragraphs and clear line breaks (leave an empty line between days).
4. Do NOT write a giant block of text. Break it up!
5. Start with a warm, exciting intro paragraph.
6. For each day, use a clear header (e.g. "### Day 1: [Theme]") and use bullet points for the places.
7. End with a bold summary of Total Distance and Total Time.
8. Do NOT output raw JSON, only the final conversational message.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()
        else:
            content = str(content).strip()
            
        return {"evidenced_response": content, "moip_result": moip_result}
        
    except Exception as e:
        logger.error(f"Itinerary formatting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"evidenced_response": "Here is your itinerary plan, though I had trouble formatting it properly.", "moip_result": moip_result}
