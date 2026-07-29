"""
src/agents/v2/graph_vocabulary.py
===================================
Semantic vocabulary maps for bridging the gap between:
  - User natural language terms (e.g. "quiet", "surf", "hike")
  - Exact graph node names in Neo4j (e.g. "Peacefulness", "Surfing", "Hiking")

Why this matters:
    The LLM classifier extracts user intent in natural language. Neo4j CONTAINS
    searches do partial matching, but a user saying "peaceful" won't match the
    graph node "Peacefulness" and "whale watching" may not match "Whale watching".
    This vocabulary layer normalises user terms to graph-known names BEFORE
    building Cypher queries, dramatically improving retrieval recall.

How to maintain:
    Add new entries whenever you add new Activity or Feature nodes to the graph.
    Values are lists because one user term may match multiple graph node names.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Activity Vocabulary
# Maps user synonyms → exact Activity node names in Neo4j
# Source: popular_activities_by_place.csv
# ──────────────────────────────────────────────────────────────────────────────

ACTIVITY_SYNONYMS: dict[str, list[str]] = {
    # Surfing
    "surf": ["Surfing"],
    "surfing": ["Surfing"],
    "surfer": ["Surfing"],
    "waves": ["Surfing"],

    # Hiking / Trekking
    "hike": ["Hiking", "Trekking"],
    "hiking": ["Hiking", "Trekking"],
    "trek": ["Trekking", "Hiking"],
    "trekking": ["Trekking", "Hiking"],
    "walking trail": ["Hiking", "Trekking"],
    "nature walk": ["Hiking", "Trekking"],

    # Swimming
    "swim": ["Swimming"],
    "swimming": ["Swimming"],
    "snorkel": ["Snorkelling"],
    "snorkelling": ["Snorkelling"],
    "snorkeling": ["Snorkelling"],

    # Safari / Wildlife
    "safari": ["Safari", "Jeep safari", "Boat safari"],
    "jeep safari": ["Jeep safari"],
    "game drive": ["Safari", "Jeep safari"],
    "wildlife watching": ["Safari", "Elephant watching"],
    "elephant watching": ["Elephant watching"],
    "whale watching": ["Whale watching"],
    "bird watching": ["Bird watching"],
    "birdwatching": ["Bird watching"],
    "birding": ["Bird watching"],
    "turtle watching": ["Turtle watching"],

    # Photography
    "photography": ["Photography"],
    "photo": ["Photography"],
    "photos": ["Photography"],
    "instagram": ["Photography"],

    # Boat
    "boat ride": ["Boat ride", "Boat safari"],
    "boat safari": ["Boat safari"],
    "boat trip": ["Boat ride"],
    "river cruise": ["Boat ride"],

    # Relaxing
    "relax": ["Relaxing"],
    "relaxing": ["Relaxing"],
    "chill": ["Relaxing"],
    "unwind": ["Relaxing"],
    "rest": ["Relaxing"],
    "sunbathe": ["Relaxing"],
    "sunbathing": ["Relaxing"],

    # Waterfall
    "waterfall visit": ["Waterfall visit"],
    "waterfall": ["Waterfall visit"],
    "see waterfall": ["Waterfall visit"],

    # Sunset / Sunrise
    "sunset": ["Sunset watching"],
    "sunset watching": ["Sunset watching"],
    "sunrise": ["Sunrise watching"],
    "sunrise watching": ["Sunrise watching"],

    # Shopping
    "shopping": ["Shopping"],
    "shop": ["Shopping"],
    "buy souvenirs": ["Shopping"],

    # Cultural / Religious
    "temple visit": ["Temple visit"],
    "religious visit": ["Religious visit"],
    "pilgrimage": ["Pilgrimage"],
    "meditation": ["Meditation"],

    # Ancient city
    "ancient city": ["Ancient city visit"],
    "ruins": ["Ruins exploration", "Ancient city visit"],
    "explore ruins": ["Ruins exploration"],

    # Viewpoint
    "viewpoint": ["Viewpoint visit"],
    "view point": ["Viewpoint visit"],
    "lookout": ["Viewpoint visit"],

    # Other
    "camping": ["Camping"],
    "ziplining": ["Ziplining"],
    "train ride": ["Train ride"],
    "tea plantation visit": ["Tea plantation visit", "Tea factory visit"],
    "tea tasting": ["Tea tasting"],
    "seafood tasting": ["Seafood tasting"],
    "street food": ["Street food tasting"],
    "spa": ["Spa treatment"],
    "fishing": ["Fishing"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature Vocabulary
# Maps user synonyms → exact Feature node names in Neo4j
# Source: popular_features_by_place.csv
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_SYNONYMS: dict[str, list[str]] = {
    # Peaceful / Quiet
    "peaceful": ["Peacefulness"],
    "peacefulness": ["Peacefulness"],
    "quiet": ["Peacefulness"],
    "calm": ["Peacefulness"],
    "serene": ["Peacefulness"],
    "tranquil": ["Peacefulness"],

    # Scenic
    "scenic": ["Scenic view"],
    "scenic view": ["Scenic view"],
    "beautiful view": ["Scenic view"],
    "nice view": ["Scenic view"],
    "viewpoint": ["Viewpoint"],
    "panoramic": ["Scenic view"],
    "landscape": ["Scenic view"],

    # Sunset / Sunrise
    "sunset": ["Sunset view"],
    "sunset view": ["Sunset view"],
    "sunrise": ["Sunrise view"],
    "sunrise view": ["Sunrise view"],

    # Clean
    "clean": ["Cleanliness"],
    "cleanliness": ["Cleanliness"],
    "hygienic": ["Cleanliness"],
    "well maintained": ["Cleanliness"],

    # Crowded
    "crowded": ["Crowdedness"],
    "crowdedness": ["Crowdedness"],
    "busy": ["Crowdedness"],
    "not crowded": ["Crowdedness"],
    "uncrowded": ["Crowdedness"],
    "crowdless": ["Crowdedness"],
    "crowdless places": ["Crowdedness"],
    "crowdlessplaces": ["Crowdedness"],
    "crowdlessplaceswhy": ["Crowdedness"],
    "not busy": ["Crowdedness"],
    "less crowded": ["Crowdedness"],
    "avoid crowds": ["Crowdedness"],

    # Family
    "family friendly": ["Family friendliness"],
    "family friendliness": ["Family friendliness"],
    "family": ["Family friendliness"],
    "kids": ["Family friendliness"],
    "child friendly": ["Family friendliness"],
    "safe for children": ["Family friendliness", "Safety"],

    # Safe
    "safe": ["Safety"],
    "safety": ["Safety"],
    "danger": ["Safety"],
    "dangerous": ["Safety"],

    # Beach
    "sandy beach": ["Sandy beach"],
    "beach": ["Beach"],
    "beachfront": ["Beach"],

    # Waterfall
    "waterfall": ["Waterfall"],

    # Mountain
    "mountain": ["Mountain"],
    "mountains": ["Mountain"],
    "hill": ["Mountain"],
    "hills": ["Mountain"],

    # Wildlife
    "wildlife": ["Wildlife"],
    "animals": ["Wildlife"],
    "elephants": ["Elephants"],
    "birds": ["Birds"],
    "turtles": ["Turtles"],
    "leopards": ["Leopards"],

    # Temple / Religious
    "temple": ["Temple"],
    "stupa": ["Stupa"],
    "buddha statue": ["Buddha statue"],
    "sculptures": ["Sculptures"],
    "paintings": ["Paintings"],
    "murals": ["Murals"],

    # Ancient
    "ancient ruins": ["Ancient ruins"],
    "ruins": ["Ancient ruins"],
    "heritage": ["Heritage site"],
    "heritage site": ["Heritage site"],
    "fort": ["Fort"],

    # Nature
    "forest": ["Forest"],
    "lake": ["Lake"],
    "river": ["River"],
    "lagoon": ["Lagoon"],
    "sea": ["Sea"],
    "waves": ["Waves"],

    # Food
    "restaurants": ["Restaurants"],
    "food": ["Food quality"],
    "cafes": ["Cafes"],

    # Accommodation
    "accommodation": ["Accommodation"],
    "hotel": ["Accommodation"],

    # Cost
    "cheap": ["Value for money", "Ticket price"],
    "affordable": ["Value for money", "Ticket price"],
    "value for money": ["Value for money"],
    "expensive": ["Value for money"],
    "ticket price": ["Ticket price"],
    "entrance fee": ["Entrance fee"],

    # Access
    "accessible": ["Public transport access", "Road access"],
    "public transport": ["Public transport access"],
    "road access": ["Road access"],

    # Tea
    "tea plantation": ["Tea plantation"],
    "tea": ["Tea plantation"],

    # Other
    "trail": ["Trail"],
    "hiking trail": ["Trail"],
    "stairs": ["Stairs"],
    "weather": ["Weather condition"],
    "shopping": ["Shops"],
    "local guide": ["Local guide"],
    "guided tour": ["Guided tours"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Facility Vocabulary
# Maps user synonyms → exact Facility node names in Neo4j
# Source: popular_facilities_by_place.csv
# ──────────────────────────────────────────────────────────────────────────────

FACILITY_SYNONYMS: dict[str, list[str]] = {
    # Entrance fee / Cost
    "entrance fee": ["Entrance fee"],
    "ticket price": ["Entrance fee"],
    "admission fee": ["Entrance fee"],
    "entry fee": ["Entrance fee"],
    "cost": ["Entrance fee"],
    "ticket": ["Ticket or receipt", "Entrance fee"],
    "receipt": ["Ticket or receipt"],
    "ticket counter": ["Ticket counter"],
    
    # Stay & Eat
    "accommodation": ["Accommodation"],
    "hotel": ["Accommodation"],
    "room": ["Accommodation"],
    "stay": ["Accommodation"],
    "restaurant": ["Restaurant"],
    "food": ["Restaurant", "Food stall"],
    "cafe": ["Cafe"],
    "coffee": ["Cafe"],
    "food stall": ["Food stall"],
    "snack": ["Food stall"],
    "souvenir shop": ["Souvenir shop"],
    "shop": ["Shop", "Souvenir shop"],
    
    # Travel & Access
    "tuk tuk service": ["Tuk tuk service"],
    "three wheeler": ["Tuk tuk service"],
    "tuktuk": ["Tuk tuk service"],
    "taxi service": ["Taxi service"],
    "cab": ["Taxi service"],
    "jeep service": ["Jeep service"],
    "jeep": ["Jeep service"],
    "boat service": ["Boat service"],
    "boat": ["Boat service"],
    "train service": ["Train service"],
    "train": ["Train service"],
    "bike rental": ["Bike rental"],
    "road access": ["Road access"],
    "public transport access": ["Public transport access"],
    "bus access": ["Public transport access"],
    
    # Guides
    "guide service": ["Guide service"],
    "tour guide": ["Guide service"],
    
    # Comfort & Infrastructure
    "steps": ["Steps or stairs"],
    "stairs": ["Steps or stairs"],
    "bridge": ["Bridge"],
    "parking": ["Parking"],
    "car park": ["Parking"],
    "toilet": ["Toilet"],
    "restroom": ["Toilet"],
    "washroom": ["Toilet"],
    "shade": ["Shade or shelter"],
    "shelter": ["Shade or shelter"],
    "signage": ["Signage"],
    "sign": ["Signage"],
    "information board": ["Information board"],
    "brochure": ["Brochure"],
    "seating": ["Seating"],
    "bench": ["Seating"],
    "shoe storage": ["Shoe storage"],
    "safety barrier": ["Safety barrier"],
    "accessibility ramp": ["Accessibility ramp"],
    "wheelchair access": ["Accessibility ramp"],
    "handrail": ["Handrail"],
    "lighting": ["Lighting"],
    "changing room": ["Changing room"],
    "drinking water": ["Drinking water"],
    "water": ["Drinking water"],
    "rest area": ["Rest area"],
    
    # Safety & Adventure
    "security": ["Security"],
    "donation box": ["Donation box"],
    "general facilities": ["General facilities"],
    "view platform": ["View platform"],
    "viewing deck": ["View platform"],
    "shower": ["Shower"],
    "clothing cover service": ["Clothing cover service"],
    "clothing": ["Clothing cover service"],
    "multilingual information": ["Multilingual information"],
    "lifeguard service": ["Lifeguard service"],
    "visitor centre": ["Visitor centre"],
    "visitor center": ["Visitor centre"],
    "towel service": ["Towel service"],
    "life jacket": ["Life jacket"],
    "waste bins": ["Waste bins"],
    "dustbin": ["Waste bins"],
    "camping facility": ["Camping facility"],
    "snorkelling equipment rental": ["Snorkelling equipment rental"],
}


def resolve_facility_terms(user_terms: list[str]) -> list[str]:
    """
    Maps a list of user-provided facility terms to exact graph Facility node names.
    Preserves the original term as fallback if no mapping found.
    Returns a deduplicated list.
    """
    resolved = []
    seen = set()
    for term in user_terms:
        lower = term.lower().strip()
        mapped = FACILITY_SYNONYMS.get(lower)
        if mapped:
            for m in mapped:
                if m not in seen:
                    resolved.append(m)
                    seen.add(m)
        else:
            fallback = term.strip()
            if fallback not in seen:
                resolved.append(fallback)
                seen.add(fallback)
    return resolved


def get_search_terms_for_facility(user_term: str) -> list[str]:
    """
    Returns all graph Facility names that a user term could map to.
    """
    lower = user_term.lower().strip()
    return FACILITY_SYNONYMS.get(lower, [user_term])


# ──────────────────────────────────────────────────────────────────────────────
# Lookup Functions
# ──────────────────────────────────────────────────────────────────────────────

def resolve_activity_terms(user_terms: list[str]) -> list[str]:
    """
    Maps a list of user-provided activity terms to exact graph Activity node names.
    Preserves the original term as fallback if no mapping found.
    Returns a deduplicated list.

    Example:
        ["surf", "whale watching"] → ["Surfing", "Whale watching"]
    """
    resolved = []
    seen = set()
    for term in user_terms:
        lower = term.lower().strip()
        mapped = ACTIVITY_SYNONYMS.get(lower)
        if mapped:
            for m in mapped:
                if m not in seen:
                    resolved.append(m)
                    seen.add(m)
        else:
            # fallback: title-case the term and use as-is
            fallback = term.strip()
            if fallback not in seen:
                resolved.append(fallback)
                seen.add(fallback)
    return resolved


def resolve_feature_terms(user_terms: list[str]) -> list[str]:
    """
    Maps a list of user-provided feature/quality terms to exact graph Feature node names.
    Preserves the original term as fallback if no mapping found.
    Returns a deduplicated list.

    Example:
        ["quiet", "family friendly"] → ["Peacefulness", "Family friendliness"]
    """
    resolved = []
    seen = set()
    for term in user_terms:
        lower = term.lower().strip()
        mapped = FEATURE_SYNONYMS.get(lower)
        if mapped:
            for m in mapped:
                if m not in seen:
                    resolved.append(m)
                    seen.add(m)
        else:
            fallback = term.strip()
            if fallback not in seen:
                resolved.append(fallback)
                seen.add(fallback)
    return resolved


def resolve_constraint_features(positive: list[str], negative: list[str]) -> tuple[list[str], list[str]]:
    """
    Resolves both positive and negative constraint terms to graph feature names.
    Returns (resolved_positive, resolved_negative).
    """
    return resolve_feature_terms(positive), resolve_feature_terms(negative)


def get_search_terms_for_activity(user_term: str) -> list[str]:
    """
    Returns all graph Activity names that a user term could map to.
    Used for building multi-term Cypher WHERE clauses.

    Example:
        "hike" → ["Hiking", "Trekking"]
    """
    lower = user_term.lower().strip()
    return ACTIVITY_SYNONYMS.get(lower, [user_term])


def get_search_terms_for_feature(user_term: str) -> list[str]:
    """
    Returns all graph Feature names that a user term could map to.
    Used for building multi-term Cypher WHERE clauses.

    Example:
        "quiet" → ["Peacefulness"]
        "family friendly" → ["Family friendliness"]
    """
    lower = user_term.lower().strip()
    return FEATURE_SYNONYMS.get(lower, [user_term])


# ──────────────────────────────────────────────────────────────────────────────
# Crowd Level Vocabulary
# Maps user synonyms → exact CrowdProfile.label values in Neo4j
# Labels: EMPTY | QUIET | MODERATE | BUSY | PACKED
# ──────────────────────────────────────────────────────────────────────────────

CROWD_LABEL_MAP: dict[str, str] = {
    # Low crowd (user wants quiet)
    "empty": "EMPTY",
    "no one": "EMPTY",
    "deserted": "EMPTY",
    "quiet": "QUIET",
    "not crowded": "QUIET",
    "uncrowded": "QUIET",
    "less crowded": "QUIET",
    "not busy": "QUIET",
    "avoid crowds": "QUIET",
    "peaceful": "QUIET",
    "calm": "QUIET",
    "tranquil": "QUIET",
    "serene": "QUIET",
    "moderate": "MODERATE",
    "somewhat busy": "MODERATE",
    # High crowd (user wants lively or wants to avoid)
    "busy": "BUSY",
    "crowded": "BUSY",
    "popular": "BUSY",
    "packed": "PACKED",
    "very crowded": "PACKED",
    "extremely crowded": "PACKED",
    "heaving": "PACKED",
}

CROWD_LEVEL_RANGE: dict[str, tuple] = {
    # Maps label → (min_level, max_level) for numeric filters
    "EMPTY":    (1, 1),
    "QUIET":    (1, 2),
    "MODERATE": (3, 3),
    "BUSY":     (4, 4),
    "PACKED":   (5, 5),
}


def resolve_crowd_label(user_term: str) -> str:
    """
    Maps a user crowd term to the closest CrowdProfile.label value.
    Returns 'QUIET' as default if no match found (safe fallback for 'not crowded' queries).
    """
    lower = user_term.lower().strip()
    return CROWD_LABEL_MAP.get(lower, "QUIET")


def get_crowd_level_range(label: str) -> tuple:
    """Returns (min_level, max_level) for a CrowdProfile label."""
    return CROWD_LEVEL_RANGE.get(label.upper(), (1, 3))


# ──────────────────────────────────────────────────────────────────────────────
# Time of Day Vocabulary
# Maps user synonyms → exact TimeProfile.time_of_day values in Neo4j
# Values: EARLY_MORNING | MID_MORNING | AFTERNOON | EVENING | NIGHT
# ──────────────────────────────────────────────────────────────────────────────

TIME_OF_DAY_MAP: dict[str, str] = {
    "early morning": "EARLY_MORNING",
    "very early": "EARLY_MORNING",
    "dawn": "EARLY_MORNING",
    "sunrise": "EARLY_MORNING",
    "morning": "MID_MORNING",
    "mid morning": "MID_MORNING",
    "late morning": "MID_MORNING",
    "noon": "AFTERNOON",
    "afternoon": "AFTERNOON",
    "midday": "AFTERNOON",
    "lunch time": "AFTERNOON",
    "evening": "EVENING",
    "sunset": "EVENING",
    "dusk": "EVENING",
    "night": "NIGHT",
    "nighttime": "NIGHT",
    "after dark": "NIGHT",
}

SEASON_MAP: dict[str, str] = {
    "dry season": "DRY",
    "dry": "DRY",
    "summer": "DRY",
    "wet season": "WET",
    "wet": "WET",
    "monsoon": "WET",
    "rainy": "WET",
    "winter": "WET",
}


def resolve_time_of_day(user_term: str) -> str | None:
    """
    Maps a user time term to the closest TimeProfile.time_of_day value.
    Returns None if no match found.
    """
    lower = user_term.lower().strip()
    return TIME_OF_DAY_MAP.get(lower)


def resolve_season(user_term: str) -> str | None:
    """Maps a user season term to the TimeProfile.season value."""
    lower = user_term.lower().strip()
    return SEASON_MAP.get(lower)


# ──────────────────────────────────────────────────────────────────────────────
# Cost Level Vocabulary
# Maps user synonyms → exact CostProfile.level values in Neo4j
# Values: FREE | LOW | MEDIUM | HIGH
# ──────────────────────────────────────────────────────────────────────────────

COST_LEVEL_MAP: dict[str, str] = {
    "free": "FREE",
    "free entry": "FREE",
    "no fee": "FREE",
    "no cost": "FREE",
    "no charge": "FREE",
    "cheap": "LOW",
    "budget": "LOW",
    "affordable": "LOW",
    "inexpensive": "LOW",
    "low cost": "LOW",
    "moderate": "MEDIUM",
    "reasonable": "MEDIUM",
    "mid range": "MEDIUM",
    "expensive": "HIGH",
    "pricey": "HIGH",
    "high cost": "HIGH",
    "luxury": "HIGH",
    "costly": "HIGH",
}


def resolve_cost_level(user_term: str) -> str | None:
    """
    Maps a user cost term to the closest CostProfile.level value.
    Returns None if no match found.
    """
    lower = user_term.lower().strip()
    return COST_LEVEL_MAP.get(lower)

