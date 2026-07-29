import difflib

def normalize_weights(preferences):
    total = sum(max(float(v), 0.0) for v in preferences.values())
    if total == 0:
        return {k: 1 / len(preferences) for k in preferences}
    return {k: max(float(v), 0.0) / total for k, v in preferences.items()}

def split_text_places(raw):
    if not raw or not raw.strip():
        return []
    return [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]

def fuzzy_match_place(name, valid_places):
    if name in valid_places:
        return name
        
    # Check for substring matches first (e.g. "Sigiriya" in "Sigiriya Lion Rock")
    name_lower = name.lower()
    for place in valid_places:
        if name_lower in place.lower():
            return place
            
    # Fallback to difflib
    matches = difflib.get_close_matches(name, valid_places, n=1, cutoff=0.5)
    if not matches:
        raise ValueError(f"Place not found: {name}. Please use exact POI name from dropdown/list.")
    return matches[0]

def min_max(value, min_value, max_value, higher_better=True):
    if max_value == min_value:
        return 1.0
    v = (value - min_value) / (max_value - min_value)
    v = max(0.0, min(1.0, v))
    return v if higher_better else 1.0 - v
