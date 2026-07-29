import sys
import os

# Add src to path so we can import graph_vocabulary
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.v2.graph_vocabulary import (
    ACTIVITY_SYNONYMS,
    FEATURE_SYNONYMS,
    FACILITY_SYNONYMS,
    CROWD_LABEL_MAP,
    TIME_OF_DAY_MAP,
    SEASON_MAP,
    COST_LEVEL_MAP
)

dicts = {
    "ACTIVITY": ACTIVITY_SYNONYMS,
    "FEATURE": FEATURE_SYNONYMS,
    "FACILITY": FACILITY_SYNONYMS,
    "CROWD": CROWD_LABEL_MAP,
    "TIME": TIME_OF_DAY_MAP,
    "SEASON": SEASON_MAP,
    "COST": COST_LEVEL_MAP
}

# Check for identical keys in multiple dictionaries
key_locations = {}
for dict_name, d in dicts.items():
    for key in d.keys():
        if key not in key_locations:
            key_locations[key] = []
        key_locations[key].append(dict_name)

conflicts = {k: v for k, v in key_locations.items() if len(v) > 1}

print("Key Conflicts:")
if conflicts:
    for k, v in conflicts.items():
        print(f"'{k}' found in: {v}")
else:
    print("None")
