import sys
import os

# Adjust sys.path to run from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.distance_matrix import DistanceMatrix

def test_distance_matrix():
    print("=== Testing Distance Matrix ===")
    dm = DistanceMatrix()
    
    # 1. Simple lookup
    d1 = dm.get_distance("Mirissa Beach", "Weligama Beach")
    print(f"Distance (Mirissa Beach -> Weligama Beach): {d1} km (Expected: ~8.1 km)")
    assert d1 is not None and abs(d1 - 8.1) < 0.2, "Mirissa to Weligama distance mismatch"

    # 2. Case-insensitivity and substring matching normalization test
    d2 = dm.get_distance("mirissa", "weligama")
    print(f"Distance (mirissa -> weligama case-insensitive): {d2} km (Expected: ~8.1 km)")
    assert d2 is not None and abs(d2 - 8.1) < 0.2, "Case-insensitive normalization mismatch"

    # 3. Dynamic Geocoding and Routing fallback test (Matara city to Mirissa Beach POI)
    print("\n--- Testing Dynamic Geocoding Fallbacks ---")
    d3 = dm.get_distance("Matara", "Mirissa Beach")
    print(f"Dynamic Distance (Matara City -> Mirissa Beach POI): {d3} km")
    assert d3 is not None and d3 > 0.0, "Dynamic geocoded distance lookup failed"

    d4 = dm.get_distance("Colombo", "Galle")
    print(f"Dynamic Distance (Colombo -> Galle): {d4} km")
    assert d4 is not None and d4 > 100.0, "Long range dynamic routing lookup failed"

    d5 = dm.get_distance("Kegalle", "Pinnawala Elephant Orphanage")
    print(f"Dynamic Distance (Kegalle City -> Pinnawala POI): {d5} km")
    assert d5 is not None and d5 > 0.0, "Kegalle to Pinnawala dynamic lookup failed"

    # 4. Pairwise Markdown Table Generation (Including general cities)
    places = ["Matara", "Mirissa Beach", "Weligama Beach", "Colombo"]
    table = dm.generate_pairwise_table(places)
    print("\nGenerated Dynamic Pairwise Distance Table:")
    print(table)
    
    # 5. Nearby Places Lookup
    nearby = dm.get_nearby_places("Mirissa Beach", top_n=5)
    print("\nTop 5 nearby places from Mirissa Beach:")
    for name, dist in nearby:
        print(f"- {name}: {dist:.1f} km")
        
    print("\n=== Distance Matrix Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_distance_matrix()
