import pandas as pd

def find_similar_places(target_place, min_shared_features=3):
    try:
        # Load the popular features dataset
        df = pd.read_csv('popular_features_by_place.csv')
    except FileNotFoundError:
        print("Error: popular_features_by_place.csv not found. Make sure you are in the correct directory.")
        return

    # 1. Check if the target place exists
    if target_place not in df['place_name'].values:
        print(f"Place '{target_place}' not found in the dataset.")
        return

    # 2. Get the set of features for the target place
    target_features = set(df[df['place_name'] == target_place]['feature'])
    print(f"\n{target_place} has {len(target_features)} features: {', '.join(target_features)}\n")

    places = df['place_name'].unique()
    results = []

    # 3. Compare the target features against all other places
    for p in places:
        if p == target_place:
            continue # Skip comparing the place to itself
        
        p_features = set(df[df['place_name'] == p]['feature'])
        shared_features = target_features.intersection(p_features)
        
        # 4. If they share enough features, add them to our results
        if len(shared_features) >= min_shared_features:
            results.append((p, len(shared_features), shared_features))

    # 5. Sort the results from highest match to lowest
    results.sort(key=lambda x: x[1], reverse=True)

    # 6. Print out the results cleanly
    print(f"--- Found {len(results)} places that share at least {min_shared_features} features with {target_place} ---\n")
    for r in results:
        place_name = r[0]
        match_count = r[1]
        shared_str = ', '.join(r[2])
        print(f"- {place_name} ({match_count} shared): {shared_str}")

if __name__ == "__main__":
    # ============================================================
    # Edit these variables below to run your own queries!
    # ============================================================
    query_place = "Arugam Bay"
    minimum_matches_required = 3
    
    find_similar_places(query_place, minimum_matches_required)
