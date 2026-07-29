import pandas as pd

def initialize_pheromones_from_past_trips(G, past_trips_file, boost=5.0):
    try:
        past = pd.read_csv(past_trips_file)
    except Exception:
        return G, 0, 0
    trip_col = "trip_id" if "trip_id" in past.columns else past.columns[0]
    order_col = "order" if "order" in past.columns else past.columns[1]
    poi_col = "poi" if "poi" in past.columns else past.columns[2]
    matched, missing = 0, 0
    for _, group in past.groupby(trip_col):
        group = group.sort_values(order_col)
        route = [str(x).strip() for x in group[poi_col].tolist()]
        for i in range(len(route)-1):
            u, v = route[i], route[i+1]
            if G.has_edge(u, v):
                G[u][v]["pheromone"] += boost
                matched += 1
            elif G.has_edge(v, u):
                G[v][u]["pheromone"] += boost
                matched += 1
            else:
                missing += 1
    return G, matched, missing
