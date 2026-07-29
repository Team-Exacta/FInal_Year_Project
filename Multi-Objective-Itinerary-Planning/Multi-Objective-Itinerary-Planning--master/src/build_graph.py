import pandas as pd
import networkx as nx

def _detect_name_col(df):
    for c in ["poi_name", "name", "place_name", "POI", "poi"]:
        if c in df.columns:
            return c
    return df.columns[0]

def _safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def build_graph(poi_file, distance_file, duration_file):
    pois = pd.read_csv(poi_file)
    dist = pd.read_csv(distance_file)
    dur = pd.read_csv(duration_file)
    name_col = _detect_name_col(pois)
    G = nx.DiGraph()

    for _, row in pois.iterrows():
        name = str(row[name_col]).strip()
        G.add_node(
            name,
            poi_id=row.get("poi_id", ""),
            category=str(row.get("category", "")),
            satisfaction_score=_safe_float(row.get("satisfaction_score", 0)),
            cost=_safe_float(row.get("cost", 0)),
            duration_time_min=_safe_float(row.get("duration_time_min", 60), 60),
            latitude=_safe_float(row.get("latitude", 0)),
            longitude=_safe_float(row.get("longitude", 0)),
        )

    dist_src = dist.columns[0]
    dur_src = dur.columns[0]
    for _, row in dist.iterrows():
        source = str(row[dist_src]).strip()
        if source not in G:
            continue
        dur_match = dur[dur[dur_src].astype(str).str.strip() == source]
        if dur_match.empty:
            continue
        dur_row = dur_match.iloc[0]
        for dest_col in dist.columns[1:]:
            dest = str(dest_col).strip()
            if dest not in G or dest == source:
                continue
            try:
                distance_km = float(row[dest_col])
                travel_time_min = float(dur_row[dest_col])
            except Exception:
                continue
            if distance_km <= 0:
                continue
            G.add_edge(source, dest, distance_km=distance_km, travel_time_min=travel_time_min, pheromone=1.0)
    return G
