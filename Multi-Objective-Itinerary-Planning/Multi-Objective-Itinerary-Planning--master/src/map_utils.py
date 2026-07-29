import os, requests, folium
SRI_LANKA_CENTER = [7.8731, 80.7718]

def get_osrm_route(coords):
    if len(coords) < 2: return []
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}"
    try:
        r = requests.get(url, params={"overview":"full","geometries":"geojson"}, timeout=20)
        r.raise_for_status(); data = r.json()
        return [(lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
    except Exception:
        return []

def create_route_map(G, route, output_path="outputs/best_route_map.html"):
    m = folium.Map(location=SRI_LANKA_CENTER, zoom_start=8, tiles="OpenStreetMap", control_scale=True)
    coords = []
    for idx, place in enumerate(route, 1):
        n = G.nodes[place]; lat, lon = float(n["latitude"]), float(n["longitude"])
        coords.append((lat, lon)); color = "red" if idx == 1 else ("green" if idx == len(route) else "blue")
        folium.Marker([lat, lon], popup=f"{idx}. {place}", tooltip=f"{idx}. {place}", icon=folium.Icon(color=color)).add_to(m)
    road = get_osrm_route(coords)
    if road: folium.PolyLine(road, color="black", weight=5, opacity=0.85, tooltip="Real road route").add_to(m)
    elif len(coords) >= 2: folium.PolyLine(coords, color="black", weight=4, opacity=0.75, tooltip="Fallback route").add_to(m)
    if coords: m.fit_bounds(coords, padding=(30, 30))
    os.makedirs(os.path.dirname(output_path), exist_ok=True); m.save(output_path)
    return m
