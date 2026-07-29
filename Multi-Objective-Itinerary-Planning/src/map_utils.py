"""
Map helpers: real road geometry via OSRM, day-coloured polylines.
Days are CONNECTED: travel from end of day N to start of day N+1 is drawn
as part of day N+1's route colour so the full trip is continuous.
"""
import os
import requests
import folium
from folium import FeatureGroup

SRI_LANKA_CENTER = [7.8731, 80.7718]

DAY_COLORS = [
    "#e41a1c",  # Day 1 red
    "#377eb8",  # Day 2 blue
    "#4daf4a",  # Day 3 green
    "#984ea3",  # Day 4 purple
    "#ff7f00",  # Day 5 orange
    "#a65628",
    "#f781bf",
    "#999999",
]

# Marker colours (folium named colours) — places use consistent icons
PLACE_ICON = "blue"
START_ICON = "red"
END_ICON = "green"


def get_osrm_route(coords, avoid_highways=True):
    """coords: list of (lat, lon). Prefer non-motorway when supported."""
    if len(coords) < 2:
        return []
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    if avoid_highways:
        params["exclude"] = "motorway"
    try:
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200 and avoid_highways:
            params.pop("exclude", None)
            r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        return [(lat, lon) for lon, lat in data["routes"][0]["geometry"]["coordinates"]]
    except Exception:
        return []


def _node_latlon(G, place):
    n = G.nodes[place]
    return float(n["latitude"]), float(n["longitude"])


def create_route_map(G, route, output_path="outputs/best_route_map.html"):
    m = folium.Map(location=SRI_LANKA_CENTER, zoom_start=8, tiles="OpenStreetMap", control_scale=True)
    coords = []
    for idx, place in enumerate(route, 1):
        if place not in G.nodes:
            continue
        lat, lon = _node_latlon(G, place)
        coords.append((lat, lon))
        color = "red" if idx == 1 else ("green" if idx == len(route) else "blue")
        folium.Marker(
            [lat, lon],
            popup=f"{idx}. {place}",
            tooltip=f"{idx}. {place}",
            icon=folium.Icon(color=color),
        ).add_to(m)
    road = get_osrm_route(coords, avoid_highways=True)
    if road:
        folium.PolyLine(road, color="black", weight=5, opacity=0.85).add_to(m)
    elif len(coords) >= 2:
        folium.PolyLine(coords, color="black", weight=4, opacity=0.75).add_to(m)
    if coords:
        m.fit_bounds(coords, padding=(30, 30))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    return m


def create_multiday_map(G, day_plans, output_path="outputs/best_route_map.html"):
    """
    day_plans: list of {day, places}

    - Place markers: blue (middle), red (trip start), green (trip end)
    - Road for each day: distinct DAY_COLORS[day-1]
    - Days are connected: path includes previous day's last place → this day's places
      so travel Museum → Hikkaduwa appears on Day 2's blue line (connection).
    """
    m = folium.Map(location=SRI_LANKA_CENTER, zoom_start=8, tiles="OpenStreetMap", control_scale=True)
    all_coords = []

    # Flatten full ordered route for start/end markers
    full_route = []
    for plan in day_plans:
        for p in plan.get("places") or []:
            if p not in full_route or full_route[-1] != p:
                full_route.append(p)

    prev_last = None  # last place of previous day

    for plan in day_plans:
        day = int(plan.get("day", 1))
        places = [p for p in (plan.get("places") or []) if p in G.nodes]
        color = DAY_COLORS[(day - 1) % len(DAY_COLORS)]
        group = FeatureGroup(name=f"Day {day}", show=True)

        # Path coordinates: connect from previous day end → today's places
        path_places = []
        if prev_last and places and prev_last != places[0]:
            path_places.append(prev_last)  # connection from previous day
        path_places.extend(places)

        coords = []
        for p in path_places:
            lat, lon = _node_latlon(G, p)
            coords.append((lat, lon))
            all_coords.append((lat, lon))

        # Markers only for places VISITED this day (not the previous-day connector)
        for idx, place in enumerate(places):
            lat, lon = _node_latlon(G, place)
            # Trip start / trip end special icons
            if full_route and place == full_route[0] and day == day_plans[0].get("day"):
                icon_color = START_ICON
            elif full_route and place == full_route[-1] and day == day_plans[-1].get("day"):
                icon_color = END_ICON
            else:
                icon_color = PLACE_ICON
            folium.CircleMarker(
                [lat, lon],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=f"Day {day} · {idx+1}. {place}",
                tooltip=f"D{day}: {place}",
            ).add_to(group)
            folium.Marker(
                [lat, lon],
                popup=f"Day {day} · {idx+1}. {place}",
                tooltip=f"D{day}: {place}",
                icon=folium.Icon(color=icon_color, icon="info-sign"),
            ).add_to(group)

        # Road polyline for this day (includes inter-day link if any)
        if len(coords) >= 2:
            road = get_osrm_route(coords, avoid_highways=True)
            if road:
                folium.PolyLine(
                    road,
                    color=color,
                    weight=6,
                    opacity=0.9,
                    tooltip=f"Day {day} road (connected)",
                ).add_to(group)
            else:
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=5,
                    opacity=0.85,
                    tooltip=f"Day {day}",
                ).add_to(group)
        elif len(coords) == 1 and prev_last:
            # Single place day but connected from previous — still try link
            link = [_node_latlon(G, prev_last), coords[0]]
            road = get_osrm_route(link, avoid_highways=True)
            if road:
                folium.PolyLine(road, color=color, weight=6, opacity=0.9,
                                tooltip=f"Day {day} arrival").add_to(group)
            else:
                folium.PolyLine(link, color=color, weight=5, opacity=0.85,
                                tooltip=f"Day {day} arrival").add_to(group)

        group.add_to(m)
        if places:
            prev_last = places[-1]

    folium.LayerControl(collapsed=False).add_to(m)
    if all_coords:
        m.fit_bounds(all_coords, padding=(40, 40))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    return m
