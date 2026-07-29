import os
import math
import pandas as pd
from src.core.logger import get_logger

logger = get_logger(__name__)

class DistanceMatrix:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DistanceMatrix, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.csv_path = os.path.join(base_dir, "data", "DistanceMatrix", "route_distance_matrix.csv")
        self.poi_coords_path = os.path.join(base_dir, "data", "graph", "poi_data.csv")
        self.df = None
        self.place_map = {}      # maps normalized name -> exact CSV column name
        self.poi_coords = {}     # maps poi_name -> (lat, lon) from poi_data.csv
        self.load_matrix()
        self._load_poi_coords()
        self._initialized = True

    def load_matrix(self):
        try:
            if os.path.exists(self.csv_path):
                # Load matrix with first column as index
                self.df = pd.read_csv(self.csv_path, index_col=0)
                # Normalize column headers for case-insensitive lookup mapping
                for col in self.df.columns:
                    norm = col.strip().lower()
                    self.place_map[norm] = col
                logger.info(f"[DistanceMatrix] Loaded road distance matrix with {len(self.df)} tourist destinations.")
            else:
                logger.warning(f"[DistanceMatrix] Matrix CSV file not found at {self.csv_path}. Distances will be disabled.")
        except Exception as e:
            logger.error(f"[DistanceMatrix] Failed to load CSV matrix: {e}")

    def normalize_name(self, name: str) -> str:
        if not name:
            return ""
        return name.strip().lower()

    def get_exact_name(self, name: str) -> str | None:
        """Returns the exact name used in the CSV columns or None if not matched."""
        norm = self.normalize_name(name)
        # Check direct mapping
        if norm in self.place_map:
            return self.place_map[norm]
        
        # Check safe prefix matching as a fallback (e.g. "Hiriketiya" matches "Hiriketiya Beach")
        # This prevents general cities (like "Kegalle") from matching POIs containing the city name (like "Pinnawala Elephant Orphanage, Kegalle")
        for k, v in self.place_map.items():
            if k.startswith(norm):
                return v
        return None

    def get_distance(self, place_a: str, place_b: str) -> float | None:
        """Gets road distance in km between two Sri Lankan places, falling back to Nominatim/OSRM dynamic routing if not in CSV."""
        if self.df is not None:
            exact_a = self.get_exact_name(place_a)
            exact_b = self.get_exact_name(place_b)
            
            if exact_a and exact_b:
                try:
                    val = self.df.loc[exact_a, exact_b]
                    if not pd.isna(val):
                        return float(val)
                except Exception:
                    pass
                    
        # Fall back to dynamic online Nominatim/OSRM API integration
        logger.info(f"[DistanceMatrix] '{place_a}' or '{place_b}' not in static matrix. Falling back to dynamic routing...")
        return self.get_dynamic_distance(place_a, place_b)

    def geocode_place(self, name: str) -> tuple[float, float] | None:
        """Converts a city or place name into (lat, lon) coordinates using OpenStreetMap's anonymous Nominatim geocoding."""
        import requests
        import urllib.parse
        
        if not name:
            return None
            
        try:
            query = f"{name}, Sri Lanka"
            encoded_query = urllib.parse.quote(query)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
            headers = {"User-Agent": "SriLankaTourismRAGAgent/1.0"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    return lat, lon
        except Exception as e:
            logger.warning(f"[DistanceMatrix] Nominatim geocoding failed for '{name}': {e}")
            
        return None

    def get_dynamic_distance(self, place_a: str, place_b: str) -> float | None:
        """Calculates precise road driving distance in km using the public Project OSRM API."""
        import requests
        
        # 1. Geocode Place A
        coords_a = self.geocode_place(place_a)
        if not coords_a:
            return None
            
        # 2. Geocode Place B
        coords_b = self.geocode_place(place_b)
        if not coords_b:
            return None
            
        lat_a, lon_a = coords_a
        lat_b, lon_b = coords_b
        
        # 3. Call public Project OSRM Route API
        try:
            url = f"http://router.project-osrm.org/route/v1/driving/{lon_a},{lat_a};{lon_b},{lat_b}?overview=false"
            headers = {"User-Agent": "SriLankaTourismRAGAgent/1.0"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    dist_meters = data["routes"][0]["distance"]
                    return dist_meters / 1000.0
        except Exception as e:
            logger.warning(f"[DistanceMatrix] Public OSRM routing failed: {e}")
            
        return None

    def get_nearby_places(self, place_name: str, top_n: int = 5) -> list[tuple[str, float]]:
        """Finds the top N closest road destinations to a given place name using the static CSV matrix."""
        if self.df is None:
            return []

        exact_name = self.get_exact_name(place_name)
        if not exact_name:
            return []

        try:
            row = self.df.loc[exact_name]
            # Exclude self (distance == 0.0) and drop missing
            row = row[row > 0.0].dropna()
            sorted_row = row.sort_values()
            return [(name, float(dist)) for name, dist in sorted_row.head(top_n).items()]
        except Exception as e:
            logger.error(f"[DistanceMatrix] Error finding nearby places: {e}")
            return []

    def _load_poi_coords(self):
        """
        Loads (lat, lon) for all 165 known POIs from poi_data.csv into self.poi_coords.
        Called once at singleton init — zero cost at query time.
        """
        try:
            if os.path.exists(self.poi_coords_path):
                poi_df = pd.read_csv(self.poi_coords_path)
                for _, row in poi_df.iterrows():
                    name = str(row.get("poi_name", "")).strip()
                    lat = row.get("latitude")
                    lon = row.get("longitude")
                    if name and pd.notna(lat) and pd.notna(lon):
                        self.poi_coords[name] = (float(lat), float(lon))
                logger.info(f"[DistanceMatrix] Loaded coordinates for {len(self.poi_coords)} POIs from poi_data.csv.")
            else:
                logger.warning(f"[DistanceMatrix] poi_data.csv not found at {self.poi_coords_path}.")
        except Exception as e:
            logger.error(f"[DistanceMatrix] Failed to load POI coordinates: {e}")

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Returns the great-circle (straight-line) distance in km between two coordinates.
        Used as a fast pre-filter before making OSRM road routing calls.
        """
        R = 6371.0  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def get_nearby_for_unknown_place(self, place_name: str, top_n: int = 8, haversine_candidates: int = 15) -> list[tuple[str, float]]:
        """
        Finds the top N nearest known POIs (by road distance) for a place NOT in the static CSV.

        Strategy:
          1. Geocode the anchor place via Nominatim (1 API call, free).
          2. Use Haversine to find the closest `haversine_candidates` POIs from poi_data.csv
             (pure math, no API calls).
          3. Call OSRM for accurate road distances on those candidates only
             (up to `haversine_candidates` API calls, free).
          4. Sort by road distance and return top N.

        Falls back gracefully at every step — returns [] if geocoding fails.
        """
        # Step 1: Geocode the anchor place
        anchor_coords = self.geocode_place(place_name)
        if not anchor_coords:
            logger.warning(f"[DistanceMatrix] Could not geocode '{place_name}'. Cannot find nearby places.")
            return []

        anchor_lat, anchor_lon = anchor_coords
        logger.info(f"[DistanceMatrix] Geocoded '{place_name}' -> lat={anchor_lat:.4f}, lon={anchor_lon:.4f}")

        if not self.poi_coords:
            logger.warning("[DistanceMatrix] No POI coordinates loaded. Cannot do Haversine pre-filter.")
            return []

        # Step 2: Haversine pre-filter — find closest N candidates by straight-line distance
        haversine_distances = []
        for poi_name, (poi_lat, poi_lon) in self.poi_coords.items():
            straight_km = self._haversine(anchor_lat, anchor_lon, poi_lat, poi_lon)
            haversine_distances.append((poi_name, straight_km))

        haversine_distances.sort(key=lambda x: x[1])
        candidates = haversine_distances[:haversine_candidates]
        logger.info(
            f"[DistanceMatrix] Haversine top-{haversine_candidates} candidates for '{place_name}': "
            f"{[c[0] for c in candidates]}"
        )

        # Step 3: OSRM road distance for each candidate
        road_results = []
        for poi_name, _ in candidates:
            poi_coords = self.poi_coords.get(poi_name)
            if not poi_coords:
                continue
            poi_lat, poi_lon = poi_coords
            import requests
            try:
                url = (
                    f"http://router.project-osrm.org/route/v1/driving/"
                    f"{anchor_lon},{anchor_lat};{poi_lon},{poi_lat}?overview=false"
                )
                headers = {"User-Agent": "SriLankaTourismRAGAgent/1.0"}
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("code") == "Ok" and data.get("routes"):
                        road_km = data["routes"][0]["distance"] / 1000.0
                        road_results.append((poi_name, road_km))
                        logger.info(f"[DistanceMatrix] OSRM: '{place_name}' -> '{poi_name}' = {road_km:.1f} km")
            except Exception as e:
                logger.warning(f"[DistanceMatrix] OSRM call failed for '{poi_name}': {e}")

        # Step 4: Sort by road distance, return top N
        road_results.sort(key=lambda x: x[1])
        return road_results[:top_n]

    def generate_pairwise_table(self, places: list[str]) -> str:
        """Generates a structured Markdown table of pairwise distances for a list of places."""
        if self.df is None or len(places) < 2:
            return ""
            
        # Get exact names and remove duplicates
        exact_places = []
        seen = set()
        for p in places:
            exact = self.get_exact_name(p)
            if exact and exact not in seen:
                exact_places.append(exact)
                seen.add(exact)
                
        if len(exact_places) < 2:
            return ""
            
        rows = []
        for i in range(len(exact_places)):
            for j in range(i + 1, len(exact_places)):
                p1 = exact_places[i]
                p2 = exact_places[j]
                dist = self.get_distance(p1, p2)
                if dist is not None:
                    rows.append((p1, p2, dist))
                    
        if not rows:
            return ""
            
        # Sort by distance (closest first)
        rows.sort(key=lambda x: x[2])
        
        table = ["| Place A | Place B | Proximity (Road Distance) |", "| :--- | :--- | :--- |"]
        for p1, p2, dist in rows:
            table.append(f"| {p1} | {p2} | {dist:.1f} km |")
            
        return "\n".join(table)
