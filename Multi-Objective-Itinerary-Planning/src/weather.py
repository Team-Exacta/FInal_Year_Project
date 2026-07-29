"""
OpenWeatherMap integration.

Get a free API key:
  https://home.openweathermap.org/users/sign_up
  https://home.openweathermap.org/api_keys

Free plan includes:
  - Current weather
  - 5-day / 3-hour forecast  (api.openweathermap.org/data/2.5/forecast)

Put your key in app.py:
  OPENWEATHERMAP_API_KEY = "your_key_here"
"""
from datetime import datetime, timedelta, date
from collections import defaultdict
import requests

BAD_WEATHER_KEYWORDS = [
    "rain", "thunderstorm", "storm", "flood", "drizzle",
    "heavy rain", "very heavy rain", "extreme rain", "squall", "shower",
]
BAD_MAIN = {"rain", "thunderstorm", "drizzle", "squall"}


def _is_bad(main, description):
    main_l = (main or "").lower()
    desc_l = (description or "").lower()
    if main_l in BAD_MAIN:
        return True
    return any(k in desc_l for k in BAD_WEATHER_KEYWORDS)


def get_weather_status(lat, lon, api_key):
    """Current weather for one coordinate."""
    if not api_key:
        return {
            "available": False, "is_bad": False, "main": "Skipped",
            "description": "No API key", "temperature": None,
        }
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        r = requests.get(
            url,
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        weather = data.get("weather", [{}])[0]
        main = str(weather.get("main", "Unknown"))
        desc = str(weather.get("description", ""))
        return {
            "available": True,
            "is_bad": _is_bad(main, desc),
            "main": main,
            "description": desc,
            "temperature": data.get("main", {}).get("temp"),
        }
    except Exception as e:
        return {
            "available": False, "is_bad": False, "main": "Error",
            "description": str(e), "temperature": None,
        }


def get_forecast_by_day(lat, lon, api_key, start_date, num_days):
    """
    5-day / 3-hour forecast grouped by calendar date.
    Returns dict: { 'YYYY-MM-DD': {is_bad, main, description, temperature, samples} }
    OpenWeather free forecast covers ~5 days from now.
    """
    if not api_key:
        return {}
    try:
        url = "https://api.openweathermap.org/data/2.5/forecast"
        r = requests.get(
            url,
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {}

    by_day = defaultdict(list)
    for item in data.get("list", []):
        dt = datetime.utcfromtimestamp(item["dt"]).date()
        weather = item.get("weather", [{}])[0]
        main = str(weather.get("main", ""))
        desc = str(weather.get("description", ""))
        temp = item.get("main", {}).get("temp")
        by_day[dt.isoformat()].append({
            "main": main,
            "description": desc,
            "is_bad": _is_bad(main, desc),
            "temperature": temp,
        })

    # Target trip dates
    if isinstance(start_date, str):
        start = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start = start_date.date()
    elif isinstance(start_date, date):
        start = start_date
    else:
        start = date.today()

    result = {}
    for i in range(num_days):
        d = (start + timedelta(days=i)).isoformat()
        samples = by_day.get(d, [])
        if not samples:
            # No forecast for that date (beyond 5 days or past)
            result[d] = {
                "available": False,
                "is_bad": False,
                "main": "No forecast",
                "description": "Outside 5-day forecast window or no data",
                "temperature": None,
            }
            continue
        bad_count = sum(1 for s in samples if s["is_bad"])
        # Majority or any midday rain → treat day as bad
        is_bad = bad_count >= max(1, len(samples) // 3)
        # Prefer a daytime sample for display
        mid = samples[len(samples) // 2]
        result[d] = {
            "available": True,
            "is_bad": is_bad,
            "main": mid["main"],
            "description": mid["description"],
            "temperature": mid.get("temperature"),
            "bad_slots": bad_count,
            "total_slots": len(samples),
        }
    return result


def build_weather_table(G, places, api_key, start_date, num_days):
    """
    Build a place × day weather table.
    Returns:
      table_rows: list of dicts for display
      bad_on_day: { day_index (1-based): set of place names that are bad that day }
      report: nested dict place -> date -> status
    """
    table_rows = []
    bad_on_day = {d: set() for d in range(1, num_days + 1)}
    report = {}

    if isinstance(start_date, str):
        start = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
    elif isinstance(start_date, datetime):
        start = start_date.date()
    elif isinstance(start_date, date):
        start = start_date
    else:
        start = date.today()

    for place in places:
        if place not in G.nodes:
            continue
        n = G.nodes[place]
        lat, lon = float(n["latitude"]), float(n["longitude"])
        daily = get_forecast_by_day(lat, lon, api_key, start, num_days)
        report[place] = daily
        row = {"Place": place}
        for i in range(num_days):
            d = (start + timedelta(days=i)).isoformat()
            st = daily.get(d, {})
            label = d
            if st.get("available"):
                cell = f"{st.get('main', '?')} ({st.get('temperature', '?')}°C)"
                if st.get("is_bad"):
                    cell = "⚠ " + cell
                    bad_on_day[i + 1].add(place)
            else:
                cell = st.get("description", "N/A")[:40]
            row[f"Day {i+1} ({d})"] = cell
        table_rows.append(row)

    return table_rows, bad_on_day, report


def filter_places_by_weather(G, places, api_key, avoid_bad_weather=True, max_checks=40):
    """Legacy: filter by *current* weather only (used when no start date)."""
    accepted, removed, report = [], [], {}
    for idx, p in enumerate(places):
        if idx >= max_checks or not api_key:
            report[p] = {
                "available": False, "is_bad": False, "main": "Skipped",
                "description": "Weather check skipped",
            }
            accepted.append(p)
            continue
        n = G.nodes[p]
        status = get_weather_status(n["latitude"], n["longitude"], api_key)
        report[p] = status
        if avoid_bad_weather and status.get("available") and status.get("is_bad"):
            removed.append(p)
        else:
            accepted.append(p)
    return accepted, removed, report


def places_bad_on_any_trip_day(bad_on_day):
    """Union of places that have bad weather on at least one trip day."""
    bad = set()
    for s in bad_on_day.values():
        bad |= s
    return bad
