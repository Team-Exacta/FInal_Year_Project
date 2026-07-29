import requests
BAD_WEATHER_KEYWORDS = ["rain", "thunderstorm", "storm", "flood", "drizzle", "heavy rain", "very heavy rain", "extreme rain", "squall"]

def get_weather_status(lat, lon, api_key):
    if not api_key:
        return {"available": False, "is_bad": False, "main": "Skipped", "description": "No API key provided"}
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        weather = data.get("weather", [{}])[0]
        main = str(weather.get("main", "Unknown"))
        desc = str(weather.get("description", "Unknown")).lower()
        is_bad = main.lower() in ["rain", "thunderstorm", "drizzle"] or any(x in desc for x in BAD_WEATHER_KEYWORDS)
        return {"available": True, "is_bad": is_bad, "main": main, "description": desc, "temperature": data.get("main", {}).get("temp")}
    except Exception as e:
        return {"available": False, "is_bad": False, "main": "Error", "description": str(e)}

def filter_places_by_weather(G, places, api_key, avoid_bad_weather=True, max_checks=60):
    accepted, removed, report = [], [], {}
    # To avoid API limits, check first max_checks candidate places only. Others pass if not checked.
    for idx, p in enumerate(places):
        if idx >= max_checks or not api_key:
            report[p] = {"available": False, "is_bad": False, "main": "Skipped", "description": "Weather check skipped to avoid API limit"}
            accepted.append(p)
            continue
        n = G.nodes[p]
        status = get_weather_status(n["latitude"], n["longitude"], api_key)
        report[p] = status
        if avoid_bad_weather and status["available"] and status["is_bad"]:
            removed.append(p)
        else:
            accepted.append(p)
    return accepted, removed, report
