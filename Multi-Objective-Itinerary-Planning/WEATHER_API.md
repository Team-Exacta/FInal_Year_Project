# OpenWeatherMap API setup

## Links
1. Create free account: https://home.openweathermap.org/users/sign_up
2. API keys page: https://home.openweathermap.org/api_keys
3. Docs (5-day forecast): https://openweathermap.org/forecast5
4. Current weather: https://openweathermap.org/current

## Where to put the key
Open file: `src/app.py`

Find:
```python
OPENWEATHERMAP_API_KEY = ""
```

Replace with your key:
```python
OPENWEATHERMAP_API_KEY = "YOUR_KEY_HERE"
```

## How weather is used
- Enter **Trip start date** in the form.
- For each candidate place and each trip day, the app calls the **5-day / 3-hour forecast**.
- A weather table (places × days) is shown.
- Places with bad weather on most/all days are avoided.
- After day-split, if a place falls on a day with bad forecast, a warning is shown.
- Free plan forecast covers about **5 days from today**.

## Note
New keys can take up to ~10 minutes to activate after signup.
