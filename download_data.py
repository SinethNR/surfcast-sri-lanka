"""
download_data.py - reproducible download of the SurfCast Sri Lanka data.

Source: Open-Meteo (https://open-meteo.com) - free, no API key, no login.
  * Marine API  -> daily wave / swell data
  * Archive API -> daily historical weather (ERA5 reanalysis)
for two Sri Lankan surf spots: Arugam Bay (east coast) and Hikkaduwa (south-west coast).

Files are cached in data/raw/. Existing files are not downloaded again
(use --force to re-download). Failed requests are retried with back-off.

Run:  python download_data.py

AI-use declaration: parts of this code were generated with Claude (Anthropic).
"""
import argparse
import os
import time

import requests

START, END = "2010-01-01", "2025-12-31"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

MARINE_VARS = ("wave_height_max,wave_direction_dominant,wave_period_max,wind_wave_height_max,"
               "wind_wave_direction_dominant,wind_wave_period_max,swell_wave_height_max,"
               "swell_wave_direction_dominant,swell_wave_period_max,swell_wave_peak_period_max")
WEATHER_VARS = ("weather_code,temperature_2m_max,precipitation_sum,precipitation_hours,wind_speed_10m_max,"
                "wind_gusts_10m_max,wind_direction_10m_dominant,relative_humidity_2m_mean,"
                "pressure_msl_mean,cloud_cover_mean")

# marine points are slightly offshore so they fall on a sea grid cell
JOBS = [
    ("arugam_marine.csv", "https://marine-api.open-meteo.com/v1/marine", 6.84, 81.87, MARINE_VARS),
    ("arugam_weather.csv", "https://archive-api.open-meteo.com/v1/archive", 6.84, 81.83, WEATHER_VARS),
    ("hikkaduwa_marine.csv", "https://marine-api.open-meteo.com/v1/marine", 6.14, 80.07, MARINE_VARS),
    ("hikkaduwa_weather.csv", "https://archive-api.open-meteo.com/v1/archive", 6.14, 80.10, WEATHER_VARS),
]


def fetch(url, params, retries=5):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200 and r.text.lower().startswith("latitude"):
                return r.text
            msg = r.text[:200]
            print(f"   attempt {attempt}: HTTP {r.status_code} {msg}")
        except requests.RequestException as e:
            print(f"   attempt {attempt}: {e}")
        wait = 65 if attempt == 1 else 30 * attempt   # Open-Meteo free tier has a per-minute limit
        print(f"   waiting {wait}s before retry ...")
        time.sleep(wait)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def main(force=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    first = True
    for name, url, lat, lon, daily in JOBS:
        path = os.path.join(OUT_DIR, name)
        if os.path.exists(path) and not force:
            print(f"[cached] {name}")
            continue
        if not first:
            print("   pausing 65s to respect the free API's per-minute limit ...")
            time.sleep(65)
        first = False
        print(f"[download] {name}")
        params = dict(latitude=lat, longitude=lon, start_date=START, end_date=END, daily=daily,
                      timezone="Asia/Colombo", format="csv")
        text = fetch(url, params)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"   saved {path} ({len(text) / 1024:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    main(ap.parse_args().force)
