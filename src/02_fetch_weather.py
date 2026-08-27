#!/usr/bin/env python3
"""Fetch and cache weekly Rawalpindi weather from Open-Meteo.

City-level weather, not UC-level. Rawalpindi's UCs sit inside roughly 25km of
each other, so a reanalysis grid cannot separate them -- every UC would receive
near-identical values and the feature would add noise while pretending to add
spatial detail. Weather therefore enters the model as a city-wide driver of
*when* the season turns, and the spatial structure comes from case history and
the neighbour graph instead.

This study fetches its own copy rather than reading the deployed app's data, so
it can be reproduced from this folder alone.

Run: python src/02_fetch_weather.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
PROCESSED = ROOT / "data_processed"

LAT, LON = 33.5651, 73.0169
# Reaches back a year before the panel so 52-week lags are defined in 2021.
START, END = "2019-01-01", "2025-06-30"

DAILY = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "precipitation_sum",
    "pressure_msl_mean", "wind_speed_10m_mean",
]

RENAME = {
    "temperature_2m_mean": "temp_mean",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "relative_humidity_2m_mean": "humidity",
    "precipitation_sum": "rain",
    "pressure_msl_mean": "pressure",
    "wind_speed_10m_mean": "wind",
}


def fetch_daily() -> pd.DataFrame:
    cache = RAW / "weather_daily.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    print(f"fetching Open-Meteo archive {START} to {END} ...")
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT, "longitude": LON,
            "start_date": START, "end_date": END,
            "daily": ",".join(DAILY), "timezone": "auto",
        },
        timeout=60,
    )
    response.raise_for_status()
    daily = pd.DataFrame(response.json()["daily"])
    daily["date"] = pd.to_datetime(daily.pop("time"))
    daily = daily.rename(columns=RENAME)

    RAW.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache, index=False)
    return daily


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    iso = daily["date"].dt.isocalendar()
    daily = daily.assign(Year=iso["year"].astype(int), Week=iso["week"].astype(int))
    weekly = daily.groupby(["Year", "Week"], as_index=False).agg(
        temp_mean=("temp_mean", "mean"),
        temp_max=("temp_max", "mean"),
        temp_min=("temp_min", "mean"),
        humidity=("humidity", "mean"),
        rain=("rain", "sum"),
        rain_days=("rain", lambda s: int((s >= 1.0).sum())),
        pressure=("pressure", "mean"),
        wind=("wind", "mean"),
        days=("date", "count"),
    )
    # Partial ISO weeks at either end of the fetch window would distort sums.
    return weekly[weekly["days"] == 7].drop(columns="days").reset_index(drop=True)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly = to_weekly(fetch_daily())
    weekly.to_csv(PROCESSED / "weather_weekly.csv", index=False)
    print(f"weekly rows {len(weekly)}  "
          f"{weekly['Year'].min()}w{weekly['Week'].iloc[0]} to "
          f"{weekly['Year'].max()}w{weekly['Week'].iloc[-1]}")
    print(weekly.tail(3).round(1).to_string(index=False))
    print(f"wrote {PROCESSED / 'weather_weekly.csv'}")


if __name__ == "__main__":
    main()
