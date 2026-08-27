#!/usr/bin/env python3
"""Turn the household-scale clustering finding into an operational number.

Script 09 established that dengue in Rawalpindi does cluster in space and time,
strongly, at 100-200m -- the scale of Aedes dispersal, and far below the
kilometres-wide Union Councils the app forecasts on.

This asks the question a district team would ask next:

    If we treat a radius around every case reported this week as a
    micro-hotspot, how much of next week's cases does that catch,
    and how much ground does it cover?

The comparison is against the UC-level alert map from the same season, which
directs effort at whole Union Councils. Both are scored the same way: area
covered against cases caught.

Run: python src/10_microhotspot_targeting.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
REPORTS = ROOT / "reports"
APP_DATA = ROOT / "app" / "data"
SOURCE = ROOT.parent / "Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise"

RADII_M = (100, 200, 500)
SEASON = (26, 50)
YEARS = (2021, 2022, 2023, 2024)


def load_points() -> gpd.GeoDataFrame:
    frame = pd.read_parquet(RAW / "line_list_rawalpindi.parquet")
    frame = frame.dropna(subset=["Latitude", "Longitude", "Entry Date"]).copy()
    iso = frame["Entry Date"].dt.isocalendar()
    frame["Year"], frame["Week"] = iso["year"].astype(int), iso["week"].astype(int)
    frame = frame[frame["Year"].isin(YEARS)]
    return gpd.GeoDataFrame(
        frame, geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
        crs="EPSG:4326").to_crs("EPSG:32643")


def tehsil_area_km2() -> float:
    ucs = gpd.read_file(SOURCE / "rawalpindi_uc.geojson").to_crs("EPSG:32643")
    return float(ucs[ucs["tehsil"] == "Rawalpindi Tehsil"].geometry.area.sum() / 1e6)


def week_pairs(points: gpd.GeoDataFrame) -> list[tuple]:
    """Consecutive in-season week pairs that both carry cases."""
    keys = sorted({(int(y), int(w)) for y, w in zip(points["Year"], points["Week"])
                   if SEASON[0] <= w <= SEASON[1]})
    lookup = set(keys)
    return [(y, w, y, w + 1) for y, w in keys if (y, w + 1) in lookup]


def evaluate(points: gpd.GeoDataFrame, radius: float) -> pd.DataFrame:
    rows = []
    for year, week, next_year, next_week in week_pairs(points):
        current = points[(points["Year"] == year) & (points["Week"] == week)]
        following = points[(points["Year"] == next_year) & (points["Week"] == next_week)]
        if current.empty or following.empty:
            continue

        # Overlapping buffers are merged, so shared ground is paid for once.
        covered = unary_union(current.geometry.buffer(radius).values)
        inside = following.geometry.within(covered).sum()

        rows.append({
            "year": year, "week": week,
            "cases_this_week": len(current),
            "cases_next_week": len(following),
            "caught": int(inside),
            "pct_caught": round(100 * inside / len(following), 1),
            "area_km2": round(covered.area / 1e6, 3),
        })
    return pd.DataFrame(rows)


def uc_baseline() -> dict | None:
    """The same trade-off for the UC alert map, from the replayed season.

    Red and Orange UCs are the ones a team would visit. Their combined area is
    the cost; the share of the season's cases falling in them is the catch.
    """
    forecasts = APP_DATA / "forecasts.json"
    if not forecasts.exists():
        return None
    payload = json.loads(forecasts.read_text())

    ucs = gpd.read_file(SOURCE / "rawalpindi_uc.geojson").to_crs("EPSG:32643")
    ucs = ucs[ucs["tehsil"] == "Rawalpindi Tehsil"].copy()
    ucs["uc"] = ucs["uc"].astype(str).str.strip()
    area = ucs.dissolve(by="uc").geometry.area / 1e6

    caught = total = 0
    areas = []
    for week in payload["weeks"]:
        flagged = [u for u in week["ucs"] if u["alert"] in ("Red", "Orange")]
        actuals = [u["actual"] for u in week["ucs"] if u["actual"] is not None]
        if not actuals:
            continue
        caught += sum(u["actual"] for u in flagged if u["actual"] is not None)
        total += sum(actuals)
        areas.append(float(sum(area.get(u["uc"], 0.0) for u in flagged)))

    if not total:
        return None
    return {"pct_caught": round(100 * caught / total, 1),
            "mean_area_km2": round(float(np.mean(areas)), 1)}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    points = load_points()
    city_area = tehsil_area_km2()
    print(f"geolocated cases {len(points):,}   Rawalpindi Tehsil {city_area:,.0f} km2\n")

    frames = []
    print("=== ring around every case reported this week ===")
    print("catch = share of NEXT week's cases falling inside the rings\n")
    for radius in RADII_M:
        table = evaluate(points, radius)
        table["radius_m"] = radius
        frames.append(table)
        weighted = 100 * table["caught"].sum() / table["cases_next_week"].sum()
        mean_area = table["area_km2"].mean()
        print(f"  {radius:>4}m radius   catch {weighted:5.1f}%   "
              f"mean area {mean_area:6.2f} km2   "
              f"({100 * mean_area / city_area:4.1f}% of the tehsil)   "
              f"n={len(table)} week pairs")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(REPORTS / "microhotspot_targeting.csv", index=False)

    peak = combined[combined["cases_this_week"] >= 50]
    print("\n=== restricted to peak weeks (50+ cases reported) ===")
    for radius in RADII_M:
        subset = peak[peak["radius_m"] == radius]
        if subset.empty:
            continue
        weighted = 100 * subset["caught"].sum() / subset["cases_next_week"].sum()
        print(f"  {radius:>4}m radius   catch {weighted:5.1f}%   "
              f"mean area {subset['area_km2'].mean():6.2f} km2   n={len(subset)}")

    baseline = uc_baseline()
    if baseline:
        print("\n=== for comparison: UC alert map, 2024 replay ===")
        print(f"  Red+Orange UCs   catch {baseline['pct_caught']}%   "
              f"mean area {baseline['mean_area_km2']} km2   "
              f"({100 * baseline['mean_area_km2'] / city_area:.1f}% of the tehsil)")

    summary = {
        "city_area_km2": round(city_area, 1),
        "by_radius": [
            {
                "radius_m": radius,
                "pct_caught": round(float(
                    100 * combined.loc[combined["radius_m"] == radius, "caught"].sum()
                    / combined.loc[combined["radius_m"] == radius, "cases_next_week"].sum()), 1),
                "mean_area_km2": round(float(
                    combined.loc[combined["radius_m"] == radius, "area_km2"].mean()), 2),
            }
            for radius in RADII_M
        ],
        "uc_baseline": baseline,
    }
    (REPORTS / "microhotspot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'microhotspot_targeting.csv'}")


if __name__ == "__main__":
    main()
