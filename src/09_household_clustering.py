#!/usr/bin/env python3
"""Does dengue cluster locally at household scale, below UC resolution?

Script 04 found that UC-level correlation does not decay with distance, and
concluded the clustering seen in Moran's I is seasonal co-movement rather than
local spread. But that curve was measured between UC *centroids*, and Rawalpindi
UCs are kilometres across. Aedes aegypti disperses 100-200m. A real local
mechanism could be operating entirely inside single UCs, invisible to any
boundary-based analysis.

The 13,850 patients geolocated between 2021 and 2024 can settle it, because they
carry both a location and a date. (This is the raw coordinate set, slightly
larger than the 13,646 that fall inside a UC polygon -- the Knox test works on
the points themselves and needs no boundary at all.)

The test is Knox's: among all pairs of cases, are pairs that are close in space
*also* close in time more often than chance? This is the right instrument here
because it isolates space-time interaction. Cases obviously cluster in space --
people live in clusters -- and obviously cluster in time -- there is a season.
Neither of those alone implies transmission. Only the interaction does: if being
near someone raises your risk *shortly after* they fall ill, that is spread.

The null holds every location and every date fixed and only breaks their
pairing, so any static population clustering and any city-wide seasonality are
preserved exactly and cannot produce a positive result on their own.

Run: python src/09_household_clustering.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
REPORTS = ROOT / "reports"

YEARS = (2021, 2022, 2023, 2024)
SPACE_M = (100, 200, 500, 1000, 2000)
TIME_DAYS = (7, 14)
PERMUTATIONS = 499

# Household-tagged records put many patients at byte-identical coordinates.
# Those pairs are informative -- household transmission is real -- but they are
# also where duplicate data entry would hide, so they are reported separately
# rather than silently inflating the shortest distance band.
SAME_POINT_M = 1.0


def load_points() -> gpd.GeoDataFrame:
    """Geolocated Rawalpindi patients in metres, with an integer day stamp."""
    frame = pd.read_parquet(RAW / "line_list_rawalpindi.parquet")
    frame = frame.dropna(subset=["Latitude", "Longitude", "Entry Date"]).copy()
    frame["Year"] = frame["Entry Date"].dt.isocalendar().year.astype(int)
    frame = frame[frame["Year"].isin(YEARS)]

    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32643")

    points["x"] = points.geometry.x
    points["y"] = points.geometry.y
    points["day"] = (points["Entry Date"] - points["Entry Date"].min()).dt.days
    return points


def knox(x: np.ndarray, y: np.ndarray, day: np.ndarray,
         space_m: float, time_days: float) -> dict:
    """Knox space-time interaction statistic with a permutation null.

    Spatial pairs are found once with a KD-tree and then held fixed; the null
    reshuffles which date belongs to which location. That is what makes the test
    cheap enough to run on tens of thousands of cases, and it is also what makes
    it valid -- the spatial configuration and the epidemic curve are both
    preserved in every replicate.
    """
    tree = cKDTree(np.column_stack([x, y]))
    pairs = tree.query_pairs(space_m, output_type="ndarray")
    if len(pairs) == 0:
        return {}

    left, right = pairs[:, 0], pairs[:, 1]
    separation = np.hypot(x[left] - x[right], y[left] - y[right])
    distinct = separation > SAME_POINT_M

    def close_in_time(times: np.ndarray, mask: np.ndarray) -> int:
        return int(np.sum(np.abs(times[left[mask]] - times[right[mask]]) <= time_days))

    results = {}
    for label, mask in (("all", np.ones(len(pairs), bool)), ("distinct", distinct)):
        if mask.sum() == 0:
            continue
        observed = close_in_time(day, mask)
        null = np.empty(PERMUTATIONS)
        for i in range(PERMUTATIONS):
            null[i] = close_in_time(RNG.permutation(day), mask)
        expected = float(null.mean())
        results[label] = {
            "space_pairs": int(mask.sum()),
            "observed": observed,
            "expected": round(expected, 1),
            "ratio": round(observed / expected, 4) if expected else None,
            "excess_pairs": int(observed - expected),
            "p_value": round((np.sum(null >= observed) + 1) / (PERMUTATIONS + 1), 4),
        }
    return results


def same_point_summary(points: gpd.GeoDataFrame) -> pd.DataFrame:
    """How much of the signal sits at literally identical coordinates."""
    rows = []
    for year in YEARS:
        subset = points[points["Year"] == year]
        grouped = subset.groupby(["x", "y"]).size()
        rows.append({
            "year": year,
            "cases": len(subset),
            "distinct_locations": int(len(grouped)),
            "locations_with_2plus": int((grouped >= 2).sum()),
            "max_at_one_location": int(grouped.max()),
            "pct_sharing_a_location": round(
                100 * float(grouped[grouped >= 2].sum()) / len(subset), 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    points = load_points()
    print(f"geolocated patients {len(points):,} across {points['Year'].nunique()} seasons\n")

    duplicates = same_point_summary(points)
    duplicates.to_csv(REPORTS / "household_duplicate_locations.csv", index=False)
    print("=== how many patients share an exact coordinate ===")
    print(duplicates.to_string(index=False))

    rows = []
    for year in YEARS:
        subset = points[points["Year"] == year]
        if len(subset) < 200:
            continue
        x = subset["x"].to_numpy()
        y = subset["y"].to_numpy()
        day = subset["day"].to_numpy()
        for space in SPACE_M:
            for time in TIME_DAYS:
                for label, result in knox(x, y, day, space, time).items():
                    rows.append({"year": year, "space_m": space, "time_days": time,
                                 "pairs": label, **result})

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "knox_space_time.csv", index=False)

    print("\n=== Knox test: ratio of observed to expected close-in-both pairs ===")
    print("ratio > 1 means cases near each other in space are also near in time")
    print("more often than chance -- the signature of local transmission.\n")

    for pair_type in ("distinct", "all"):
        subset = table[table["pairs"] == pair_type]
        if subset.empty:
            continue
        note = ("excluding pairs at identical coordinates"
                if pair_type == "distinct" else "including same-household pairs")
        print(f"--- {note} ---")
        pivot = subset.pivot_table(index=["year", "time_days"],
                                   columns="space_m", values="ratio")
        print(pivot.round(3).to_string())
        significant = (subset["p_value"] < 0.05).mean()
        print(f"    tests with p < 0.05: {significant * 100:.0f}%"
              f"  ({int((subset['p_value'] < 0.05).sum())} of {len(subset)})\n")

    pooled = (table[table["pairs"] == "distinct"]
              .groupby(["space_m", "time_days"])
              .agg(mean_ratio=("ratio", "mean"),
                   min_p=("p_value", "min"), max_p=("p_value", "max"))
              .round(4).reset_index())
    print("=== pooled across seasons, distinct locations only ===")
    print(pooled.to_string(index=False))

    (REPORTS / "knox_summary.json").write_text(json.dumps({
        "permutations": PERMUTATIONS,
        "pooled": pooled.to_dict(orient="records"),
        "duplicate_locations": duplicates.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'knox_space_time.csv'}")


if __name__ == "__main__":
    main()
