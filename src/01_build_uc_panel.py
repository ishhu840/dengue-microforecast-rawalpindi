#!/usr/bin/env python3
"""Turn patient-level dengue records into a UC x week panel.

The line list holds one row per confirmed patient with a diagnosis date and,
from 2021 onward, a GPS coordinate for the residence. Every downstream model
in this study reads the panel this script writes; nothing else touches the
Excel file.

Two decisions here matter more than the rest:

**Location comes from coordinates, not from the UC text column.** The line
list's own UC labels are free text entered by many hands -- "R-79-DHOKE
MUNSHEE" against the map's "DHOK MUNSHI KHAN" -- and only about 11% of them
match the official boundary names. The coordinates match 98.5% under a
point-in-polygon join. So the text column is ignored entirely.

**Absent UC-weeks are real zeros, not missing data.** A UC with no patient in
week 32 reported no cases that week; it did not fail to report. The panel is
therefore built on the full UC x week grid and filled with zeros, which is
what makes the count models and the neighbour features well defined.

Reads (never writes) the source study folder. Run: python src/01_build_uc_panel.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise"
LINE_LIST = SOURCE / "Working File - UC Mapped.xlsx"
UC_GEOJSON = SOURCE / "rawalpindi_uc.geojson"

RAW = ROOT / "data_raw"
PROCESSED = ROOT / "data_processed"

# GPS tagging began in 2021; earlier years carry dates but no coordinates, so
# they cannot be placed on the map and are out of scope for a spatial model.
GPS_FIRST_YEAR = 2021
GPS_LAST_YEAR = 2024  # 2025 holds 4 Rawalpindi records -- the season had not started

# The app's operational area. The wider district (Gujar Khan, Murree, Kahuta,
# Taxila, Kotli Sattian) is kept in the case extract for context but the panel
# and the alert map cover the tehsil that carries the outbreaks.
TARGET_TEHSIL = "Rawalpindi Tehsil"

COLUMNS = [
    "Patient ID", "Entry Date", "District", "Latitude", "Longitude",
    "Tehsil", "UC", "Date of onset", "Confirmation Date", "Reporting Date",
    "Admission Date", "Age", "Gender",
]

# Excel stores dates as days since this epoch.
EXCEL_EPOCH = "1899-12-30"


def excel_date(series: pd.Series) -> pd.Series:
    """Parse the three date encodings the export mixes together.

    The same workbook stores Entry Date as an Excel serial (45744), Date of
    onset as serials held in an object column, and Confirmation Date as text
    with a trailing timezone label ("03/28/2025 09:14 AM PKT"). Pandas parses
    none of these uniformly, and the timezone suffix silently defeats
    to_datetime, which is what makes Confirmation Date look 100% present and
    100% unparseable at the same time.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.to_datetime(numeric, unit="D", origin=EXCEL_EPOCH, errors="coerce")

    text = series.astype(str).str.replace(
        r"\s*(PKT|PST|UTC|GMT)\s*$", "", regex=True).str.strip()
    parsed = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=False)
    return out.fillna(parsed).dt.normalize()


def load_line_list() -> pd.DataFrame:
    """Patient records, cached to parquet so later runs skip the 22MB read."""
    cache = RAW / "line_list_rawalpindi.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"reading {LINE_LIST.name} ...")
    frame = pd.read_excel(LINE_LIST, usecols=COLUMNS)
    frame["District"] = frame["District"].astype(str).str.strip().str.title()
    frame = frame[frame["District"] == "Rawalpindi"].copy()

    for column in ["Entry Date", "Date of onset", "Confirmation Date",
                   "Reporting Date", "Admission Date"]:
        frame[column] = excel_date(frame[column])

    # The free-text UC and Gender cells mix strings with stray numerics, which
    # parquet will not store. They are kept only for provenance, so cast.
    for column in ["UC", "Gender", "Tehsil"]:
        frame[column] = frame[column].astype(str)

    RAW.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def join_to_ucs(cases: pd.DataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Place each geolocated patient inside a Union Council polygon."""
    ucs = gpd.read_file(UC_GEOJSON).to_crs("EPSG:4326")
    ucs = ucs.rename(columns={"uc": "UC_name", "tehsil": "UC_tehsil"})
    ucs["UC_name"] = ucs["UC_name"].astype(str).str.strip()

    located = cases.dropna(subset=["Latitude", "Longitude"]).copy()
    points = gpd.GeoDataFrame(
        located,
        geometry=gpd.points_from_xy(located["Longitude"], located["Latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, ucs[["UC_name", "UC_tehsil", "geometry"]],
                       how="left", predicate="within")
    # A point on a shared border can match two polygons; keep the first.
    joined = joined[~joined.index.duplicated(keep="first")]

    inside = joined["UC_name"].notna()
    print(f"geolocated patients      {len(joined):,}")
    print(f"inside a UC polygon      {inside.sum():,}  ({100 * inside.mean():.1f}%)")
    return joined[inside].drop(columns="geometry"), ucs


def build_panel(cases: pd.DataFrame, ucs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Full UC x ISO-week grid of case counts, zeros included."""
    cases = cases.copy()
    iso = cases["Entry Date"].dt.isocalendar()
    cases["Year"], cases["Week"] = iso["year"].astype(int), iso["week"].astype(int)
    cases = cases[cases["Year"].between(GPS_FIRST_YEAR, GPS_LAST_YEAR)]
    cases = cases[cases["UC_tehsil"] == TARGET_TEHSIL]

    counts = (
        cases.groupby(["UC_name", "Year", "Week"])
        .size().rename("cases").reset_index()
    )

    # The grid: every UC of the tehsil against every ISO week in range, so a
    # quiet UC contributes zeros rather than gaps.
    uc_names = sorted(ucs.loc[ucs["UC_tehsil"] == TARGET_TEHSIL, "UC_name"].unique())
    weeks = pd.DataFrame(
        [(y, w) for y in range(GPS_FIRST_YEAR, GPS_LAST_YEAR + 1)
         for w in range(1, int(date_weeks_in_year(y)) + 1)],
        columns=["Year", "Week"],
    )
    grid = weeks.merge(pd.DataFrame({"UC_name": uc_names}), how="cross")
    panel = grid.merge(counts, on=["UC_name", "Year", "Week"], how="left")
    panel["cases"] = panel["cases"].fillna(0).astype(int)

    panel["week_start"] = [
        pd.Timestamp.fromisocalendar(int(y), int(w), 1)
        for y, w in zip(panel["Year"], panel["Week"])
    ]
    return panel.sort_values(["UC_name", "week_start"]).reset_index(drop=True)


def date_weeks_in_year(year: int) -> int:
    """ISO weeks in a year -- 52 or 53."""
    return pd.Timestamp(year, 12, 28).isocalendar()[1]


def build_adjacency(ucs: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict]:
    """Which UCs share a boundary.

    This is the study's spatial graph. It is the geographic half of the
    adjacency used in the inter-regional dengue GNN literature; the mobility
    half is unavailable here (workplace UC is recorded for 3.8% of patients),
    and at city scale it would mean less anyway -- neighbouring UCs co-outbreak
    because mosquitoes disperse ~100-200m and neighbourhoods share drainage and
    water storage, not because residents commute between them.

    A small buffer absorbs digitising slivers that leave true neighbours a
    fraction of a metre apart.
    """
    tehsil = ucs[ucs["UC_tehsil"] == TARGET_TEHSIL].reset_index(drop=True)
    metric = tehsil.to_crs("EPSG:32643")  # UTM 43N, metres
    buffered = metric.buffer(25)

    neighbours: dict[str, list[str]] = {}
    for i, name in enumerate(tehsil["UC_name"]):
        touching = buffered.index[buffered.intersects(buffered.iloc[i])]
        neighbours[name] = sorted(
            tehsil.loc[j, "UC_name"] for j in touching if j != i
        )

    rows = [
        {"UC_name": uc, "neighbour": nb}
        for uc, nbs in neighbours.items() for nb in nbs
    ]
    degrees = [len(v) for v in neighbours.values()]
    print(f"UCs in {TARGET_TEHSIL}       {len(tehsil)}")
    print(f"adjacency edges          {len(rows)}  (mean degree {np.mean(degrees):.1f})")
    isolated = [k for k, v in neighbours.items() if not v]
    if isolated:
        print(f"isolated UCs             {isolated}")
    return pd.DataFrame(rows), neighbours


def reporting_delay(cases: pd.DataFrame) -> pd.DataFrame:
    """Days from symptom onset to confirmation, per patient.

    This is the quantity that decides how stale a live forecast is on any given
    Monday, and it is not published for Rawalpindi anywhere we could find.
    """
    frame = cases.dropna(subset=["Date of onset", "Confirmation Date"]).copy()
    frame["delay_days"] = (frame["Confirmation Date"] - frame["Date of onset"]).dt.days
    # Negative or absurd values are data-entry noise, not short delays.
    frame = frame[frame["delay_days"].between(0, 60)]
    frame["Year"] = frame["Confirmation Date"].dt.isocalendar().year.astype(int)
    # Published as a distribution, not as one row per patient. The repository is
    # public, and a per-patient row carrying year, neighbourhood and an exact
    # delay would single out the only case in a quiet UC. Counts per (year,
    # delay) support every statistic this study reports -- median, mean, p90,
    # share confirmed within N days -- and single out nobody.
    return (frame.groupby(["Year", "delay_days"])
            .size().rename("patients").reset_index())


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    cases = load_line_list()
    print(f"Rawalpindi patient records {len(cases):,}")

    located, ucs = join_to_ucs(cases)
    panel = build_panel(located, ucs)
    edges, neighbours = build_adjacency(ucs)
    delays = reporting_delay(located)

    panel.to_csv(PROCESSED / "uc_weekly_panel.csv", index=False)
    edges.to_csv(PROCESSED / "uc_adjacency.csv", index=False)
    (PROCESSED / "uc_neighbours.json").write_text(
        json.dumps(neighbours, indent=1), encoding="utf-8")
    delays.to_csv(PROCESSED / "reporting_delay.csv", index=False)

    seasonal = panel[panel["Week"].between(26, 50)]
    print()
    print(f"panel rows               {len(panel):,}  ({panel['UC_name'].nunique()} UCs)")
    print(f"panel cases              {panel['cases'].sum():,}")
    print(f"non-zero UC-weeks        {(panel['cases'] > 0).sum():,}"
          f"  ({100 * (panel['cases'] > 0).mean():.1f}% of grid)")
    print(f"in weeks 26-50           {100 * (seasonal['cases'] > 0).mean():.1f}% non-zero")
    print(f"delay records            {len(delays):,}")
    print()
    print("cases by year")
    print(panel.groupby("Year")["cases"].sum().to_string())
    print()
    print(f"wrote {PROCESSED / 'uc_weekly_panel.csv'}")


if __name__ == "__main__":
    main()
