#!/usr/bin/env python3
"""Produce the week-by-week forecasts the app displays.

There is no live surveillance feed for the current season, and inventing one
would make the app a mock-up. Instead this replays the 2024 season honestly:
for every week of that season the models are retrained on data strictly
earlier than the target week and asked to forecast it, then the answer is
stored next to what actually happened.

That gives the app something a static demo cannot -- every number on screen is
a real out-of-sample forecast that can be checked against the truth, and a user
can step through the season and watch the model succeed and fail.

The same function generates a live forecast the moment weekly counts start
arriving; `forecast_week` does not care whether the target week is historical
or in the future, only that no data from it is in the training frame.

Run: python src/06_generate_forecasts.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
MODELS = ROOT / "models"
APP_DATA = ROOT / "app" / "data"
SOURCE = ROOT.parent / "Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise"

REPLAY_YEAR = 2024
REPLAY_WEEKS = range(26, 51)

META = json.loads((MODELS / "model_metadata.json").read_text())
COUNT_FEATURES = META["count_features"]
ONSET_FEATURES = META["onset_features"]
NB_SIZE = META["nb_size_full"]
THRESHOLDS = META["alert_thresholds"]
SILENT_WEEKS = META["silent_weeks"]

COUNT_PARAMS = {
    "objective": "count:poisson", "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5.0,
    "reg_lambda": 2.0, "seed": 42,
}
ONSET_PARAMS = {
    "objective": "binary:logistic", "max_depth": 3, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5.0,
    "reg_lambda": 2.0, "seed": 42,
}


def alert_level(expected: float, onset_probability: float | None) -> str:
    """Colour for one UC-week.

    Expected cases set the level. A silent UC with a high onset probability is
    lifted to Watch, because "quiet but about to start" is the case a
    burden-weighted map can never show and the one a vector-control team most
    wants a week's notice on.
    """
    if expected >= THRESHOLDS["red"]:
        return "Red"
    if expected >= THRESHOLDS["orange"]:
        return "Orange"
    if expected >= THRESHOLDS["yellow"]:
        return "Yellow"
    if onset_probability is not None and onset_probability >= 0.25:
        return "Watch"
    return "Green"


def interval(mean: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.maximum(mean, 1e-6)
    probability = NB_SIZE / (NB_SIZE + mean)
    tail = (1.0 - level) / 2.0
    return (stats.nbinom.ppf(tail, NB_SIZE, probability),
            stats.nbinom.ppf(1.0 - tail, NB_SIZE, probability))


def city_interval(uc_means: np.ndarray, level: float) -> tuple[int, int]:
    """Interval for the city total, from the sum of the per-UC distributions.

    Applying the UC-level dispersion straight to the city total is wrong and
    badly so: it treats 87 neighbourhoods as one enormous UC and returns a band
    like 103-1311 around a forecast of 612. Errors partly cancel when counts are
    added, so the sum is proportionally tighter than its parts.

    Variances are added and a negative binomial is moment-matched to the total.
    This assumes UC errors are independent, which they are not -- a season that
    runs hot runs hot everywhere -- so the band is, if anything, a little narrow
    at the city level. Coverage is measured in script 07 rather than asserted.
    """
    mean = float(np.sum(uc_means))
    variance = float(np.sum(uc_means + uc_means ** 2 / NB_SIZE))
    tail = (1.0 - level) / 2.0
    if variance <= mean or mean <= 0:
        return int(stats.poisson.ppf(tail, max(mean, 1e-6))), \
               int(stats.poisson.ppf(1.0 - tail, max(mean, 1e-6)))
    size = mean ** 2 / (variance - mean)
    probability = size / (size + mean)
    return (int(stats.nbinom.ppf(tail, size, probability)),
            int(stats.nbinom.ppf(1.0 - tail, size, probability)))


def forecast_week(frame: pd.DataFrame, year: int, week: int) -> dict | None:
    """Forecast one target week using only rows that precede it.

    A row in the feature frame is stamped with the week its features come from
    and carries the following week as its target. So training on rows strictly
    before the target week means the model has never seen the answer, and the
    single row per UC stamped one week earlier is what gets scored.
    """
    target_start = pd.Timestamp.fromisocalendar(year, week, 1)
    origin_start = target_start - pd.Timedelta(days=7)

    history = frame[frame["week_start"] < origin_start]
    current = frame[frame["week_start"] == origin_start]
    if history.empty or current.empty:
        return None

    # Burden is recomputed from history alone at every step, so a UC that grows
    # through the season is reflected without the target week leaking in. It has
    # to be attached before the NaN drop, since it is one of the model's inputs.
    burden = history.groupby("UC_name")["cases"].sum()
    history = history.assign(uc_burden=history["UC_name"].map(burden).fillna(0.0))
    current = current.assign(uc_burden=current["UC_name"].map(burden).fillna(0.0))

    history = history.dropna(subset=["target"] + COUNT_FEATURES)
    current = current.dropna(subset=COUNT_FEATURES)
    if history.empty or current.empty:
        return None

    count_booster = xgb.train(
        COUNT_PARAMS, xgb.DMatrix(history[COUNT_FEATURES], label=history["target"]),
        num_boost_round=400)
    expected = np.maximum(count_booster.predict(xgb.DMatrix(current[COUNT_FEATURES])), 0.0)

    onset_probability = onset_scores(history, current)

    low80, high80 = interval(expected, 0.80)
    low95, high95 = interval(expected, 0.95)

    actual = current["target"].to_numpy(float)
    has_actual = not np.isnan(actual).all()

    ucs = []
    for i, (_, row) in enumerate(current.iterrows()):
        silent = all(row.get(f"own_lag{l}", 1) == 0 for l in range(1, SILENT_WEEKS + 1))
        probability = float(onset_probability[i]) if silent else None
        ucs.append({
            "uc": row["UC_name"],
            "expected": round(float(expected[i]), 2),
            "low80": int(low80[i]), "high80": int(high80[i]),
            "low95": int(low95[i]), "high95": int(high95[i]),
            "last_week": int(row["own_lag1"]),
            "onset_probability": round(probability, 3) if probability is not None else None,
            "alert": alert_level(float(expected[i]), probability),
            "actual": None if not has_actual or np.isnan(actual[i]) else int(actual[i]),
        })

    ucs.sort(key=lambda r: r["expected"], reverse=True)
    city_expected = float(expected.sum())
    city_low, city_high = city_interval(expected, 0.80)

    return {
        "year": year, "week": week,
        "week_start": target_start.date().isoformat(),
        "origin_week_start": origin_start.date().isoformat(),
        "city_expected": round(city_expected, 1),
        "city_low80": city_low, "city_high80": city_high,
        "city_actual": int(np.nansum(actual)) if has_actual else None,
        "training_weeks": int(history["week_start"].nunique()),
        "alert_counts": count_alerts(ucs),
        "ucs": ucs,
    }


def onset_scores(history: pd.DataFrame, current: pd.DataFrame) -> np.ndarray:
    """Probability that each currently-silent UC reports a case next week."""
    silent = np.ones(len(history), dtype=bool)
    for lag in range(1, SILENT_WEEKS + 1):
        silent &= history[f"own_lag{lag}"].fillna(-1).eq(0).to_numpy()
    training = history[silent].assign(onset=lambda d: (d["target"] > 0).astype(int))

    if training["onset"].nunique() < 2:
        return np.zeros(len(current))
    booster = xgb.train(
        ONSET_PARAMS, xgb.DMatrix(training[ONSET_FEATURES], label=training["onset"]),
        num_boost_round=300)
    return booster.predict(xgb.DMatrix(current[ONSET_FEATURES]))


def count_alerts(ucs: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in ucs:
        counts[row["alert"]] = counts.get(row["alert"], 0) + 1
    return counts


def uc_boundaries() -> dict:
    """Simplified UC polygons, small enough to ship inside the page."""
    import geopandas as gpd

    ucs = gpd.read_file(SOURCE / "rawalpindi_uc.geojson")
    ucs = ucs[ucs["tehsil"] == "Rawalpindi Tehsil"].copy()
    ucs["uc"] = ucs["uc"].astype(str).str.strip()
    ucs = ucs.dissolve(by="uc").reset_index()[["uc", "geometry"]]
    # ~40m tolerance: invisible at city zoom, roughly halves the payload.
    ucs["geometry"] = ucs.geometry.simplify(0.0004, preserve_topology=True)
    return json.loads(ucs.to_json())


def main() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PROCESSED / "uc_features.csv", parse_dates=["week_start"])

    weeks = []
    for week in REPLAY_WEEKS:
        result = forecast_week(frame, REPLAY_YEAR, week)
        if result is None:
            continue
        weeks.append(result)
        marker = "" if result["city_actual"] is None else f" actual {result['city_actual']:>4}"
        print(f"  {REPLAY_YEAR} w{week:02d}  expected {result['city_expected']:>6.1f}"
              f"  [{result['city_low80']}-{result['city_high80']}]{marker}")

    validation = pd.read_csv(ROOT / "reports" / "final_model_validation.csv")
    comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    onset_test = pd.read_csv(ROOT / "reports" / "onset_spatial_test.csv")
    decay = pd.read_csv(ROOT / "reports" / "distance_decay.csv")
    delays = pd.read_csv(PROCESSED / "reporting_delay.csv")

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "mode": "replay",
        "mode_note": (
            f"No live surveillance feed exists for the current season, so the app "
            f"replays {REPLAY_YEAR}. Every week shown was forecast using only data "
            f"from before it, and is displayed next to what actually happened."
        ),
        "alert_thresholds": THRESHOLDS,
        "nb_size": NB_SIZE,
        "weeks": weeks,
        "validation": validation.to_dict(orient="records"),
        "model_comparison": comparison[comparison["scope"] == "season w26-50"]
            .groupby("model", as_index=False)[["MAE", "RMSE", "PoissonDev", "Top10Hit"]]
            .mean().round(4).to_dict(orient="records"),
        "onset_spatial_test": onset_test.to_dict(orient="records"),
        "distance_decay": decay.to_dict(orient="records"),
        "reporting_delay": {
            "n": int(len(delays)),
            "median_days": float(delays["delay_days"].median()),
            "mean_days": round(float(delays["delay_days"].mean()), 1),
            "p90_days": float(delays["delay_days"].quantile(0.90)),
            "within_7_days_pct": round(float((delays["delay_days"] <= 7).mean() * 100), 1),
            "within_10_days_pct": round(float((delays["delay_days"] <= 10).mean() * 100), 1),
        },
    }

    # Scripts 09 and 10 are analysis-only and may not have run yet; their results
    # feed the app's findings panel when they exist.
    for key, name in (("knox", "knox_summary.json"),
                      ("microhotspot", "microhotspot_summary.json")):
        path = ROOT / "reports" / name
        if path.exists():
            payload[key] = json.loads(path.read_text())

    (APP_DATA / "forecasts.json").write_text(json.dumps(payload), encoding="utf-8")
    (APP_DATA / "uc_boundaries.geojson").write_text(json.dumps(uc_boundaries()), encoding="utf-8")

    size_mb = (APP_DATA / "forecasts.json").stat().st_size / 1e6
    geo_mb = (APP_DATA / "uc_boundaries.geojson").stat().st_size / 1e6
    print(f"\nweeks generated {len(weeks)}   forecasts.json {size_mb:.2f}MB   "
          f"boundaries {geo_mb:.2f}MB")


if __name__ == "__main__":
    main()
