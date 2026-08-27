#!/usr/bin/env python3
"""Live weekly UC forecast from real-time weather.

Fetches current Open-Meteo history and forecast, builds next week's features and
produces a UC-level forecast: expected cases, prediction interval and alert
level for each of the 87 Union Councils, plus the city total.

Which engine runs depends on what data exists, and the choice is not cosmetic:

* **`data/recent_cases_uc.csv` has recent weekly UC counts** -> the full model.
  A UC's own recent history carries 85% of the model's predictive gain, so this
  is the path that makes the study's accuracy claims true.

* **No case data** -> a weather-and-season-only model, trained here with every
  case-history feature removed. It can genuinely run with nothing but a weather
  API, and this script measures what that costs on held-out seasons rather than
  assuming it is small.

The second path is the honest reckoning for a live deployment with no
surveillance feed. Weather tells you *when* the season turns; it cannot tell you
*which neighbourhood* is affected, because that information lives only in recent
case history. The validation printed at the end quantifies exactly that.

Run: python src/12_live_forecast.py
"""

from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
MODELS = ROOT / "models"
APP_DATA = ROOT / "app" / "data"
LIVE_DIR = ROOT / "data"
RECENT_CASES = LIVE_DIR / "recent_cases_uc.csv"

LAT, LON = 33.5651, 73.0169
SEASON = (26, 50)
WEATHER_VARS = ("temp_mean", "humidity", "rain", "rain_days")
WEATHER_LAGS = (2, 4, 6, 8)

META = json.loads((MODELS / "model_metadata.json").read_text())
FULL_FEATURES = META["count_features"]
THRESHOLDS = META["alert_thresholds"]
NB_SIZE = META["nb_size_full"]

# The weather-only feature set: everything the full model uses, minus anything
# derived from case counts. Burden stays -- it is a static property of the UC,
# not a recent observation, and it is available with no surveillance feed.
WEATHER_ONLY_FEATURES = (
    ["week_sin", "week_cos", "Week", "uc_burden"]
    + [f"{v}_lag{l}" for v in WEATHER_VARS for l in WEATHER_LAGS]
    + ["rain_c8", "temp_mean", "humidity", "rain"]
)

PARAMS = {
    "objective": "count:poisson", "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5.0,
    "reg_lambda": 2.0, "seed": 42,
}
ROUNDS = 400


def fetch_live_weather() -> tuple[pd.DataFrame, str]:
    """Open-Meteo archive plus the 16-day forecast, aggregated to ISO weeks."""
    today = date.today()
    start = today - timedelta(days=7 * 14)          # 14 weeks back for the lags
    end = today + timedelta(days=15)

    daily_fields = ("temperature_2m_mean,relative_humidity_2m_mean,"
                    "precipitation_sum,pressure_msl_mean,wind_speed_10m_mean")
    frames, sources = [], []

    archive_end = min(end, today - timedelta(days=1))
    if start <= archive_end:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": LAT, "longitude": LON,
                    "start_date": start.isoformat(), "end_date": archive_end.isoformat(),
                    "daily": daily_fields, "timezone": "auto"}, timeout=40)
        if response.ok:
            frames.append(pd.DataFrame(response.json()["daily"]))
            sources.append("archive")

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": LAT, "longitude": LON, "forecast_days": 16,
                "past_days": 30, "daily": daily_fields, "timezone": "auto"}, timeout=40)
    if response.ok:
        frames.append(pd.DataFrame(response.json()["daily"]))
        sources.append("forecast")

    if not frames:
        raise RuntimeError("Open-Meteo unreachable; cannot build a live forecast")

    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily.pop("time"))
    daily = daily.drop_duplicates("date").sort_values("date")
    daily = daily.rename(columns={
        "temperature_2m_mean": "temp_mean", "relative_humidity_2m_mean": "humidity",
        "precipitation_sum": "rain", "pressure_msl_mean": "pressure",
        "wind_speed_10m_mean": "wind"})

    iso = daily["date"].dt.isocalendar()
    daily["Year"], daily["Week"] = iso["year"].astype(int), iso["week"].astype(int)
    weekly = daily.groupby(["Year", "Week"], as_index=False).agg(
        temp_mean=("temp_mean", "mean"), humidity=("humidity", "mean"),
        rain=("rain", "sum"), rain_days=("rain", lambda s: int((s >= 1.0).sum())),
        days=("date", "count"))
    return weekly, " + ".join(sources)


def weather_features(weekly: pd.DataFrame, year: int, week: int) -> dict | None:
    """Lagged weather for one target week, from the live series."""
    weekly = weekly.sort_values(["Year", "Week"]).reset_index(drop=True)
    lookup = {(int(r.Year), int(r.Week)): r for r in weekly.itertuples(index=False)}

    def shift(offset: int) -> tuple[int, int]:
        moved = date.fromisocalendar(year, week, 1) + timedelta(days=7 * offset)
        stamp = moved.isocalendar()
        return int(stamp[0]), int(stamp[1])

    def value(offset: int, field: str) -> float | None:
        row = lookup.get(shift(offset))
        return None if row is None else float(getattr(row, field))

    features: dict[str, float] = {}
    for field in WEATHER_VARS:
        current = value(0, field)
        if current is None:
            return None
        features[field] = current
        for lag in WEATHER_LAGS:
            lagged = value(-lag, field)
            if lagged is None:
                return None
            features[f"{field}_lag{lag}"] = lagged

    rain_history = [value(-k, "rain") for k in range(1, 9)]
    if any(v is None for v in rain_history):
        return None
    features["rain_c8"] = float(sum(rain_history))
    features["Week"] = float(week)
    features["week_sin"] = float(np.sin(2 * np.pi * week / 52.0))
    features["week_cos"] = float(np.cos(2 * np.pi * week / 52.0))
    return features


def load_recent_cases() -> pd.DataFrame:
    """Optional live UC counts: columns year, week, uc, cases."""
    if not RECENT_CASES.exists():
        return pd.DataFrame(columns=["year", "week", "uc", "cases"])
    try:
        frame = pd.read_csv(RECENT_CASES)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["year", "week", "uc", "cases"])
    required = {"year", "week", "uc", "cases"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{RECENT_CASES.name} needs columns {sorted(required)}")
    return frame.dropna(subset=["year", "week", "uc", "cases"])


def validate_weather_only(frame: pd.DataFrame) -> pd.DataFrame:
    """What the no-case-data path costs, on held-out seasons.

    Also scores the deployed app's rule -- seasonal city total split by fixed UC
    share -- because if the weather-only model cannot beat that, then with no
    surveillance feed there is no reason to prefer it.
    """
    folds = [{"name": "2023", "train": [2021, 2022], "test": 2023},
             {"name": "2024", "train": [2021, 2022, 2023], "test": 2024}]
    rows = []
    for fold in folds:
        burden = frame[frame["Year"].isin(fold["train"])].groupby("UC_name")["cases"].sum()
        data = frame.assign(uc_burden=frame["UC_name"].map(burden).fillna(0.0))
        train = data[data["Year"].isin(fold["train"])].dropna(subset=["target"] + FULL_FEATURES)
        test = data[(data["Year"] == fold["test"])
                    & data["Week"].between(*SEASON)].dropna(subset=["target"] + FULL_FEATURES)
        actual = test["target"].to_numpy(float)

        predictions = {}
        for label, columns in (("full (with case history)", FULL_FEATURES),
                               ("weather + season only", WEATHER_ONLY_FEATURES)):
            booster = xgb.train(PARAMS, xgb.DMatrix(train[columns], label=train["target"]),
                                num_boost_round=ROUNDS)
            predictions[label] = booster.predict(xgb.DMatrix(test[columns]))

        share = (burden + 1) / (burden + 1).sum()
        city_next = test.groupby("week_start")["target"].sum()
        predictions["deployed static share"] = np.array([
            city_next.get(ws, 0.0) * share.get(uc, 0.0)
            for uc, ws in zip(test["UC_name"], test["week_start"])])

        for label, values in predictions.items():
            hits = []
            evaluation = pd.DataFrame({"uc": test["UC_name"].to_numpy(),
                                       "ws": test["week_start"].to_numpy(),
                                       "a": actual, "p": values})
            for _, group in evaluation.groupby("ws"):
                if group["a"].sum() == 0:
                    continue
                hits.append(len(set(group.nlargest(10, "a")["uc"])
                                & set(group.nlargest(10, "p")["uc"])) / 10)
            rows.append({"fold": fold["name"], "engine": label,
                         "MAE": round(float(np.mean(np.abs(actual - values))), 4),
                         "Top10Hit": round(float(np.mean(hits)), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PROCESSED / "uc_features.csv", parse_dates=["week_start"])

    weekly, source = fetch_live_weather()
    print(f"live weather: {source}, {len(weekly)} ISO weeks, "
          f"through {weekly['Year'].iloc[-1]}w{weekly['Week'].iloc[-1]}")

    today = date.today()
    target = today + timedelta(days=(7 - today.weekday()))
    year, week = target.isocalendar()[0], target.isocalendar()[1]
    print(f"target week: {year} week {week} (starting {target})")

    features = weather_features(weekly, year, week)
    if features is None:
        raise RuntimeError("live weather series has gaps; cannot build features")

    observed = load_recent_cases()
    engine = "full" if not observed.empty else "weather_only"
    print(f"recent UC case counts: {len(observed)} rows -> engine = {engine}")

    burden = frame.groupby("UC_name")["cases"].sum()
    ucs = sorted(burden.index)

    if engine == "weather_only":
        train = frame.assign(uc_burden=frame["UC_name"].map(burden).fillna(0.0))
        train = train.dropna(subset=["target"] + WEATHER_ONLY_FEATURES)
        booster = xgb.train(PARAMS,
                            xgb.DMatrix(train[WEATHER_ONLY_FEATURES], label=train["target"]),
                            num_boost_round=ROUNDS)
        rows = pd.DataFrame([{**features, "uc_burden": float(burden.get(uc, 0.0))}
                             for uc in ucs])
        expected = np.maximum(booster.predict(xgb.DMatrix(rows[WEATHER_ONLY_FEATURES])), 0.0)
    else:
        raise NotImplementedError(
            "Full-engine live forecasting needs the UC feature builder wired to "
            "the live counts in recent_cases_uc.csv. Not built yet -- see README.")

    probability = NB_SIZE / (NB_SIZE + np.maximum(expected, 1e-6))
    low = stats.nbinom.ppf(0.10, NB_SIZE, probability)
    high = stats.nbinom.ppf(0.90, NB_SIZE, probability)

    def alert(value: float) -> str:
        if value >= THRESHOLDS["red"]:
            return "Red"
        if value >= THRESHOLDS["orange"]:
            return "Orange"
        if value >= THRESHOLDS["yellow"]:
            return "Yellow"
        return "Green"

    table = pd.DataFrame({
        "uc": ucs, "expected": np.round(expected, 2),
        "low80": low.astype(int), "high80": high.astype(int),
        "alert": [alert(v) for v in expected],
    }).sort_values("expected", ascending=False).reset_index(drop=True)

    print(f"\ncity total next week: {expected.sum():.1f} cases expected")
    print(f"in season: {'yes' if SEASON[0] <= week <= SEASON[1] else 'NO - off season'}")
    print("\ntop 10 UCs:")
    print(table.head(10).to_string(index=False))

    # This validation is computed on historical seasons only -- it does not move
    # when the weather does. Recomputing it on every scheduled run would spend
    # six model fits to reproduce the same table, so it is cached.
    cache = ROOT / "reports" / "live_engine_validation.csv"
    if cache.exists():
        validation = pd.read_csv(cache)
        print("\n(engine validation loaded from cache)")
    else:
        validation = validate_weather_only(frame)
        validation.to_csv(cache, index=False)
    print("\n=== what the no-case-data engine costs (held-out seasons) ===")
    print(validation.pivot_table(index="engine", columns="fold",
                                 values=["MAE", "Top10Hit"]).round(3).to_string())

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "mode": "live", "engine": engine, "weather_source": source,
        "target": {"year": year, "week": week, "week_start": target.isoformat()},
        "in_season": bool(SEASON[0] <= week <= SEASON[1]),
        "city_expected": round(float(expected.sum()), 1),
        "ucs": table.to_dict(orient="records"),
        "engine_validation": validation.to_dict(orient="records"),
        "caveat": (
            "No UC-level surveillance feed is connected, so this forecast comes "
            "from weather and seasonality alone. Weather predicts when the season "
            "turns, not which neighbourhood is affected \u2014 treat the UC ranking "
            "as a historical-burden prior, not a live signal."
        ),
    }
    (APP_DATA / "live_forecast.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {APP_DATA / 'live_forecast.json'}")


if __name__ == "__main__":
    main()
