#!/usr/bin/env python3
"""Build UC-week features and test whether spatial information earns its place.

The question this study exists to answer:

    Given a Union Council's own case history, does knowing what its
    neighbouring UCs are doing improve next week's forecast?

That is the city-scale version of the claim made by the inter-regional dengue
GNN literature. It is worth testing rather than assuming, because the mechanism
differs. Between districts of a country, the graph carries *people*: an infected
traveller seeds a new region. Between UCs of one city, residents already mix
daily, so the graph cannot be about travel. If neighbours matter here it is
because Aedes aegypti disperses only 100-200m and adjacent neighbourhoods share
drainage, water storage and micro-climate -- the outbreak is one physical
cluster that the UC boundaries happen to cut in half.

Same arithmetic, different biology, and it might simply not hold. A negative
result is reported as a negative result.

Seven models are compared under rolling-origin validation, including the
allocation rule the deployed Rawalpindi alert app uses today, so the study can
say whether any of this beats what is already running.

Run: python src/03_features_and_models.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

# Dengue in Rawalpindi is a monsoon-tail disease. Scoring across all 52 weeks
# rewards a model for predicting zero in February, which no one needs. Headline
# metrics are reported on the transmission season; whole-year numbers are kept
# alongside so nothing is hidden.
SEASON = (26, 50)

# Two folds is what four years of GPS-tagged data allows. Each trains only on
# seasons strictly before the one it is scored on.
FOLDS = [
    {"name": "test 2023", "train": [2021, 2022], "test": 2023},
    {"name": "test 2024", "train": [2021, 2022, 2023], "test": 2024},
]

OWN_LAGS = (1, 2, 3, 4)
ROLL_WINDOWS = (2, 4, 8)
WEATHER_VARS = ("temp_mean", "humidity", "rain", "rain_days")
WEATHER_LAGS = (2, 4, 6, 8)

PARAMS = {
    "objective": "count:poisson", "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5.0,
    "reg_lambda": 2.0, "seed": 42,
}
ROUNDS = 400


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def build_features(panel: pd.DataFrame, neighbours: dict, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per UC-week, carrying only information available at that week.

    The target is the *next* week's count. Every feature is computed from week
    t or earlier and the label comes from t+1, so a row can never see its own
    answer.
    """
    frame = panel.sort_values(["UC_name", "week_start"]).reset_index(drop=True)
    grouped = frame.groupby("UC_name", group_keys=False)

    # Lags are numbered relative to the *target* week, not to this row: a row
    # stamped week t predicts t+1, so own_lag1 is week t's own count -- known at
    # forecast time -- and own_lag2 is t-1. The shift is therefore lag-1.
    for lag in OWN_LAGS:
        frame[f"own_lag{lag}"] = grouped["cases"].shift(lag - 1)
    for window in ROLL_WINDOWS:
        frame[f"own_roll{window}"] = grouped["cases"].transform(
            lambda s, w=window: s.rolling(w).mean())
    # Direction of travel, not just level: is this UC climbing or fading?
    frame["own_delta"] = frame["own_lag1"] - frame["own_lag2"]

    # City-wide activity. A UC's own count is noisy at these volumes; the city
    # signal tells the model whether the season is on at all.
    city = frame.groupby("week_start")["cases"].sum().rename("city_cases")
    frame = frame.merge(city, on="week_start", how="left")
    city_frame = frame.drop_duplicates("week_start")[["week_start", "city_cases"]].sort_values("week_start")
    city_frame["city_lag1"] = city_frame["city_cases"]
    city_frame["city_lag2"] = city_frame["city_cases"].shift(1)
    city_frame["city_roll4"] = city_frame["city_cases"].rolling(4).mean()
    frame = frame.merge(
        city_frame[["week_start", "city_lag1", "city_lag2", "city_roll4"]],
        on="week_start", how="left")

    frame = add_neighbour_features(frame, neighbours)

    # Weather is city-wide, joined on the week and lagged. Lags matter because
    # rain fills containers weeks before a human case is confirmed.
    weather = weather.sort_values(["Year", "Week"]).reset_index(drop=True)
    for var in WEATHER_VARS:
        for lag in WEATHER_LAGS:
            weather[f"{var}_lag{lag}"] = weather[var].shift(lag)
    weather["rain_c8"] = weather["rain"].shift(1).rolling(8).sum()
    frame = frame.merge(weather, on=["Year", "Week"], how="left")

    frame["week_sin"] = np.sin(2 * np.pi * frame["Week"] / 52.0)
    frame["week_cos"] = np.cos(2 * np.pi * frame["Week"] / 52.0)

    # The label: next week's count for this UC.
    frame["target"] = frame.groupby("UC_name")["cases"].shift(-1)
    return frame


def add_neighbour_features(frame: pd.DataFrame, neighbours: dict) -> pd.DataFrame:
    """What the adjacent UCs are doing, at the same week and one week before.

    Built by pivoting to a UC x week matrix and multiplying by the adjacency
    matrix -- the same operation a graph convolution performs, written out.
    """
    wide = frame.pivot_table(index="week_start", columns="UC_name",
                             values="cases", aggfunc="sum").fillna(0.0)
    ucs = list(wide.columns)
    index = {uc: i for i, uc in enumerate(ucs)}

    adjacency = np.zeros((len(ucs), len(ucs)))
    for uc, nbs in neighbours.items():
        if uc not in index:
            continue
        for nb in nbs:
            if nb in index:
                adjacency[index[uc], index[nb]] = 1.0

    counts = wide.to_numpy()
    degree = np.maximum(adjacency.sum(axis=1), 1.0)

    nb_sum = counts @ adjacency.T
    nb_mean = nb_sum / degree
    nb_max = np.stack([
        (counts * adjacency[i]).max(axis=1) if adjacency[i].any()
        else np.zeros(counts.shape[0]) for i in range(len(ucs))
    ], axis=1)

    def as_long(matrix: np.ndarray, name: str) -> pd.DataFrame:
        return (pd.DataFrame(matrix, index=wide.index, columns=ucs)
                .stack().rename(name).reset_index()
                .rename(columns={"level_1": "UC_name"}))

    parts = [as_long(nb_sum, "nb_sum1"), as_long(nb_mean, "nb_mean1"),
             as_long(nb_max, "nb_max1")]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["week_start", "UC_name"])

    frame = frame.merge(merged, on=["week_start", "UC_name"], how="left")
    ordered = frame.sort_values(["UC_name", "week_start"])
    for column in ["nb_sum1", "nb_mean1", "nb_max1"]:
        frame[column.replace("1", "2")] = ordered.groupby("UC_name")[column].shift(1).reindex(frame.index)
    # Is the neighbourhood heating up faster than this UC is?
    frame["nb_delta"] = frame["nb_mean1"] - frame["nb_mean2"]
    return frame


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    own = ([f"own_lag{l}" for l in OWN_LAGS]
           + [f"own_roll{w}" for w in ROLL_WINDOWS]
           + ["own_delta", "city_lag1", "city_lag2", "city_roll4",
              "week_sin", "week_cos", "Week", "uc_burden"])
    spatial = ["nb_sum1", "nb_mean1", "nb_max1", "nb_sum2", "nb_mean2",
               "nb_max2", "nb_delta"]
    weather = ([f"{v}_lag{l}" for v in WEATHER_VARS for l in WEATHER_LAGS]
               + ["rain_c8", "temp_mean", "humidity", "rain"])
    return {"own": own, "spatial": spatial, "weather": weather}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Proper scoring rule for counts; squared error is not one."""
    predicted = np.maximum(predicted, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(actual > 0, actual * np.log(actual / predicted), 0.0)
    return float(2.0 * np.mean(term - (actual - predicted)))


def hit_rate(evaluation: pd.DataFrame, k: int = 10) -> float:
    """Of the k UCs the model flags each week, what share are truly in the top k?

    This is the metric the app's map is actually judged on -- a health team can
    visit ten neighbourhoods, not eighty-seven.
    """
    scores = []
    for _, week in evaluation.groupby("week_start"):
        if week["actual"].sum() == 0:
            continue
        true_top = set(week.nlargest(k, "actual")["UC_name"])
        pred_top = set(week.nlargest(k, "pred")["UC_name"])
        scores.append(len(true_top & pred_top) / k)
    return float(np.mean(scores)) if scores else float("nan")


def score(evaluation: pd.DataFrame, label: str, fold: str, scope: str) -> dict:
    actual = evaluation["actual"].to_numpy(float)
    predicted = np.maximum(evaluation["pred"].to_numpy(float), 0.0)
    return {
        "model": label, "fold": fold, "scope": scope,
        "MAE": round(float(np.mean(np.abs(actual - predicted))), 4),
        "RMSE": round(float(np.sqrt(np.mean((actual - predicted) ** 2))), 4),
        "PoissonDev": round(poisson_deviance(actual, predicted), 4),
        "Top10Hit": round(hit_rate(evaluation), 4),
        "n": len(evaluation),
    }


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def run_fold(frame: pd.DataFrame, sets: dict, fold: dict) -> tuple[list[dict], pd.DataFrame]:
    train = frame[frame["Year"].isin(fold["train"])].copy()
    test = frame[frame["Year"] == fold["test"]].copy()

    # UC burden is a training-period statistic. Computing it over all years
    # would leak the test season's outbreak back into the features.
    burden = train.groupby("UC_name")["cases"].sum()
    share = (burden + 1) / (burden + 1).sum()
    for part in (train, test):
        part["uc_burden"] = part["UC_name"].map(burden).fillna(0.0)

    train = train.dropna(subset=["target"] + sets["own"])
    test = test.dropna(subset=["target"] + sets["own"])

    predictions: dict[str, np.ndarray] = {}
    actual = test["target"].to_numpy(float)

    # -- reference points -------------------------------------------------
    predictions["Always zero"] = np.zeros(len(test))
    predictions["Persistence"] = test["own_lag1"].to_numpy(float)

    seasonal = train.groupby(["UC_name", "Week"])["cases"].mean()
    predictions["Seasonal UC mean"] = np.array([
        seasonal.get((uc, wk), 0.0)
        for uc, wk in zip(test["UC_name"], test["Week"])
    ])

    # The deployed app's rule, handed the true city total for next week. It
    # cannot do better than this, so if it still loses, the loss is in the
    # allocation and not in the city forecast feeding it.
    city_next = test.groupby("week_start")["target"].sum()
    predictions["Static share (current app)"] = np.array([
        city_next.get(ws, 0.0) * share.get(uc, 0.0)
        for uc, ws in zip(test["UC_name"], test["week_start"])
    ])

    # -- learned models ---------------------------------------------------
    variants = {
        "XGB own history": sets["own"],
        "XGB own + weather": sets["own"] + sets["weather"],
        "XGB own + spatial": sets["own"] + sets["spatial"],
        "XGB own + spatial + weather": sets["own"] + sets["spatial"] + sets["weather"],
    }
    boosters = {}
    for label, columns in variants.items():
        dtrain = xgb.DMatrix(train[columns], label=train["target"])
        dtest = xgb.DMatrix(test[columns])
        booster = xgb.train(PARAMS, dtrain, num_boost_round=ROUNDS)
        predictions[label] = booster.predict(dtest)
        boosters[label] = booster

    rows = []
    for label, values in predictions.items():
        evaluation = pd.DataFrame({
            "UC_name": test["UC_name"].to_numpy(),
            "week_start": test["week_start"].to_numpy(),
            "Week": test["Week"].to_numpy(),
            "actual": actual, "pred": values,
        })
        rows.append(score(evaluation, label, fold["name"], "all weeks"))
        in_season = evaluation[evaluation["Week"].between(*SEASON)]
        rows.append(score(in_season, label, fold["name"], "season w26-50"))

    return rows, save_importance(boosters, fold)


def save_importance(boosters: dict, fold: dict) -> pd.DataFrame:
    booster = boosters["XGB own + spatial + weather"]
    gains = booster.get_score(importance_type="gain")
    return (pd.DataFrame({"feature": list(gains), "gain": list(gains.values())})
            .assign(fold=fold["name"])
            .sort_values("gain", ascending=False))


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PROCESSED / "uc_weekly_panel.csv", parse_dates=["week_start"])
    neighbours = json.loads((PROCESSED / "uc_neighbours.json").read_text())
    weather = pd.read_csv(PROCESSED / "weather_weekly.csv")

    frame = build_features(panel, neighbours, weather)
    sets = feature_sets(frame)
    frame.to_csv(PROCESSED / "uc_features.csv", index=False)
    print(f"feature rows {len(frame):,}  own={len(sets['own'])} "
          f"spatial={len(sets['spatial'])} weather={len(sets['weather'])}")

    rows, importances = [], []
    for fold in FOLDS:
        fold_rows, importance = run_fold(frame, sets, fold)
        rows += fold_rows
        importances.append(importance)

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "model_comparison.csv", index=False)
    pd.concat(importances).to_csv(REPORTS / "feature_importance.csv", index=False)

    order = ["Always zero", "Persistence", "Seasonal UC mean",
             "Static share (current app)", "XGB own history",
             "XGB own + weather", "XGB own + spatial",
             "XGB own + spatial + weather"]

    for scope in ["season w26-50", "all weeks"]:
        subset = results[results["scope"] == scope]
        table = subset.pivot_table(index="model", values=["MAE", "RMSE", "PoissonDev", "Top10Hit"])
        table = table.reindex(order)
        print(f"\n=== {scope} (mean over {len(FOLDS)} folds) ===")
        print(table[["MAE", "RMSE", "PoissonDev", "Top10Hit"]].round(4).to_string())

    print("\n=== per fold, season only ===")
    season = results[results["scope"] == "season w26-50"]
    print(season.pivot_table(index="model", columns="fold", values="MAE")
          .reindex(order).round(4).to_string())

    print(f"\nwrote {REPORTS / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
