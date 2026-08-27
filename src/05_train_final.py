#!/usr/bin/env python3
"""Train the two production models and calibrate their uncertainty.

Scripts 03 and 04 settled what the app should and should not do. The neighbour
graph did not earn a place, so it is not in the deployed feature set. What did
earn a place is a pair of models answering two different operational questions:

  1. **How many?**  A Poisson-objective gradient booster over a UC's own case
     history, city-wide activity and lagged weather. Best aggregate error of
     everything tested, and it beats the fixed historical-share allocation the
     current alert app uses even when that rule is handed a perfect city total.

  2. **Where next?** A classifier restricted to UCs that have been silent for
     two weeks, scoring the chance each lights up next week. This is the
     question a fixed share can never answer -- a share says the same thing
     every week -- and it reaches AUC 0.83-0.87 on held-out seasons.

Both are then given prediction intervals that are *measured*, not assumed. A
negative binomial dispersion is fitted on out-of-fold residuals and the
resulting 80% and 95% intervals are scored for coverage on held-out seasons, so
the app can state what fraction of weeks its band actually contains.

Run: python src/05_train_final.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

SEASON = (26, 50)
SILENT_WEEKS = 2
FOLDS = [{"name": "test 2023", "train": [2021, 2022], "test": 2023},
         {"name": "test 2024", "train": [2021, 2022, 2023], "test": 2024}]
ALL_YEARS = [2021, 2022, 2023, 2024]

COUNT_PARAMS = {
    "objective": "count:poisson", "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5.0,
    "reg_lambda": 2.0, "seed": 42,
}
ONSET_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc", "max_depth": 3,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 5.0, "reg_lambda": 2.0, "seed": 42,
}
ROUNDS = 400

WEATHER_VARS = ("temp_mean", "humidity", "rain", "rain_days")
WEATHER_LAGS = (2, 4, 6, 8)

COUNT_FEATURES = (
    ["own_lag1", "own_lag2", "own_lag3", "own_lag4",
     "own_roll2", "own_roll4", "own_roll8", "own_delta",
     "city_lag1", "city_lag2", "city_roll4",
     "week_sin", "week_cos", "Week", "uc_burden"]
    + [f"{v}_lag{l}" for v in WEATHER_VARS for l in WEATHER_LAGS]
    + ["rain_c8", "temp_mean", "humidity", "rain"]
)
ONSET_FEATURES = ["own_lag1", "own_lag2", "own_lag3", "own_lag4", "own_roll4",
                  "own_roll8", "city_lag1", "city_roll4", "uc_burden",
                  "week_sin", "week_cos"]


def add_burden(frame: pd.DataFrame, train_years: list[int]) -> pd.DataFrame:
    """Historical burden from the training window only, never the test season."""
    burden = frame[frame["Year"].isin(train_years)].groupby("UC_name")["cases"].sum()
    out = frame.copy()
    out["uc_burden"] = out["UC_name"].map(burden).fillna(0.0)
    return out


def out_of_fold_mean(train: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Predictions for each training year from a model that never saw it.

    Dispersion must be fitted on errors the model actually makes on unseen
    weeks. Fitting it on in-sample predictions drives the estimate to the
    Poisson limit -- the booster has already absorbed the training noise -- and
    yields intervals far too narrow to survive a real season.

    The first training year has no earlier year to learn from, so it is scored
    by the seasonal mean of the years that follow it; those rows contribute
    spread information without pretending to be a forecast.
    """
    years = sorted(train["Year"].unique())
    predictions = np.full(len(train), np.nan)
    position = {year: (train["Year"] == year).to_numpy() for year in years}

    for index, year in enumerate(years):
        mask = position[year]
        if index == 0:
            later = train[train["Year"] != year]
            seasonal = later.groupby(["UC_name", "Week"])["target"].mean()
            predictions[mask] = [
                seasonal.get((uc, wk), later["target"].mean())
                for uc, wk in zip(train.loc[mask, "UC_name"], train.loc[mask, "Week"])
            ]
            continue
        earlier = train[train["Year"].isin(years[:index])]
        booster = xgb.train(COUNT_PARAMS,
                            xgb.DMatrix(earlier[features], label=earlier["target"]),
                            num_boost_round=ROUNDS)
        predictions[mask] = booster.predict(xgb.DMatrix(train.loc[mask, features]))

    return np.maximum(predictions, 1e-6)


def fit_dispersion(actual: np.ndarray, mean: np.ndarray) -> float:
    """Negative binomial size parameter by maximum likelihood, mean held fixed.

    Counts here are overdispersed -- a handful of UC-weeks carry dozens of cases
    while most carry none -- so Poisson intervals would be far too narrow. The
    booster supplies the mean; this fits only the spread around it.
    """
    mean = np.maximum(mean, 1e-6)

    def negative_log_likelihood(log_size: float) -> float:
        size = np.exp(log_size)
        probability = size / (size + mean)
        return -np.sum(stats.nbinom.logpmf(actual, size, probability))

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=(np.log(1e-3), np.log(1e3)), method="bounded")
    return float(np.exp(result.x))


def nb_interval(mean: np.ndarray, size: float, level: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.maximum(mean, 1e-6)
    probability = size / (size + mean)
    tail = (1.0 - level) / 2.0
    return (stats.nbinom.ppf(tail, size, probability),
            stats.nbinom.ppf(1.0 - tail, size, probability))


def validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Held-out season performance, including measured interval coverage."""
    rows, dispersions = [], []

    for fold in FOLDS:
        data = add_burden(frame, fold["train"])
        train = data[data["Year"].isin(fold["train"])].dropna(subset=["target"] + COUNT_FEATURES)
        test = data[data["Year"] == fold["test"]].dropna(subset=["target"] + COUNT_FEATURES)

        booster = xgb.train(COUNT_PARAMS,
                            xgb.DMatrix(train[COUNT_FEATURES], label=train["target"]),
                            num_boost_round=ROUNDS)

        # Dispersion is fitted on out-of-fold training errors, then applied
        # unchanged to the test fold -- fitting it on the test season would
        # guarantee the coverage number and make it meaningless.
        size = fit_dispersion(train["target"].to_numpy(float),
                              out_of_fold_mean(train, COUNT_FEATURES))
        dispersions.append(size)

        season = test[test["Week"].between(*SEASON)]
        mean = booster.predict(xgb.DMatrix(season[COUNT_FEATURES]))
        actual = season["target"].to_numpy(float)

        record = {"fold": fold["name"], "n": len(season), "nb_size": round(size, 3),
                  "MAE": round(float(np.mean(np.abs(actual - mean))), 4),
                  "RMSE": round(float(np.sqrt(np.mean((actual - mean) ** 2))), 4)}
        for level in (0.80, 0.95):
            low, high = nb_interval(mean, size, level)
            record[f"PICP{int(level * 100)}"] = round(
                float(np.mean((actual >= low) & (actual <= high))), 4)
            record[f"MPIW{int(level * 100)}"] = round(float(np.mean(high - low)), 3)

        # onset model on the same fold
        silent_train = silent_rows(train)
        silent_test = silent_rows(season)
        if silent_train["onset"].nunique() > 1 and silent_test["onset"].nunique() > 1:
            onset_booster = xgb.train(
                ONSET_PARAMS,
                xgb.DMatrix(silent_train[ONSET_FEATURES], label=silent_train["onset"]),
                num_boost_round=300)
            scores = onset_booster.predict(xgb.DMatrix(silent_test[ONSET_FEATURES]))
            record["onset_AUC"] = round(roc_auc_score(silent_test["onset"], scores), 4)
            record["onset_n"] = len(silent_test)
        rows.append(record)

    return pd.DataFrame(rows), float(np.mean(dispersions))


def silent_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for lag in range(1, SILENT_WEEKS + 1):
        mask &= frame[f"own_lag{lag}"].fillna(-1).eq(0).to_numpy()
    out = frame[mask].copy()
    out["onset"] = (out["target"] > 0).astype(int)
    return out


def alert_thresholds(frame: pd.DataFrame) -> dict:
    """Alert cut points from the observed in-season distribution of UC-weeks.

    Anchored on what actually happens rather than on round numbers. Quantiles of
    the UC-weeks that reported at least one case: Yellow at the 25th percentile,
    Orange at the 75th, Red at the 95th -- which on 2021-2024 data comes out at
    1, 6 and 29 cases. Reported alongside how often each level would have fired.
    """
    season = frame[frame["Week"].between(*SEASON) & frame["Year"].isin(ALL_YEARS)]
    counts = season["cases"].to_numpy(float)
    active = counts[counts > 0]
    return {
        "yellow": float(np.round(np.quantile(active, 0.25), 2)),
        "orange": float(np.round(np.quantile(active, 0.75), 2)),
        "red": float(np.round(np.quantile(active, 0.95), 2)),
        "basis": "quantiles of non-zero UC-week counts, weeks 26-50, 2021-2024",
        "n_active_uc_weeks": int(len(active)),
    }


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(PROCESSED / "uc_features.csv", parse_dates=["week_start"])

    results, dispersion = validate(frame)
    results.to_csv(REPORTS / "final_model_validation.csv", index=False)
    print("=== held-out season validation ===")
    print(results.to_string(index=False))

    # ---- fit production models on everything ----
    data = add_burden(frame, ALL_YEARS)
    full = data.dropna(subset=["target"] + COUNT_FEATURES)
    count_booster = xgb.train(COUNT_PARAMS,
                              xgb.DMatrix(full[COUNT_FEATURES], label=full["target"]),
                              num_boost_round=ROUNDS)
    size = fit_dispersion(full["target"].to_numpy(float),
                          out_of_fold_mean(full, COUNT_FEATURES))

    silent = silent_rows(full[full["Week"].between(*SEASON)])
    onset_booster = xgb.train(ONSET_PARAMS,
                              xgb.DMatrix(silent[ONSET_FEATURES], label=silent["onset"]),
                              num_boost_round=300)

    count_booster.save_model(str(MODELS / "uc_count_model.json"))
    onset_booster.save_model(str(MODELS / "uc_onset_model.json"))

    burden = full.groupby("UC_name")["cases"].sum()
    thresholds = alert_thresholds(frame)
    metadata = {
        "count_features": COUNT_FEATURES,
        "onset_features": ONSET_FEATURES,
        "nb_size_full": round(size, 4),
        "nb_size_validation_mean": round(dispersion, 4),
        "silent_weeks": SILENT_WEEKS,
        "season": list(SEASON),
        "alert_thresholds": thresholds,
        "uc_burden": {k: int(v) for k, v in burden.items()},
        "validation": results.to_dict(orient="records"),
        "trained_on": "2021-2024 GPS-tagged Rawalpindi Tehsil UC-weeks",
        "spatial_features_excluded": (
            "Neighbour-graph features were tested (script 04) and did not "
            "improve either count error or onset AUC; they are excluded rather "
            "than carried as decoration."
        ),
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"NB dispersion (full fit)  size = {size:.3f}")
    print(f"alert thresholds          {thresholds['yellow']} / "
          f"{thresholds['orange']} / {thresholds['red']}  (Y/O/R)")
    print(f"onset training rows       {len(silent):,}  "
          f"({silent['onset'].mean() * 100:.1f}% positive)")
    print(f"\nwrote {MODELS / 'uc_count_model.json'}")
    print(f"wrote {MODELS / 'uc_onset_model.json'}")


if __name__ == "__main__":
    main()
