#!/usr/bin/env python3
"""Test the spatial hypothesis where it can actually be seen: outbreak onset.

Script 03 found that adding neighbour features to a UC-week forecast barely
moves aggregate error. That is a weak test of the spatial claim, because
aggregate error is dominated by UCs already mid-outbreak, where a UC's own
recent counts tell you nearly everything and no neighbour can add much.

The inter-regional GNN literature is not really claiming otherwise. Its claim
is about *spread* -- somewhere quiet becomes somewhere active because of what
is happening next door. That event is rare, it is invisible in a mean absolute
error over 87 UCs, and it is exactly what an early-warning app needs to catch.

So this script isolates it:

    Among UC-weeks where the UC has been silent for two weeks,
    does neighbour activity predict that it lights up next week?

Silent UCs are scored as a binary classification (lights up or not) with ROC
AUC and average precision, comparing own-history features against the same
features plus the neighbour graph. A permutation test on the AUC difference
says whether any gap is real at this sample size.

Two further diagnostics run alongside: Moran's I on the weekly case surface,
to establish whether spatial clustering exists at all before asking a model to
exploit it, and a distance-decay curve showing how far the neighbour signal
reaches.

Run: python src/04_spatial_hypothesis.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"
REPORTS = ROOT / "reports"
SOURCE = ROOT.parent / "Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise"

SEASON = (26, 50)
SILENT_WEEKS = 2          # how long a UC must be quiet to count as "at risk of onset"
FOLDS = [{"name": "test 2023", "train": [2021, 2022], "test": 2023},
         {"name": "test 2024", "train": [2021, 2022, 2023], "test": 2024}]

CLS_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc", "max_depth": 3,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 5.0, "reg_lambda": 2.0, "seed": 42,
}
ROUNDS = 300
PERMUTATIONS = 2000

OWN = ["own_lag1", "own_lag2", "own_lag3", "own_lag4", "own_roll4", "own_roll8",
       "city_lag1", "city_roll4", "uc_burden_static", "week_sin", "week_cos"]
SPATIAL = ["nb_sum1", "nb_mean1", "nb_max1", "nb_sum2", "nb_mean2", "nb_max2", "nb_delta"]


def load_features() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / "uc_features.csv", parse_dates=["week_start"])
    # A burden term that does not depend on the fold, so the onset subset can
    # be built once. It uses 2021-2022 only -- the earliest fold's training
    # window -- so it is never informed by a season it is scored on.
    early = frame[frame["Year"].isin([2021, 2022])]
    burden = early.groupby("UC_name")["cases"].sum()
    frame["uc_burden_static"] = frame["UC_name"].map(burden).fillna(0.0)
    return frame


def onset_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """UC-weeks that are currently silent, labelled by whether they light up."""
    silent = np.ones(len(frame), dtype=bool)
    for lag in range(1, SILENT_WEEKS + 1):
        silent &= frame[f"own_lag{lag}"].fillna(-1).eq(0).to_numpy()

    subset = frame[silent & frame["Week"].between(*SEASON)].copy()
    subset = subset.dropna(subset=["target"] + OWN + SPATIAL)
    subset["onset"] = (subset["target"] > 0).astype(int)
    return subset


def fit_auc(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[float, float, np.ndarray]:
    booster = xgb.train(
        CLS_PARAMS,
        xgb.DMatrix(train[columns], label=train["onset"]),
        num_boost_round=ROUNDS,
    )
    predicted = booster.predict(xgb.DMatrix(test[columns]))
    return (roc_auc_score(test["onset"], predicted),
            average_precision_score(test["onset"], predicted),
            predicted)


def permutation_pvalue(labels: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Is AUC(b) - AUC(a) larger than chance re-pairing would give?

    The two models score the same rows, so the comparison is paired: shuffle
    which of the pair each row's scores belong to and rebuild the difference.
    """
    observed = roc_auc_score(labels, b) - roc_auc_score(labels, a)
    count = 0
    for _ in range(PERMUTATIONS):
        swap = RNG.random(len(labels)) < 0.5
        pa = np.where(swap, b, a)
        pb = np.where(swap, a, b)
        if abs(roc_auc_score(labels, pb) - roc_auc_score(labels, pa)) >= abs(observed):
            count += 1
    return (count + 1) / (PERMUTATIONS + 1)


def morans_i(values: np.ndarray, weights: np.ndarray) -> float:
    """Spatial autocorrelation of one week's case surface."""
    deviation = values - values.mean()
    denominator = (deviation ** 2).sum()
    if denominator == 0 or weights.sum() == 0:
        return np.nan
    numerator = float(deviation @ weights @ deviation)
    return (len(values) / weights.sum()) * (numerator / denominator)


def spatial_autocorrelation(panel: pd.DataFrame, neighbours: dict) -> pd.DataFrame:
    """Moran's I week by week -- does clustering exist before we model it?"""
    wide = panel.pivot_table(index="week_start", columns="UC_name",
                             values="cases", aggfunc="sum").fillna(0.0)
    ucs = list(wide.columns)
    index = {uc: i for i, uc in enumerate(ucs)}
    weights = np.zeros((len(ucs), len(ucs)))
    for uc, nbs in neighbours.items():
        if uc in index:
            for nb in nbs:
                if nb in index:
                    weights[index[uc], index[nb]] = 1.0

    rows = []
    for week_start, values in wide.iterrows():
        iso = week_start.isocalendar()
        if not SEASON[0] <= iso[1] <= SEASON[1] or values.sum() < 5:
            continue
        # Log scale: raw counts are so skewed that one large UC dominates.
        observed = morans_i(np.log1p(values.to_numpy()), weights)
        null = [morans_i(RNG.permutation(np.log1p(values.to_numpy())), weights)
                for _ in range(199)]
        rows.append({
            "week_start": week_start, "Year": iso[0], "Week": iso[1],
            "cases": int(values.sum()), "morans_I": observed,
            "p_value": (np.sum(np.abs(null) >= abs(observed)) + 1) / 200,
        })
    return pd.DataFrame(rows)


def distance_decay(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlation between a UC's cases and other UCs', binned by distance.

    If transmission is a local physical cluster, correlation should fall away
    over a few kilometres. If it is flat, UC boundaries are not capturing
    anything spatial and the graph has nothing to carry.
    """
    ucs = gpd.read_file(SOURCE / "rawalpindi_uc.geojson").to_crs("EPSG:32643")
    ucs = ucs[ucs["tehsil"] == "Rawalpindi Tehsil"].copy()
    ucs["uc"] = ucs["uc"].astype(str).str.strip()
    # The boundary file carries 88 polygons under 87 distinct names -- one UC is
    # split across two rings. Merging them keeps one centroid per name, which is
    # also how the panel counts it.
    ucs = ucs.dissolve(by="uc").reset_index()
    centroids = ucs.set_index("uc").geometry.centroid

    wide = panel.pivot_table(index="week_start", columns="UC_name",
                             values="cases", aggfunc="sum").fillna(0.0)
    wide = wide[[c for c in wide.columns if c in centroids.index]]
    correlation = np.log1p(wide).corr()

    names = list(wide.columns)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({
                "km": centroids[a].distance(centroids[b]) / 1000.0,
                "corr": correlation.loc[a, b],
            })
    pairs = pd.DataFrame(rows).dropna()
    pairs["bin"] = pd.cut(pairs["km"], [0, 2, 4, 6, 8, 12, 20, 100])
    return (pairs.groupby("bin", observed=True)
            .agg(pairs=("corr", "size"), mean_corr=("corr", "mean"))
            .round(3).reset_index())


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame = load_features()
    panel = pd.read_csv(PROCESSED / "uc_weekly_panel.csv", parse_dates=["week_start"])
    neighbours = json.loads((PROCESSED / "uc_neighbours.json").read_text())

    # ---- 1. does spatial clustering exist at all? ----
    moran = spatial_autocorrelation(panel, neighbours)
    moran.to_csv(REPORTS / "morans_i_weekly.csv", index=False)
    significant = (moran["p_value"] < 0.05).mean()
    print("=== Moran's I, in-season weeks with >=5 cases ===")
    print(f"weeks tested        {len(moran)}")
    print(f"mean Moran's I      {moran['morans_I'].mean():.3f}")
    print(f"weeks p < 0.05      {significant * 100:.0f}%")

    # ---- 2. how far does the signal reach? ----
    decay = distance_decay(panel)
    decay.to_csv(REPORTS / "distance_decay.csv", index=False)
    print("\n=== correlation of weekly counts by distance between UCs ===")
    print(decay.to_string(index=False))

    # ---- 3. the onset test ----
    subset = onset_subset(frame)
    print(f"\n=== onset test: silent {SILENT_WEEKS}+ weeks, in season ===")
    print(f"candidate UC-weeks  {len(subset):,}   lit up next week "
          f"{subset['onset'].sum():,} ({100 * subset['onset'].mean():.1f}%)")

    rows = []
    for fold in FOLDS:
        train = subset[subset["Year"].isin(fold["train"])]
        test = subset[subset["Year"] == fold["test"]]
        if train["onset"].nunique() < 2 or test["onset"].nunique() < 2:
            continue
        auc_own, ap_own, p_own = fit_auc(train, test, OWN)
        auc_sp, ap_sp, p_sp = fit_auc(train, test, OWN + SPATIAL)
        pvalue = permutation_pvalue(test["onset"].to_numpy(), p_own, p_sp)
        rows.append({
            "fold": fold["name"], "n_test": len(test),
            "onsets": int(test["onset"].sum()),
            "AUC_own": round(auc_own, 4), "AUC_own_spatial": round(auc_sp, 4),
            "dAUC": round(auc_sp - auc_own, 4),
            "AP_own": round(ap_own, 4), "AP_own_spatial": round(ap_sp, 4),
            "p_permutation": round(pvalue, 4),
        })

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "onset_spatial_test.csv", index=False)
    print()
    print(results.to_string(index=False))
    print(f"\nwrote {REPORTS / 'onset_spatial_test.csv'}")


if __name__ == "__main__":
    main()
