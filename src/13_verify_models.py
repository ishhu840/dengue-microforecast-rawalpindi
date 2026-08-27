#!/usr/bin/env python3
"""Check the committed models still behave before publishing a forecast.

CI installs whatever XGBoost is current. A model saved by one version and scored
by a much later one can load without complaint and quietly return different
numbers -- which, on a scheduled job that pushes straight to a public dashboard,
would ship wrong case counts with nobody watching.

So this is a regression test with teeth: it scores both boosters on a fixed
synthetic input and compares against fingerprints recorded when the models were
trained. Any drift fails the job, and the dashboard keeps showing the last good
forecast rather than a silently corrupted one.

Regenerate the fingerprints deliberately, after an intended retrain:

    python src/13_verify_models.py --update

Run: python src/13_verify_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FINGERPRINT = MODELS / "model_fingerprint.json"

# Tight enough to catch a real behaviour change, loose enough to survive
# last-bit floating point differences between platforms.
TOLERANCE = 1e-4


def probe(n_features: int, rows: int = 8) -> np.ndarray:
    """A fixed, arbitrary input matrix. Seeded, so it is identical everywhere."""
    rng = np.random.default_rng(20260827)
    return rng.uniform(0.0, 5.0, size=(rows, n_features))


def score(name: str, features: list[str]) -> list[float]:
    booster = xgb.Booster()
    booster.load_model(str(MODELS / name))
    matrix = xgb.DMatrix(probe(len(features)), feature_names=features)
    return [float(v) for v in booster.predict(matrix)]


def collect() -> dict:
    meta = json.loads((MODELS / "model_metadata.json").read_text())
    return {
        "xgboost": xgb.__version__,
        "count": {
            "n_features": len(meta["count_features"]),
            "predictions": score("uc_count_model.json", meta["count_features"]),
        },
        "onset": {
            "n_features": len(meta["onset_features"]),
            "predictions": score("uc_onset_model.json", meta["onset_features"]),
        },
    }


def main() -> None:
    current = collect()

    if "--update" in sys.argv or not FINGERPRINT.exists():
        FINGERPRINT.write_text(json.dumps(current, indent=2), encoding="utf-8")
        action = "updated" if "--update" in sys.argv else "created"
        print(f"fingerprint {action} (xgboost {current['xgboost']})")
        print(f"wrote {FINGERPRINT}")
        return

    expected = json.loads(FINGERPRINT.read_text())
    problems = []

    for model in ("count", "onset"):
        if current[model]["n_features"] != expected[model]["n_features"]:
            problems.append(
                f"{model}: feature count changed "
                f"{expected[model]['n_features']} -> {current[model]['n_features']}")
            continue
        drift = np.max(np.abs(
            np.array(current[model]["predictions"])
            - np.array(expected[model]["predictions"])))
        status = "ok" if drift <= TOLERANCE else "DRIFT"
        print(f"  {model:6s} {current[model]['n_features']:2d} features  "
              f"max drift {drift:.2e}  {status}")
        if drift > TOLERANCE:
            problems.append(f"{model}: predictions drifted by {drift:.2e} "
                            f"(tolerance {TOLERANCE:.0e})")

    if expected["xgboost"] != current["xgboost"]:
        print(f"  note: xgboost {expected['xgboost']} -> {current['xgboost']} "
              f"(version change alone is fine if predictions match)")

    if problems:
        print("\nMODEL VERIFICATION FAILED")
        for problem in problems:
            print(f"  - {problem}")
        print("\nThe published dashboard was left untouched. Investigate before "
              "forcing an update with --update.")
        sys.exit(1)

    print("\nmodels verified — safe to publish")


if __name__ == "__main__":
    main()
