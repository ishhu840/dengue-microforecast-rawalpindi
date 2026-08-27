#!/usr/bin/env python3
"""Score the 2024 replay: accuracy, interval coverage and alert usefulness.

Script 06 produced 25 genuine out-of-sample weekly forecasts. This scores them
the way an operational system should be scored -- not only "how close was the
number" but "would acting on this have helped".

Three things are measured:

* **Coverage.** What fraction of weeks the 80% and 95% bands actually contained,
  at UC level and at city level. This is the number the deployed app has never
  been able to state about its own band, and it is reported whatever it says.
* **Alert usefulness.** For each colour, how often it fired and what actually
  happened in those UC-weeks, plus how many of the true top-10 UCs the map put
  in its own top 10.
* **Onset lead time.** Of the UCs that started reporting cases during the
  season, how many were flagged Watch or higher the week before they did.

Run: python src/07_evaluate_replay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "app" / "data"
REPORTS = ROOT / "reports"

LEVELS = ["Red", "Orange", "Yellow", "Watch", "Green"]


def load_rows(payload: dict) -> pd.DataFrame:
    rows = []
    for week in payload["weeks"]:
        for uc in week["ucs"]:
            if uc["actual"] is None:
                continue
            rows.append({**uc, "year": week["year"], "week": week["week"]})
    return pd.DataFrame(rows)


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, subset in [("all UC-weeks", frame),
                          ("UC-weeks with cases", frame[frame["actual"] > 0])]:
        for level in (80, 95):
            inside = ((subset["actual"] >= subset[f"low{level}"])
                      & (subset["actual"] <= subset[f"high{level}"]))
            rows.append({
                "scope": label, "nominal": f"{level}%", "n": len(subset),
                "PICP": round(float(inside.mean()), 4),
                "MPIW": round(float((subset[f"high{level}"] - subset[f"low{level}"]).mean()), 2),
            })
    return pd.DataFrame(rows)


def city_coverage(payload: dict) -> dict:
    weeks = [w for w in payload["weeks"] if w["city_actual"] is not None]
    inside = [w["city_low80"] <= w["city_actual"] <= w["city_high80"] for w in weeks]
    actual = np.array([w["city_actual"] for w in weeks], float)
    predicted = np.array([w["city_expected"] for w in weeks], float)
    return {
        "weeks": len(weeks),
        "PICP80": round(float(np.mean(inside)), 4),
        "MAE": round(float(np.mean(np.abs(actual - predicted))), 1),
        "MAPE_active": round(float(np.mean(
            np.abs(actual[actual >= 10] - predicted[actual >= 10]) / actual[actual >= 10]) * 100), 1),
        "peak_week_actual": int(np.argmax(actual) + weeks[0]["week"]),
        "peak_week_predicted": int(np.argmax(predicted) + weeks[0]["week"]),
    }


def alert_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in LEVELS:
        subset = frame[frame["alert"] == level]
        if subset.empty:
            continue
        rows.append({
            "alert": level,
            "UC_weeks_fired": len(subset),
            "share_of_all": round(100 * len(subset) / len(frame), 1),
            "mean_actual_cases": round(float(subset["actual"].mean()), 2),
            "pct_with_any_case": round(float((subset["actual"] > 0).mean() * 100), 1),
            "pct_with_5plus": round(float((subset["actual"] >= 5).mean() * 100), 1),
            "total_actual_cases": int(subset["actual"].sum()),
        })
    table = pd.DataFrame(rows)
    table["pct_of_all_cases"] = (
        100 * table["total_actual_cases"] / frame["actual"].sum()).round(1)
    return table


def top_k_hit(frame: pd.DataFrame, k: int = 10) -> float:
    scores = []
    for _, week in frame.groupby(["year", "week"]):
        if week["actual"].sum() == 0:
            continue
        truth = set(week.nlargest(k, "actual")["uc"])
        flagged = set(week.nlargest(k, "expected")["uc"])
        scores.append(len(truth & flagged) / k)
    return round(float(np.mean(scores)), 4)


def onset_lead(frame: pd.DataFrame) -> dict:
    """Was a UC flagged the week before its first case of the season?"""
    frame = frame.sort_values(["uc", "week"])
    flagged, missed = 0, 0
    for uc, group in frame.groupby("uc"):
        active = group[group["actual"] > 0]
        if active.empty:
            continue
        first = active.iloc[0]
        # only counts as an onset if the UC was silent going into that week
        if first["last_week"] != 0:
            continue
        if first["alert"] in ("Watch", "Yellow", "Orange", "Red"):
            flagged += 1
        else:
            missed += 1
    total = flagged + missed
    return {
        "first_activations": total,
        "flagged_week_before": flagged,
        "recall": round(flagged / total, 4) if total else None,
    }


def main() -> None:
    payload = json.loads((APP_DATA / "forecasts.json").read_text())
    frame = load_rows(payload)
    print(f"scored UC-weeks {len(frame):,} across {frame['week'].nunique()} weeks\n")

    cover = coverage(frame)
    cover.to_csv(REPORTS / "replay_coverage.csv", index=False)
    print("=== interval coverage (UC level) ===")
    print(cover.to_string(index=False))

    city = city_coverage(payload)
    print("\n=== city level ===")
    for key, value in city.items():
        print(f"  {key:22s} {value}")

    alerts = alert_table(frame)
    alerts.to_csv(REPORTS / "replay_alert_performance.csv", index=False)
    print("\n=== alert level performance ===")
    print(alerts.to_string(index=False))

    lead = onset_lead(frame)
    print("\n=== onset detection ===")
    for key, value in lead.items():
        print(f"  {key:22s} {value}")

    hit = top_k_hit(frame)
    print(f"\n  top-10 hit rate        {hit}")

    summary = {"city": city, "onset": lead, "top10_hit": hit,
               "coverage": cover.to_dict(orient="records"),
               "alerts": alerts.to_dict(orient="records")}
    (REPORTS / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The app shows these, so they travel with the payload.
    payload["replay_evaluation"] = summary
    (APP_DATA / "forecasts.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'replay_summary.json'}")


if __name__ == "__main__":
    main()
