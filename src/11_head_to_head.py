#!/usr/bin/env python3
"""Compare the deployed app's UC allocation against this study's model.

The two methods are scored on identical weeks against identical truth, and the
deployed rule is deliberately given every advantage:

* it is handed the **true city total** for the target week, which operationally
  it would never have -- its own city forecast would carry its own error;
* it is scored twice on burden weights. **As shipped** uses the exact weights in
  the live app's `rawalpindi_uc_forecast.geojson` -- but those were built from
  2021-2024 patient data, so scoring them on 2024 lets them see the season they
  are being tested on. **Pre-test** rebuilds the same rule from 2021-2023 only,
  which is what the deployed app would actually have had in hand at the start of
  the 2024 season. Both are reported; the pre-test row is the fair one;
* it uses the live thresholds (0.5 / 3.0 / 8.0) and the five-UC watch floor.

So every difference reported here is attributable to the allocation step alone.
The deployed app's code is read, never written.

Run: python src/11_head_to_head.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "app" / "data"
REPORTS = ROOT / "reports"
DEPLOYED = ROOT.parent / "Alert App" / "data" / "rawalpindi_uc_forecast.geojson"

# The live app's constants, from src/update_live_forecast.py.
UC_YELLOW, UC_ORANGE, UC_RED = 0.5, 3.0, 8.0
MIN_YELLOW_WATCH_UCS = 5
MIN_CITY_CASES_FOR_WATCH = 1.0


def deployed_weights() -> dict[str, float]:
    """Historical burden weights exactly as shipped in the live app.

    These were computed over 2021-2024, so on a 2024 test they already contain
    the answer. Kept for reference, not for the headline comparison.
    """
    geo = json.loads(DEPLOYED.read_text())
    features = [f for f in geo["features"]
                if f.get("properties", {}).get("tehsil") == "Rawalpindi Tehsil"]
    return {f["properties"]["uc"]: max(float(f["properties"].get("historical_cases", 0)), 0) + 1
            for f in features}


def pretest_weights(panel_path: Path, before_year: int) -> dict[str, float]:
    """The same rule rebuilt from seasons strictly before the test year.

    This is what the deployed app could honestly have known going into 2024, and
    it is the version the model should be compared against.
    """
    panel = pd.read_csv(panel_path)
    burden = panel[panel["Year"] < before_year].groupby("UC_name")["cases"].sum()
    return {uc: float(value) + 1 for uc, value in burden.items()}


def deployed_alert(expected: float) -> str:
    if expected >= UC_RED:
        return "Red"
    if expected >= UC_ORANGE:
        return "Orange"
    if expected >= UC_YELLOW:
        return "Yellow"
    return "Green"


def deployed_forecast(weights: dict, city_total: float, ucs: list[str]) -> dict:
    """One week of the live app's allocation, watch floor included."""
    total = sum(weights.get(uc, 1.0) for uc in ucs) or 1.0
    expected = {uc: city_total * weights.get(uc, 1.0) / total for uc in ucs}
    alerts = {uc: deployed_alert(value) for uc, value in expected.items()}

    if city_total >= MIN_CITY_CASES_FOR_WATCH:
        ranked = sorted(expected, key=lambda u: expected[u], reverse=True)
        for uc in ranked[:MIN_YELLOW_WATCH_UCS]:
            if alerts[uc] == "Green" and expected[uc] > 0:
                alerts[uc] = "Yellow"
    return {"expected": expected, "alerts": alerts}


def top_k(values: dict, k: int) -> set:
    return set(sorted(values, key=lambda u: values[u], reverse=True)[:k])


def main() -> None:
    payload = json.loads((APP_DATA / "forecasts.json").read_text())
    weights = deployed_weights()
    test_year = max(w["year"] for w in payload["weeks"])
    fair_weights = pretest_weights(ROOT / "data_processed" / "uc_weekly_panel.csv", test_year)
    print(f"deployed weights read from {DEPLOYED.parent.parent.name}/data/  "
          f"({len(weights)} UCs)\n")

    rows, ranking_snapshots = [], {}
    for week in payload["weeks"]:
        scored = [u for u in week["ucs"] if u["actual"] is not None]
        if not scored:
            continue
        ucs = [u["uc"] for u in scored]
        actual = {u["uc"]: float(u["actual"]) for u in scored}
        city_total = sum(actual.values())

        old = deployed_forecast(weights, city_total, ucs)
        old_fair = deployed_forecast(fair_weights, city_total, ucs)
        new = {u["uc"]: u["expected"] for u in scored}
        new_alerts = {u["uc"]: u["alert"] for u in scored}

        # The deployed rule is handed the true city total, so its grand total is
        # exact by construction while the model's carries forecast error. To
        # isolate *allocation* skill -- the only thing the two rules differ on --
        # the model's UC values are rescaled to the same true total.
        scale = city_total / sum(new.values()) if sum(new.values()) > 0 else 0.0
        new_scaled = {uc: value * scale for uc, value in new.items()}

        truth_top10 = top_k(actual, 10)
        rows.append({
            "year": week["year"], "week": week["week"], "city_actual": int(city_total),
            "old_MAE": float(np.mean([abs(old["expected"][u] - actual[u]) for u in ucs])),
            "fair_MAE": float(np.mean([abs(old_fair["expected"][u] - actual[u]) for u in ucs])),
            "fair_top10": len(top_k(old_fair["expected"], 10) & top_k(actual, 10)) / 10,
            "fair_caught": sum(actual[u] for u in ucs if old_fair["alerts"][u] in ("Red", "Orange")),
            "fair_red_orange": sum(1 for u in ucs if old_fair["alerts"][u] in ("Red", "Orange")),
            "new_MAE": float(np.mean([abs(new[u] - actual[u]) for u in ucs])),
            "new_MAE_scaled": float(np.mean([abs(new_scaled[u] - actual[u]) for u in ucs])),
            "old_top10": len(top_k(old["expected"], 10) & truth_top10) / 10,
            "new_top10": len(top_k(new, 10) & truth_top10) / 10,
            "old_red_orange": sum(1 for u in ucs if old["alerts"][u] in ("Red", "Orange")),
            "new_red_orange": sum(1 for u in ucs if new_alerts[u] in ("Red", "Orange")),
            "old_caught": sum(actual[u] for u in ucs if old["alerts"][u] in ("Red", "Orange")),
            "new_caught": sum(actual[u] for u in ucs if new_alerts[u] in ("Red", "Orange")),
        })
        ranking_snapshots[week["week"]] = {
            "old": sorted(old_fair["expected"], key=lambda u: old_fair["expected"][u], reverse=True)[:10],
            "new": sorted(new, key=lambda u: new[u], reverse=True)[:10],
            "truth": sorted(actual, key=lambda u: actual[u], reverse=True)[:10],
        }

    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "head_to_head.csv", index=False)

    total_cases = table["city_actual"].sum()
    print("=== accuracy over 25 weeks of the 2024 season ===")
    print("(the deployed rule is given the TRUE city total for every week)\n")
    print(f"{'':28s}{'DEPLOYED APP':>16s}{'NEW MODEL':>14s}{'change':>12s}")
    for label, old_key, new_key, fmt, better in [
        ("MAE/UC, own city total", "fair_MAE", "new_MAE", "{:.3f}", "lower"),
        ("MAE/UC, same city total", "fair_MAE", "new_MAE_scaled", "{:.3f}", "lower"),
        ("Top-10 hit rate", "fair_top10", "new_top10", "{:.1%}", "higher"),
    ]:
        old_value, new_value = table[old_key].mean(), table[new_key].mean()
        delta = (new_value - old_value) / old_value * 100
        arrow = "better" if ((delta < 0) == (better == "lower")) else "worse"
        print(f"{label:28s}{fmt.format(old_value):>16s}{fmt.format(new_value):>14s}"
              f"{delta:>+10.0f}%  {arrow}")

    print(f"\n  (for reference, the as-shipped weights -- which have seen 2024 -- "
          f"score MAE {table['old_MAE'].mean():.3f} and top-10 {table['old_top10'].mean():.1%})")

    old_catch = 100 * table["fair_caught"].sum() / total_cases
    new_catch = 100 * table["new_caught"].sum() / total_cases
    old_flagged, new_flagged = table["fair_red_orange"].mean(), table["new_red_orange"].mean()
    print(f"{'Cases caught by Red+Orange':28s}{old_catch:>15.1f}%{new_catch:>13.1f}%"
          f"{new_catch - old_catch:>+10.1f}pp  worse")
    print(f"{'UCs flagged Red+Orange/week':28s}{old_flagged:>16.1f}{new_flagged:>14.1f}"
          f"{100 * (new_flagged - old_flagged) / old_flagged:>+10.0f}%  fewer")
    print(f"{'Cases caught per UC flagged':28s}"
          f"{old_catch / old_flagged:>16.1f}{new_catch / new_flagged:>14.1f}"
          f"{100 * ((new_catch / new_flagged) / (old_catch / old_flagged) - 1):>+10.0f}%  better")

    # ---- the structural difference, shown rather than described ----
    print("\n=== does the map actually change week to week? ===")
    weeks = sorted(ranking_snapshots)
    first, peak = weeks[0], max(weeks, key=lambda w: table.set_index("week").loc[w, "city_actual"])
    for name in ("old", "new"):
        changes = sum(
            1 for a, b in zip(weeks, weeks[1:])
            if ranking_snapshots[a][name] != ranking_snapshots[b][name])
        label = "DEPLOYED" if name == "old" else "NEW"
        print(f"  {label:9s} top-10 list changed in {changes:2d} of {len(weeks) - 1} week transitions")

    print(f"\n=== top 5 UCs, quiet week {first} vs peak week {peak} ===")
    print(f"{'rank':<6}{'DEPLOYED w' + str(first):<22}{'DEPLOYED w' + str(peak):<22}"
          f"{'NEW w' + str(first):<22}{'NEW w' + str(peak):<22}")
    for i in range(5):
        print(f"{i + 1:<6}"
              f"{ranking_snapshots[first]['old'][i][:20]:<22}"
              f"{ranking_snapshots[peak]['old'][i][:20]:<22}"
              f"{ranking_snapshots[first]['new'][i][:20]:<22}"
              f"{ranking_snapshots[peak]['new'][i][:20]:<22}")

    print(f"\n=== peak week {peak}: who was actually worst, and who said so ===")
    snapshot = ranking_snapshots[peak]
    print(f"{'rank':<6}{'TRUTH':<24}{'DEPLOYED said':<24}{'NEW said':<24}")
    for i in range(8):
        print(f"{i + 1:<6}{snapshot['truth'][i][:22]:<24}"
              f"{snapshot['old'][i][:22]:<24}{snapshot['new'][i][:22]:<24}")

    print(f"\nwrote {REPORTS / 'head_to_head.csv'}")


if __name__ == "__main__":
    main()
