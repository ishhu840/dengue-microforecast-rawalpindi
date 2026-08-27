# Study 5 — Union Council Level Weekly Dengue Forecasting for Rawalpindi

Weekly, neighbourhood-level dengue forecasting for Rawalpindi Tehsil: how many
cases next week, which of the 87 Union Councils they will fall in, and what
alert level each UC should carry.

This study is **self-contained and independent**. It reads the patient line list
and UC boundaries from `../Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise`
and writes nothing outside this folder. The deployed Rawalpindi Dengue Early
Alert App in `../Alert App` is not modified, not imported, and not affected.

## Why this study exists

The deployed alert app forecasts a city total and then splits it across UCs by
each UC's fixed historical share:

```
expected_uc = city_forecast × (historical_cases_uc + 1) / Σ(historical_cases + 1)
```

Those shares never change. The relative ranking of UCs is identical in every
week of every year — the map scales up and down but never reorders. This is a
reasonable first approximation, and it was built from the same patient data used
here, but it collapses a space-and-time dataset into one number per UC and
discards the time dimension entirely.

13,857 of those patient records carry a GPS coordinate, and 13,646 of those fall
inside a Union Council polygon. This study keeps the time dimension and asks what
becomes possible.

## What was found

**1. The neighbour graph does not help.** Adding an adjacency graph over UCs —
the city-scale analogue of the inter-regional dengue GNN literature — changed
onset AUC by −0.010 and +0.006 across two held-out seasons. Opposite signs,
neither significant (permutation p = 0.06, 0.23), and average precision fell in
both. Spatial features accounted for 4.7% of model gain against 84.9% for a
UC's own history.

**2. But that is a resolution artefact, and this is the study's central
finding.** The UC-level correlation does not fall with distance (0.287 at 0–2 km,
0.298 at 20 km+) — so at UC scale the clustering is just the monsoon arriving
over the whole city at once.

Repeat the measurement on the 13,850 household coordinates and the answer
reverses. A Knox space–time test finds case pairs close in space are also close
in time far more often than chance, and the excess **decays sharply with
distance**:

| Separation | Excess within 7 days |
|---|---|
| 100 m | **1.80×** |
| 200 m | 1.51× |
| 500 m | 1.27× |
| 2000 m | 1.09× |

All 40 tests significant at p = 0.002, in every season independently. It decays
with time too (1.80× at 7 days vs 1.47× at 14 days at 100 m) — the signature of
transmission, not of a static or seasonal artefact.

**Local transmission is real and strong at 100–200 m. Union Councils are
kilometres wide, so aggregating to UC totals averages it away.** The graph did
not fail because dengue lacks local structure; it failed because the analysis
unit is 10–20× coarser than the process.

The generalisable lesson: **before adopting or rejecting a spatial architecture,
measure the scale your process actually runs at and compare it to your analysis
unit.** A study that stopped at the UC-level null would have concluded, wrongly,
that dengue in Rawalpindi has no local spatial structure.

**3. Reporting is fast.** Median 5 days from symptom onset to confirmation, 92%
within 7 days, 98% within 10 (n = 8,433). Last week's counts are essentially
complete by the following Monday. This does not appear to be published elsewhere
for Rawalpindi.

**4. The alert map is worth acting on.** Over the replayed 2024 season, Red and
Orange together covered 6.2% of UC-weeks and contained 79.9% of all cases. Every
single Red and Orange UC-week had at least one case.

**5. Micro-hotspot targeting beats the UC map per unit of effort.** Rings drawn
around every case reported this week catch a disproportionate share of next
week's cases, over 83 in-season week pairs:

| Strategy | Catches | Area | Cases per km² |
|---|---|---|---|
| 200 m rings | 42.3% | 14.3 km² | **3.0** |
| 500 m rings | 72.9% | 55.2 km² | 1.3 |
| UC map, Red + Orange | 79.8% | 90.8 km² | 0.9 |

With capacity for a few km² a week, rings are **five times more area-efficient**
than UC-level alerting — and they need no model at all, only last week's
geocoded case list. With capacity for ~90 km², the UC map reaches more cases.
Different budgets, not competing methods.

**6. Uncertainty is measured, not assumed.** Negative binomial intervals, with
dispersion fitted on out-of-fold residuals. On UC-weeks that reported cases,
the 80% band covered 84.3% and the 95% band covered 95.5%.

## Results at a glance

| | |
|---|---|
| Patient records used | 13,646 GPS-located and matched to a UC, 2021–2024 |
| Point-in-polygon match rate | 98.5% |
| Union Councils | 87 (Rawalpindi Tehsil) |
| City forecast error, replayed 2024 season | MAE 49.6 cases/week, 22.2% at peak |
| Peak week predicted / actual | week 42 / week 44 |
| Top-10 UC hit rate | 87% (vs 83% for "same as last week" — see findings §3.3) |
| Onset recall (flagged the week before first case) | 40% (31 of 78) |
| 80% / 95% coverage, UC-weeks with cases | 84.3% / 95.5% |

Model comparison, mean over held-out seasons, transmission weeks 26–50:

| Model | MAE | Poisson dev. | Top-10 hit |
|---|---|---|---|
| Always zero | 1.532 | 69.451 | 30% |
| Persistence | 0.839 | 5.105 | 71% |
| Seasonal UC mean | 1.432 | 5.324 | 60% |
| Static share (deployed app's rule, given the true city total) | 0.889 | 1.202 | 52% |
| XGB own history | 0.831 | 1.011 | 55% |
| XGB own + spatial | 0.828 | 1.000 | 56% |
| **XGB own + weather** | **0.779** | **0.927** | 57% |
| XGB own + spatial + weather | 0.792 | 0.937 | 56% |

The static-share rule is handed the *true* city total for the week — information
it would not have in practice — and is still beaten.

## The app

`app/index.html` is a single self-contained file. No web server, no CDN, no map
tile provider, no external font. Open it by double-clicking.

Because no live surveillance feed exists for the current season, it **replays
the 2024 season**: for each of weeks 26–50 the models were retrained on data
strictly earlier than the target week, then asked to forecast it. Every number
on screen is a real out-of-sample forecast shown next to what actually happened.
Step through the weeks with the slider or arrow keys.

`forecast_week()` in `src/06_generate_forecasts.py` does not care whether the
target week is historical or future — it becomes a live forecast the moment
weekly counts start arriving.

## Running it

```bash
python src/01_build_uc_panel.py        # line list -> UC x week panel + adjacency + delays
python src/02_fetch_weather.py         # Open-Meteo weekly weather
python src/03_features_and_models.py   # features + 8-model comparison
python src/04_spatial_hypothesis.py    # Moran's I, distance decay, onset test
python src/05_train_final.py           # production models + calibrated intervals
python src/06_generate_forecasts.py    # 2024 replay forecasts
python src/07_evaluate_replay.py       # coverage, alert performance, onset lead
python src/09_household_clustering.py  # Knox space-time test on household points
python src/10_microhotspot_targeting.py# ring-vs-UC targeting trade-off
python src/08_build_app.py             # assemble app/index.html
```

python src/11_head_to_head.py           # deployed allocation rule vs this model
python src/12_live_forecast.py          # LIVE forecast from real-time weather
```

Scripts 09 and 10 are analysis only and do not feed the models; 08 reads their
output for the app's findings panel, so run 08 after them. Scripts 11 and 12 are
independent of the rest.

## Live mode (script 12)

`src/12_live_forecast.py` fetches current Open-Meteo history and forecast and
produces next week's UC forecast, writing `app/data/live_forecast.json`.

**It runs, and its UC ranking is not yet trustworthy.** With no UC-level case
feed the only thing distinguishing one UC from another is static historical
burden — so the weather-only engine reduces to the deployed app's fixed share,
and measures out that way:

| Engine | MAE 2023 | MAE 2024 | Top-10 hit 2023 | Top-10 hit 2024 |
|---|---|---|---|---|
| Deployed static share (given true city total) | 0.812 | 0.966 | 0.462 | 0.571 |
| Full model, with case history | **0.520** | 1.038 | **0.554** | **0.583** |
| Weather + season only | 0.847 | 1.535 | 0.462 | 0.575 |

The weather-only engine's top-10 hit rate is identical to the static share's, to
within noise, in both seasons. **Weather predicts when the season turns, not
which neighbourhood is affected.** That information exists only in recent case
counts.

To switch to the accurate engine, drop weekly counts into
`data/recent_cases_uc.csv` with columns `year,week,uc,cases`. The full live path
is not wired yet — script 12 raises `NotImplementedError` on that branch rather
than silently producing something unvalidated.

Requires `pandas numpy geopandas shapely xgboost scikit-learn scipy requests
pyarrow openpyxl`.

## Live deployment

The dashboard refreshes itself. `.github/workflows/refresh-forecast.yml` runs
daily at 02:40 UTC (07:40 PKT) and on manual dispatch:

1. `src/13_verify_models.py` — scores both committed models on a fixed input and
   compares against `models/model_fingerprint.json`. If a newer XGBoost in CI
   changes predictions, the job **fails here** and the published dashboard keeps
   the last good forecast rather than shipping silently wrong numbers.
2. `src/12_live_forecast.py` — fetches current Open-Meteo history and 16-day
   forecast, builds next week's features, writes `app/data/live_forecast.json`.
3. `src/08_build_app.py` — rebuilds `index.html` with the new numbers inlined.
4. Commits the changed files and deploys the repo root to GitHub Pages.

Daily rather than weekly because the 16-day weather forecast is revised daily;
the target week only rolls over on Mondays, but its numbers keep sharpening.

CI installs only `pandas numpy requests xgboost scipy` — the live path needs no
geospatial stack, because the UC polygons were converted to SVG paths when the
app was first built and are carried inside `app/data/uc_boundaries.geojson`.

Rebuilding the *models* (as opposed to refreshing the forecast) requires the raw
line list and the full dependency set, and is done locally — see **Running it**.

## Patient data is deliberately not in this repository

The source line list has one row per confirmed dengue patient, carrying the
**exact GPS coordinate of their home** along with age and sex. That file, and
anything derived from it at patient level, is excluded by `.gitignore`:

```
data_raw/          *.xlsx  *.parquet  **/line_list*  **/patients*
```

What *is* committed is aggregated to Union Council x week counts — 87 areas,
weekly totals — which cannot identify a household. `reporting_delay.csv` has had
its patient identifier dropped for the same reason.

If you clone this repo you can refresh and redeploy the forecast, but you cannot
rebuild the panel without separately obtaining the line list under whatever data
agreement governs it.

## Layout

```
src/                 numbered pipeline, run in order
data_raw/            cached line list and daily weather (regenerable)
data_processed/      UC panel, adjacency, features, weather, reporting delays
models/              trained boosters + metadata with calibration constants
reports/             every validation table this study produced
app/                 self-contained index.html and its data
docs/                methodology and findings
```

## Known limitations

- **Four seasons of GPS data.** Coordinates begin in 2021. Two validation folds
  is what that allows; a third year would materially strengthen the conclusions.
- **Two-thirds of records are geolocatable.** 13,857 of 29,912 Rawalpindi
  patients carry coordinates, and coverage varies by year (66–94%). If tagging
  is not missing-at-random across neighbourhoods, UC shares are biased.
- **The neighbour null result is a null result, not proof of absence.** With 87
  UCs over four seasons the study is powered to detect a moderate effect, not a
  small one.
- **City-level intervals slightly under-cover** (76% against a nominal 80%)
  because summing UC variances assumes independence between UCs, which does not
  hold in a season that runs hot everywhere.
- **No live feed.** This forecasts well when given recent counts. Obtaining a
  weekly count from the district health office remains the single highest-value
  step for operational use — of anything in this study or the deployed app.
- **The `#` in this folder's name** breaks `file://` URLs and some web servers.
  It matches the naming of the sibling studies, so it was kept, but rename the
  folder before serving the app over HTTP.
