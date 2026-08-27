# Methodology

## 1. Study design

A retrospective forecasting study using routinely collected dengue surveillance
records from Rawalpindi District, Punjab, Pakistan. The unit of analysis is the
Union Council week. The forecast horizon is one week ahead.

**Primary question.** Can weekly dengue incidence be forecast at Union Council
resolution accurately enough to direct vector-control activity?

**Secondary question.** Does inter-UC spatial structure — the neighbour graph —
improve that forecast beyond each UC's own history and city-wide seasonality?

The secondary question is the study's scientific contribution and is treated as
a hypothesis to be tested, not an architecture to be adopted.

## 2. Data

### 2.1 Case data

Line list of laboratory-confirmed dengue patients, Punjab Tier I–II districts,
2013–2025 (`Working File - UC Mapped.xlsx`, 80,686 records). Rawalpindi District
contributes 29,912.

Each record carries patient identifiers, an entry date, a confirmation date,
symptom onset date where recorded, free-text residential address, and — from
2021 — a residential GPS coordinate.

**Date handling.** The export mixes three encodings in one workbook: Excel
serials for Entry Date, serials inside an object column for Date of onset, and
text with a trailing timezone label for Confirmation Date
(`03/28/2025 09:14 AM PKT`). The timezone suffix defeats `pd.to_datetime`
silently, which makes Confirmation Date appear 100% present and 100%
unparseable. All three are handled explicitly in `excel_date()`.

Weekly aggregation uses **Entry Date** on ISO week boundaries. Entry Date is
complete for every record, whereas onset is recorded for 54% and reporting date
for 52%.

### 2.2 Geolocation

Patients are assigned to Union Councils by **point-in-polygon join against the
official boundaries**, not by the line list's own UC text field.

This choice is forced by the data. The free-text UC labels are entered by many
hands across many facilities and only about 11% match the official boundary
names — `R-79-DHOKE MUNSHEE` against `DHOK MUNSHI KHAN`, `Kulyal` against
`KULYAL`, and so on. The coordinates match at **98.5%** (13,646 of 13,857).

Coordinates exist only from 2021, so the panel covers 2021–2024. 2025 holds four
Rawalpindi records — the season had not begun when the extract was taken — and
is excluded.

| Year | Patients | With GPS | GPS % |
|---|---|---|---|
| 2021 | 2,507 | 2,093 | 83.5 |
| 2022 | 4,767 | 4,471 | 93.8 |
| 2023 | 2,653 | 1,762 | 66.4 |
| 2024 | 6,605 | 5,524 | 83.6 |

### 2.3 Boundaries

`rawalpindi_uc.geojson`, 182 Union Councils across six tehsils. The study covers
**Rawalpindi Tehsil**, which carries the outbreaks: 88 polygons under 87
distinct names (one UC is split across two rings and is dissolved to a single
feature).

### 2.4 Weather

Open-Meteo reanalysis for 33.5651 N, 73.0169 E, 2019–2025, aggregated to ISO
weeks: mean temperature, mean humidity, rainfall total, rainy days, pressure,
wind. Partial ISO weeks at either end of the fetch window are dropped so weekly
sums are comparable.

Weather enters as a **city-wide** driver. Rawalpindi's UCs lie within roughly 25
km of one another; a reanalysis grid cannot separate them, so per-UC weather
would add noise while appearing to add spatial resolution.

## 3. The panel

The full UC × ISO-week grid for 2021–2024: 87 UCs × 208 weeks = 18,096 rows,
12,688 cases.

**Absent UC-weeks are zeros, not missing values.** A UC with no patient in a
given week reported no cases; it did not fail to report. This matters for the
count models and makes the neighbour features well defined. 10.3% of the grid is
non-zero across the year, 21.2% within the transmission season.

## 4. Features

All features are computed from week *t* or earlier; the label is week *t+1*.

| Group | Features |
|---|---|
| Own history | cases at lags 1–4, rolling means over 2/4/8 weeks, week-on-week delta |
| City activity | city total at lags 1–2, 4-week rolling mean |
| Burden | cumulative UC cases over the training window only |
| Season | ISO week, sin/cos of week |
| Weather | temp, humidity, rain, rainy days at lags 2/4/6/8; 8-week cumulative rain |
| **Spatial** | neighbour sum / mean / max at lags 1–2, neighbour delta |

**Leakage control.** UC burden is recomputed from the training window at every
fold and, in the replay, at every week — computing it once over all years would
carry the test season's outbreak into the features. All rolling and cumulative
weather terms are shifted by at least one week.

### 4.1 The spatial graph

Adjacency is polygon contiguity: two UCs are connected if their boundaries
touch, with a 25 m buffer to absorb digitising slivers. 493 edges, mean degree
5.7.

Neighbour features are computed by pivoting to a UC × week matrix and
multiplying by the adjacency matrix — the same operation a graph convolution
performs, written out explicitly.

**The mobility half of the standard formulation is not available.** Workplace UC
is recorded for 3.8% of patients and permanent UC for 2.9%, far too sparse for
an origin–destination matrix. This is stated rather than approximated with a
gravity model, because at city scale a gravity prior would encode nothing beyond
population and distance, both of which the model already sees.

## 5. Models

Eight, under identical validation:

1. Always zero — floor, given 90% of the grid is zero
2. Persistence — next week equals this week
3. Seasonal UC mean — training-period mean for that UC and week-of-year
4. **Static share** — the deployed app's rule, `city_total × fixed UC share`
5. XGBoost, own history
6. XGBoost, own + weather
7. XGBoost, own + spatial
8. XGBoost, own + spatial + weather

Boosters use `count:poisson`, depth 4, learning rate 0.05, 400 rounds.

Model 4 is included because a new method should have to beat what is already
deployed. It is given the **true city total** for the target week — information
it would not have operationally — so any loss is attributable to the allocation
rule and not to an upstream city forecast.

## 6. Validation

**Rolling origin by season.** Train 2021–2022 → test 2023; train 2021–2023 →
test 2024. Two folds is what four seasons of GPS data permits.

**Metrics.** MAE, RMSE, Poisson deviance (a proper scoring rule for counts;
squared error is not), and a top-10 hit rate — of the ten UCs the map flags,
how many are truly in the week's worst ten. A health team can visit ten
neighbourhoods, not eighty-seven.

**Scope.** Headline metrics are reported on transmission weeks 26–50. Scoring
across all 52 weeks rewards a model for predicting zero in February. Whole-year
numbers are reported alongside.

## 7. Testing the spatial hypothesis

Aggregate error is a weak test: it is dominated by UCs already mid-outbreak,
where own history explains nearly everything. The spatial claim is about
*spread* — somewhere quiet becoming active. Three tests isolate it.

**7.1 Does clustering exist?** Moran's I on the log weekly case surface, with a
199-permutation null, for every in-season week with at least five cases.

**7.2 Does it decay with distance?** Pairwise correlation of UC weekly counts,
binned by distance between UC centroids. Local transmission implies decay; a
flat curve implies a shared external driver.

**7.3 Does it predict onset?** Among UC-weeks where the UC has been silent for
two consecutive weeks, a binary classifier for whether it reports a case next
week. Own-history features against the same plus the neighbour graph, compared
by ROC AUC and average precision, with a 2,000-draw paired permutation test on
the AUC difference.

## 8. Uncertainty

Prediction intervals are negative binomial. The booster supplies the mean; a
dispersion parameter is fitted by maximum likelihood with the mean held fixed.

**Dispersion is fitted on out-of-fold residuals.** Fitting on in-sample
predictions drives the estimate to the Poisson limit — the booster has already
absorbed the training noise — and produces intervals far too narrow to survive a
real season. In this study that error moved the fitted size from ~1.3 to the
1000 upper bound.

City-level intervals sum the per-UC variances and moment-match a negative
binomial to the total. Applying the UC-level dispersion directly to the city
total treats 87 neighbourhoods as one enormous UC and returns absurd bands
(103–1311 around a forecast of 612). The sum assumes independence across UCs,
which does not hold; measured city coverage is 76% against a nominal 80%,
consistent with that assumption being mildly optimistic.

## 9. Alert levels

Thresholds are quantiles of the observed in-season distribution of non-zero
UC-week counts, 2021–2024: Yellow at 1 case, Orange at 6, Red at 29.

A fifth level, **Watch**, is assigned to a UC that is currently silent but whose
onset probability exceeds 0.25. This is the state a burden-weighted map can
never represent — a fixed share says the same thing every week — and it is the
one a vector-control team most wants a week's notice of.

## 10. The replay

No live surveillance feed exists for the current season. Rather than mock one,
the app replays 2024: for each of weeks 26–50 the models are retrained on data
strictly earlier than the target week and asked to forecast it. Every displayed
number is a genuine out-of-sample forecast stored next to the observed outcome.

Weekly retraining is not only honest but better — the top-10 hit rate rises from
55% under fixed two-fold training to 87% under weekly retraining, because the
model sees the current season's trajectory as it develops.
