# F1 Intelligence

A backend for Formula 1 analytics, built in phases as a way to learn data science and ML on
real data rather than a tutorial dataset.

**Four features, built easiest to hardest:**

| Phase | Feature | Status |
|---|---|---|
| 1 | Data foundation — ingest FastF1 timing data into Postgres | **done** |
| 2 | Tire degradation model | **done** |
| 3 | Pit stop strategy simulator | **done** |
| 4 | Race prediction | **done** |
| 5 | Dashboard API | **done** |

All five phases are complete: **22 endpoints, 279 tests, 6 seasons of data.**

The order is deliberate. Tire degradation is a clean regression problem with ~150k rows per
season, which makes it the right first model. The pit simulator consumes the tire model. Race
prediction comes last because ~24 races a year means ~190 races of training data in total, and
it is full of data-leakage traps that are far easier to recognise once you have done honest
validation on an easier problem.

---

## Setup

Postgres 16 must be running. On macOS with Homebrew it already is, as a login service.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,notebook]"

createdb f1_intelligence
createdb f1_intelligence_test
for db in f1_intelligence f1_intelligence_test; do
  psql -d $db -c "CREATE SCHEMA IF NOT EXISTS raw;
                  CREATE SCHEMA IF NOT EXISTS features;
                  CREATE SCHEMA IF NOT EXISTS ml;"
done

cp .env.example .env     # edit if your Postgres user differs
.venv/bin/alembic upgrade head
```

## Ingest data

```bash
# One session
.venv/bin/python -m app.ingestion.cli session 2024 Monza R

# Races and qualifying for a season (start here -- most value per minute)
.venv/bin/python -m app.ingestion.cli season 2024 --only R,Q

# Everything, including practice
.venv/bin/python -m app.ingestion.cli season 2024

# What's in the database?
.venv/bin/python -m app.ingestion.cli status

# Data-quality checks -- run after every backfill
.venv/bin/python -m app.ingestion.cli doctor
```

Every command is safe to re-run: the loader upserts on natural keys. That matters because
FastF1 revises timing data for a while after a session.

The first run of a season is network-bound. FastF1 caches responses under `data/fastf1_cache`,
so re-runs are fast.

## Run the API

```bash
.venv/bin/uvicorn app.main:app --reload
```

Interactive docs at <http://localhost:8000/docs>.

```
GET /health
GET /api/v1/seasons
GET /api/v1/seasons/{year}/events
GET /api/v1/events/{event_id}/sessions
GET /api/v1/sessions/{id}/laps       ?driver=&compound=&accurate_only=&green_only=
GET /api/v1/sessions/{id}/results
GET /api/v1/sessions/{id}/stints
GET /api/v1/sessions/{id}/pit-stops
GET /api/v1/drivers

POST /api/v1/tire/degradation        Predicted lap-time loss as a tire ages
GET  /api/v1/tire/stints/{id}        Measured per-stint degradation (Stage A OLS)
GET  /api/v1/tire/model-info         Active model, holdout metrics, baseline

POST /api/v1/strategy/simulate       Rank pit strategies, deterministic + Monte Carlo
POST /api/v1/strategy/compare        Your own strategies, head to head
GET  /api/v1/strategy/inputs/{c}     The measured inputs driving a circuit

GET  /api/v1/prediction/race/{id}    What the model says about a past race
POST /api/v1/prediction/race         Predict an upcoming race from a grid
GET  /api/v1/prediction/backtest     Walk-forward metrics, beside the baselines

GET  /api/v1/dashboard/season/{year} Standings, constructors, per-circuit pace
GET  /api/v1/dashboard/driver/{abbr} Career record, best and worst circuits
GET  /api/v1/dashboard/race/{id}     Result, stint chart, prediction vs actual
GET  /api/v1/dashboard/models        Every registered model and its metrics
```

## Train the prediction model

```bash
.venv/bin/python -m app.ml.prediction.train
```

Runs a leakage audit first and **refuses to train if it fails**, then walk-forward
backtests Ridge / RandomForest / XGBoost over 81 races and registers the winner.

## Refresh the dashboard

```bash
.venv/bin/python -m app.ingestion.cli refresh-views
```

Rebuilds the three materialized views. Run after every ingest — plain views are always
current, only materialized ones go stale.

## Train the tire model

```bash
.venv/bin/python -m app.ml.tire.train
```

Builds `features.stint_laps`, fits Ridge / XGBoost / HistGradientBoosting, validates with
`GroupKFold` by session plus a held-out season, and registers the result in `ml.model_versions`.
Artifacts land in `models/` (gitignored) — XGBoost as JSON, sklearn as joblib.

## Tests

```bash
.venv/bin/python -m pytest
```

Tests run against the real `f1_intelligence_test` Postgres database, each inside a transaction
that is rolled back. Running on the same engine as production means array types, `ON CONFLICT`
semantics and CHECK constraints are all genuinely exercised.

## Notebooks

```bash
.venv/bin/jupyter lab
```

`notebooks/01_explore_fastf1.ipynb` walks through the raw data and the places it will mislead you.
Read it before trusting anything downstream.

`notebooks/02_tire_degradation.ipynb` builds the degradation model, and is really about one
statistical idea: within a stint, fuel load and tire age are **perfectly collinear**
(corr = 1.000), so their effects cannot be separated by any algorithm.

`notebooks/03_pit_strategy.ipynb` turns those rates into a decision. Its lesson is to look for the
closed form before reaching for simulation — one line of algebra gives the optimal stop count, a
derivable threshold, and a test oracle.

---

## Architecture

```
app/
  config.py            pydantic-settings
  db/
    base.py            engine, session factory, naming conventions
    models.py          SQLAlchemy ORM
  ingestion/
    fastf1_client.py   FastF1 wrapper (always telemetry=False)
    transform.py       DataFrame -> rows; all data gotchas handled here
    loader.py          idempotent upserts
    cli.py             typer commands
  api/routes/          FastAPI endpoints
  ml/
    fuel.py            physics-based fuel-load correction
    categories.py      pinned categorical vocabularies (unseen -> NaN)
    registry.py        XGBoost->JSON / sklearn->joblib + ml.model_versions
    tire/              phase 2: dataset, stage_a, train, predict
    strategy/          phase 3: inputs, enumerate, simulator, monte_carlo
    prediction/        phase 4: dataset, train, predict
```

### Three Postgres schemas

| schema | holds |
|---|---|
| `raw` | ingested FastF1 data, never mutated by modeling code |
| `features` | views: `stint_laps`, `pit_loss`, `race_conditions`, `race_features`; matviews: `mv_driver_season`, `mv_team_season`, `mv_circuit_season` |
| `ml` | model metadata, versions, metrics |

The separation means a modeling bug can never corrupt the ingested data, and re-deriving
features is always safe.

### Things the data will do to you

Six behaviours of FastF1 data that produce working code with wrong answers. All are handled in
`app/ingestion/transform.py` and demonstrated in notebook 01.

1. **`LapTime` is a `Timedelta` and often `NaT`.** Stored as nullable `BIGINT` milliseconds —
   exact, correctly sortable, and NULL stays distinct from zero.

2. **`TrackStatus` is a concatenated flag string, not a number.** `'14'` means the lap was partly
   clear *and* partly under safety car. `int('1267')` yields 1267, which is nonsense that raises
   no error. We parse it to a sorted `SMALLINT[]`, so `'12'` and `'21'` become the same canonical
   `{1,2}` rather than two distinct arrays that split a `GROUP BY`.

3. **`IsAccurate` is free quality control.** FastF1's own flag for a comparable lap; it excludes
   in-laps, out-laps and timing dropouts.

4. **`Compound` is relative, not absolute.** SOFT/MEDIUM/HARD describe that weekend's allocation
   of C1–C5, so a Monza soft and a Monaco soft are different rubber. Any model using compound
   needs a circuit interaction.

5. **`TyreLife` counts laps across sessions.** A "new" race tire that did a qualifying run is not
   at zero. `FreshTyre` tells you whether the stint truly began on a new set.

6. **Pit stop duration does not exist in the data.** It is derived from `PitInTime` (in-lap) and
   `PitOutTime` (the *next* lap). The naive difference is not stationary time, and sometimes not
   a pit stop at all — at Brazil 2024 a red flag fell with cars in the pit lane, making it 23
   minutes. Implausible values are stored as NULL rather than averaged into Phase 3's pit-loss
   estimate.

7. **`Compound` is sometimes the literal string `"None"`.** Not Python `None`. `pd.isna("None")`
   is `False`, so it slips through a naive `str()` and becomes a phantom compound category. The
   CHECK constraint caught it during a backfill; `to_str` now maps sentinel strings to NULL.

8. **`session.date` is naive but already UTC.** Passing a naive datetime to `TIMESTAMPTZ` makes
   Postgres read it as server-local time, shifting every session date by the local UTC offset —
   by a different amount on a different machine. `_as_utc()` localises explicitly, which matters
   because Phase 4 orders races by time.

9. **`is_accurate` and the green-flag filter together are still not enough.** Mexico 2023 lap 36
   records **148.5 s** against a ~83 s norm, with `track_status = {1}` and `is_accurate = true` —
   it is the standing-restart lap after a red flag, including time stationary on the grid. Only
   17 such laps exist in 46 races, but one inside a 34-lap stint wrecks a linear fit. **Phase 2
   must add a pace-based outlier filter**; `cli.py doctor` reports the count as a `[WARN]`.

### Querying green-flag laps

```sql
-- Strictly clear. This is the right default for pace modeling: a yellow-flag
-- lap means drivers lifted, so its time is not representative.
WHERE l.is_accurate AND l.track_status = ARRAY[1]::smallint[]

-- Looser: keeps yellow-flag laps, excludes SC / red / VSC.
WHERE l.is_accurate AND NOT (l.track_status && ARRAY[4,5,6,7]::smallint[])
```

---

## Phase 2 results

Trained on 2019–2023, held out 2024 (21,036 laps):

| model | holdout MAE | r² | vs baseline |
|---|---|---|---|
| Baseline ("this lap == its stint median") | 0.5276s | — | — |
| Stage B Ridge | 0.4913s | 0.129 | +6.9% |
| Stage C XGBoost (primary) | 0.4885s | 0.126 | +7.4% |
| Stage C HistGBR (control) | 0.4856s | 0.143 | +8.0% |

All three within 1.2% — once fuel-corrected, degradation is close to linear in tire age.
Regularisation mattered far more than model choice: 600 trees at depth 6 scored 0.5410s, *worse
than Ridge*; 120 trees at depth 3 scored 0.4846s.

**Known limitation that matters for Phase 3:** the pooled model predicts a median **68% of the
degradation rate Stage A measures**, and compresses the spread across circuits ~4×. The
regularisation that minimised per-lap MAE also flattened the curve. See
[docs/phase2-tire-degradation.md](docs/phase2-tire-degradation.md) — use Stage A's measured
`(circuit, compound)` table, or calibrate, before feeding a simulator that integrates it.

## Phase 3 results

Strategies ranked by an exhaustive search (701,552 candidates at Sakhir) scored by a closed form,
then re-run across thousands of Monte Carlo races with safety cars.

| circuit | laps | pit loss | mean wear | best | real-world |
|---|---|---|---|---|---|
| Monaco | 78 | 22.4s | 0.010 s/lap | **1-stop** | always a 1-stop |
| Monza | 53 | 25.9s | 0.052 s/lap | **1-stop** | Leclerc won 2024 on a 1-stop |
| Sakhir | 57 | 24.5s | 0.112 s/lap | **2-stop** | classically a 2-stop |

The whole trade-off is one line — `slope * L*(L-1)/2 + stops * pit_loss` — from which stint order
provably cannot matter, long stints are punished quadratically, and the 1-stop/2-stop threshold is
**derivable** (0.106 s/lap at 50 laps, 22s pit loss). Brute force agrees with the algebra exactly.

**Why Phase 2's finding mattered.** Switching `degradation_source` from the measured Stage A table
to the Phase 2 pooled model changes the Sakhir recommendation from **2-stop to 1-stop** — the
model's ~40% flatness is not a small error downstream, it is a different decision.

## Phase 4 results

Walk-forward backtest over 81 races (train on everything before race N, predict race N):

| | MAE (positions) | Spearman | top-3 | top-1 |
|---|---|---|---|---|
| baseline: grid order | 3.522 | +0.604 | 59.7% | 44.4% |
| **Ridge** | **3.136** | **+0.677** | **66.3%** | **56.8%** |
| Random forest | 3.218 | +0.676 | 60.1% | 54.3% |
| XGBoost | 3.226 | +0.671 | 61.3% | 54.3% |

**+11.0% over the grid baseline**, which is strong because overtaking is hard. Podium
probability is well calibrated (Brier 0.069 vs 0.128 base rate, **+45.6% skill**): predicted
0.70 → observed 0.71.

**Ridge wins** — the mirror image of Phase 2, where a booster won on 104k rows. With 2,416
rows the linear model is best. *The amount of data decides the model.*

The leakage guard is a window frame ending at `1 PRECEDING`, using `GROUPS` not `ROWS`.
With `ROWS`, a team-level feature reaches the driver's own teammate **in the race being
predicted** — see [docs/phase4-race-prediction.md](docs/phase4-race-prediction.md) for the
Ferrari table that demonstrates it.

## Phase 5 results

2024 dashboard, straight from the materialized views:

| driver | team | points | wins | podiums |
|---|---|---|---|---|
| VER | Red Bull Racing | 399 | 9 | 14 |
| NOR | McLaren | 344 | 4 | 13 |
| LEC | Ferrari | 327 | 3 | 13 |

Constructors: McLaren 609, Ferrari 595, Red Bull 537 — McLaren won that championship.
Points are **race-only** (sprints were never ingested), and every response says so.

`tire_mgmt_s_per_lap` measures a driver's degradation slope **relative to the field median
in the same race**, which cancels the fuel effect exactly without needing the fuel model.
Sainz and Alonso lead it, both genuinely known for tire management.

## Documentation

| Doc | Covers |
|---|---|
| [phase1-data-foundation.md](docs/phase1-data-foundation.md) | Phase 1 design decisions and data gotchas |
| [phase1-code-walkthrough.md](docs/phase1-code-walkthrough.md) | Phase 1 function by function, with line numbers |
| [phase2-tire-degradation.md](docs/phase2-tire-degradation.md) | Phase 2 findings, results, limitations |
| [phase2-code-walkthrough.md](docs/phase2-code-walkthrough.md) | Phase 2 function by function, with line numbers |
| [phase3-pit-strategy.md](docs/phase3-pit-strategy.md) | Phase 3 findings, results, limitations |
| [phase3-code-walkthrough.md](docs/phase3-code-walkthrough.md) | Phase 3 function by function, with line numbers |
| [phase4-race-prediction.md](docs/phase4-race-prediction.md) | Phase 4 findings, the leakage guards, results |
| [phase4-code-walkthrough.md](docs/phase4-code-walkthrough.md) | Phase 4 function by function, with line numbers |
| [phase5-dashboard.md](docs/phase5-dashboard.md) | Phase 5 findings, materialized views, caveats |
| [phase5-code-walkthrough.md](docs/phase5-code-walkthrough.md) | Phase 5 function by function, with line numbers |

## Data source

[FastF1](https://docs.fastf1.dev) 3.8.3. Full timing data is available from 2018 onward, and
appears 30–120 minutes after a session ends. FastF1 routes standings and results through
[Jolpica](https://jolpi.ca), the successor to the now-dead Ergast API — ignore older tutorials
that call Ergast directly.
