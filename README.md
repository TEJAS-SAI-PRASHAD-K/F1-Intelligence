# F1 Intelligence

A backend for Formula 1 analytics, built in phases as a way to learn data science and ML on
real data rather than a tutorial dataset.

**Four features, built easiest to hardest:**

| Phase | Feature | Status |
|---|---|---|
| 1 | Data foundation — ingest FastF1 timing data into Postgres | **done** |
| 2 | Tire degradation model | next |
| 3 | Pit stop strategy simulator | planned |
| 4 | Race prediction | planned |
| 5 | Dashboard API | planned |

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
```

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

`notebooks/01_explore_fastf1.ipynb` walks through the raw data and the six places it will
mislead you. Read it before trusting anything downstream.

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
  ml/                  phases 2-4
```

### Three Postgres schemas

| schema | holds |
|---|---|
| `raw` | ingested FastF1 data, never mutated by modeling code |
| `features` | derived tables and views built from `raw` |
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

## Data source

[FastF1](https://docs.fastf1.dev) 3.8.3. Full timing data is available from 2018 onward, and
appears 30–120 minutes after a session ends. FastF1 routes standings and results through
[Jolpica](https://jolpi.ca), the successor to the now-dead Ergast API — ignore older tutorials
that call Ergast directly.
