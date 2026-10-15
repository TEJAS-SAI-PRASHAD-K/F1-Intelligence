# Phase 5 — Dashboard API

## What this phase is for

Aggregation endpoints a frontend can consume, plus a window onto the model registry.
No new modelling — this phase is about serving what Phases 1–4 produced, efficiently
and honestly.

**The lesson of this phase:** when the database can do the caching, let it. And when
a number is not what it looks like, say so in the response.

## What was built

| Component | File | Does |
|---|---|---|
| Materialized views | `alembic/.../bf59eb622519_*.py` | 3 matviews + unique indexes |
| Refresh command | `app/ingestion/cli.py` | `refresh-views`, concurrent by default |
| API | `app/api/routes/dashboard.py` | 4 endpoints |
| Schemas | `app/schemas/dashboard.py` | Response contracts |
| Tests | `tests/test_dashboard_api.py` | 20 tests |

---

## Materialized views, not an application cache

The dashboard aggregates over 171k laps and 2,400 race results. Computing that per
request would be slow, so the aggregates are **materialized views** refreshed after
each ingest.

A Python cache would have to be invalidated, kept consistent across worker processes,
and warmed after a restart. Postgres already holds the data and the query planner, so
an application cache would be solving a problem the database solves better — and
solving it worse, because a per-process cache goes stale differently in each worker.

| view | rows | holds |
|---|---|---|
| `features.mv_driver_season` | 131 | standings, form, DNFs, tire management |
| `features.mv_team_season` | 60 | constructor table |
| `features.mv_circuit_season` | 113 | pace, pit stops, safety-car laps |

### `CONCURRENTLY` needs a unique index

`REFRESH MATERIALIZED VIEW CONCURRENTLY` keeps the old contents readable while the new
ones build, so the dashboard never serves an empty table mid-refresh.

It **requires** a unique index on the view. Without one Postgres fails at refresh
time, not at creation — an easy thing to miss until the first refresh in production.
The migration creates one per view immediately after it.

```bash
.venv/bin/python -m app.ingestion.cli refresh-views
```

```
[ OK ] features.mv_driver_season             131 rows  (0.6s)
[ OK ] features.mv_team_season                60 rows  (0.0s)
[ OK ] features.mv_circuit_season            113 rows  (5.5s)
```

Each view refreshes in its own transaction, so one failure does not roll back the
views that already succeeded. Plain views (`stint_laps`, `pit_loss`,
`race_conditions`, `race_features`) need nothing — they are queries and always
current. Only the materialized ones store a snapshot and go stale.

---

## The tire-management trick

`mv_driver_season.tire_mgmt_s_per_lap` measures how well a driver preserves tires, and
it had a problem.

A raw lap-time-versus-tire-age slope is contaminated by fuel burn — Phase 2's entire
subject — and the fuel correction is a Python assumption that deliberately does not
live in SQL. Replicating 110 kg × 0.03 s/kg inside a view would duplicate the
assumption in a second place, where it could drift.

**The fix:** compare each driver's slope to the **median slope of the field in the
same race**. Every car in a race burns fuel at essentially the same rate, so the fuel
effect is common to all of them, and subtracting the field median cancels it exactly —
no assumption required.

```sql
avg(driver_slope - field_slope) / 1000.0 AS tire_mgmt_s_per_lap
```

Negative means "degrades more slowly than the field that day", which is a better
definition of tire management than an absolute slope anyway, because it controls for
circuit and conditions at the same time.

2024, drivers with ≥10 measured races:

| driver | tire_mgmt_s_per_lap |
|---|---|
| MAG | −0.0231 |
| SAI | −0.0144 |
| RIC | −0.0136 |
| ALO | −0.0090 |
| BOT | −0.0066 |

Sainz and Alonso are both genuinely known for tire management, which is a reasonable
sanity check on a derived metric.

---

## Results agree with reality

2024 drivers:

| driver | team | points | wins | podiums | mean finish |
|---|---|---|---|---|---|
| VER | Red Bull Racing | 399 | 9 | 14 | 3.62 |
| NOR | McLaren | 344 | 4 | 13 | 4.29 |
| LEC | Ferrari | 327 | 3 | 13 | 4.54 |

2024 constructors: **McLaren 609, Ferrari 595, Red Bull 537** — McLaren won that
championship.

### Points are race-only, and the response says so

Win counts are exactly right. **Points are lower than official championship points**,
because sprint sessions were never ingested (`--only R,Q`). Official 2024 was
Verstappen 437, McLaren 666.

A number that looks like championship points but is not has to say so, so
`SEASON_NOTES` is returned on every season response and
`DriverStandingOut.points` carries the caveat in its field description. The
alternative — a number that is quietly wrong by 8% — is worse than no number.

---

## One place the dashboard reuses rather than reimplements

The race dashboard shows per-stint degradation. The first version computed
`regr_slope(lap_time_ms, tyre_life)` in SQL, which is **the exact mistake Phase 2
exists to prevent**: uncorrected slopes come out near zero or negative, because the
fuel effect (~0.055 s/lap of gain) nearly cancels tire wear.

Worse, the column was aliased `lap_time_fuel_corr` while holding raw lap times — a
name that asserted something false.

It now calls Phase 2's `stage_a` path, which fuel-corrects first. That keeps **one
definition of "degradation"** in the project instead of a second, wrong one. Verified
by `test_stint_degradation_is_fuel_corrected`, which asserts most measurable stints
show *positive* degradation.

### And a 40× speedup found along the way

`observed_stint_fits` called `ds.build_dataset(engine)` and then filtered to one
session — loading ~104,000 rows and running the whole fuel-correction pipeline to keep
about 1,000 of them. The race dashboard took **12 seconds per request**.

Pushing the filter into SQL made it **0.29s**, a 40× improvement, with an identical
answer. The combinatorics were never the problem; the data volume was.

---

## A testing decision worth explaining

Every other suite uses `conftest.client`, whose `get_db` override hands routes a
session bound to `f1_intelligence_test`. That database holds only ORM tables —
`Base.metadata.create_all` does not create views — so `features.mv_driver_season` does
not exist there, and would be empty if it did.

The dashboard's entire job is aggregating 171k real laps. Testing it against an empty
schema would exercise nothing. So `test_dashboard_api.py` defines its own
`dashboard_client` fixture that overrides `get_db` with a **real-database** session.
Every dashboard endpoint is read-only, so this cannot leave anything behind.

This was found the direct way: the first version of the suite asserted that
`features.*` queries "resolve against the real database", which was simply wrong —
they resolve against whatever connection the session holds. 16 tests failed with
`UndefinedTable`.

---

## Known limitations

1. **Points exclude sprints.** Documented in every response. Fixing it means ingesting
   `S` and `SQ` sessions and adding their points.
2. **`prediction_vs_actual` on the race dashboard is in-sample.** The registered model
   saw that race in training, so `prediction_note` says so. It is for inspecting
   reasoning, not measuring accuracy.
3. **Views go stale after an ingest** until `refresh-views` runs. The season endpoint's
   404 message names the command, so the failure is self-explaining.
4. **`mv_circuit_season` takes 5.5s to refresh**, the slowest of the three, because it
   scans `features.stint_laps`. Fine for a post-ingest step; it would not be fine
   per request, which is why it is materialized.
5. **Tire management needs ≥8 clean laps per stint** to fit a slope, so drivers who
   rarely complete long green-flag runs have a null or noisy figure.
   `tire_mgmt_races` reports the sample size so a reader can discount it.

---

## Verification

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -m app.ingestion.cli refresh-views
.venv/bin/python -m pytest tests/test_dashboard_api.py -q
.venv/bin/uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/api/v1/dashboard/season/2024 | python -m json.tool
curl -s localhost:8000/api/v1/dashboard/driver/VER  | python -m json.tool
curl -s localhost:8000/api/v1/dashboard/models      | python -m json.tool
```

Check: the 2024 points leader is Verstappen with 9 wins, McLaren tops the constructor
table, the sprint-points caveat appears in `notes`, and tire-management values straddle
zero (they must, being measured relative to the field median).
