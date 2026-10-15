"""phase5: dashboard materialized views

The dashboard serves aggregates over 171k laps and 2,400 race results. Computing
those per request would be wasteful and slow, so they are MATERIALIZED views
refreshed after each ingest.

WHY MATERIALIZED VIEWS RATHER THAN AN APPLICATION CACHE
-------------------------------------------------------
A Python cache would have to be invalidated, kept consistent across processes,
and warmed after a restart. Postgres already has the data and the query planner,
so a cache layer in the application would be solving a problem the database
solves better -- and solving it worse, because a per-process cache goes stale
differently in each worker.

`REFRESH MATERIALIZED VIEW CONCURRENTLY` keeps the old contents readable while
the new ones build, so the dashboard never serves an empty table mid-refresh.
It REQUIRES a unique index on the view, which is why each one below has a
`UNIQUE INDEX` created immediately after it. Without that index the CONCURRENTLY
form fails at runtime, not at creation -- an easy thing to miss until the first
refresh in production.

THE TIRE-MANAGEMENT TRICK
------------------------
`driver_tire_management` measures how well a driver preserves tires, and it has
a problem: a raw lap-time-versus-tire-age slope is contaminated by fuel burn
(Phase 2's whole subject), and the fuel correction is a Python assumption that
deliberately does not live in SQL.

The fix is to compare each driver's slope to the MEDIAN SLOPE OF THE FIELD IN
THE SAME RACE. Every car in a race burns fuel at essentially the same rate, so
the fuel effect is common to all of them and subtracting the field median
cancels it exactly -- no assumption required.

So a negative value means "degrades more slowly than the field that day", which
is a cleaner definition of tire management than an absolute slope anyway, since
it controls for circuit and conditions at the same time.

Revision ID: bf59eb622519
Revises: 195b31c0a04c
"""

from collections.abc import Sequence

from alembic import op

revision: str = "bf59eb622519"
down_revision: str | None = "195b31c0a04c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# A stint needs this many clean laps before regr_slope means anything.
MIN_LAPS_FOR_SLOPE = 8

DRIVER_SEASON = f"""
CREATE MATERIALIZED VIEW features.mv_driver_season AS
WITH race_rows AS (
    SELECT e.season_year,
           d.id            AS driver_id,
           d.abbreviation  AS driver,
           d.full_name,
           t.name          AS team,
           sr.position::float  AS finish_position,
           sr.grid_position::float,
           sr.points::float,
           (sr.status = 'Finished' OR sr.status LIKE '%Lap%') AS classified
    FROM raw.session_results sr
    JOIN raw.sessions s ON s.id = sr.session_id AND s.session_type = 'R'
    JOIN raw.events   e ON e.id = s.event_id
    JOIN raw.drivers  d ON d.id = sr.driver_id
    LEFT JOIN raw.teams t ON t.id = sr.team_id
    WHERE sr.position IS NOT NULL
),
-- Per-stint raw slope, then each driver's median, then the FIELD median for the
-- same race. Subtracting the two cancels the fuel effect (see the docstring).
stint_slopes AS (
    SELECT sl.session_id,
           sl.driver_id,
           regr_slope(sl.lap_time_ms, sl.tyre_life) AS slope_ms_per_lap
    FROM features.stint_laps sl
    GROUP BY sl.session_id, sl.driver_id, sl.stint_number
    HAVING count(*) >= {MIN_LAPS_FOR_SLOPE}
),
driver_race_slope AS (
    SELECT session_id, driver_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY slope_ms_per_lap) AS driver_slope
    FROM stint_slopes GROUP BY session_id, driver_id
),
field_race_slope AS (
    SELECT session_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY slope_ms_per_lap) AS field_slope
    FROM stint_slopes GROUP BY session_id
),
tire_management AS (
    SELECT e.season_year, drs.driver_id,
           avg(drs.driver_slope - frs.field_slope) / 1000.0 AS tire_mgmt_s_per_lap,
           count(*) AS races_measured
    FROM driver_race_slope drs
    JOIN field_race_slope frs ON frs.session_id = drs.session_id
    JOIN raw.sessions s ON s.id = drs.session_id
    JOIN raw.events   e ON e.id = s.event_id
    GROUP BY e.season_year, drs.driver_id
)
SELECT r.season_year,
       r.driver,
       max(r.full_name)                                   AS full_name,
       mode() WITHIN GROUP (ORDER BY r.team)              AS team,
       count(*)                                           AS races,
       sum(r.points)                                      AS points,
       count(*) FILTER (WHERE r.finish_position = 1)       AS wins,
       count(*) FILTER (WHERE r.finish_position <= 3)      AS podiums,
       count(*) FILTER (WHERE r.finish_position <= 10)     AS points_finishes,
       count(*) FILTER (WHERE NOT r.classified)            AS dnfs,
       avg(r.finish_position)                              AS mean_finish,
       avg(r.grid_position)                                AS mean_grid,
       -- Positive means the driver gains places relative to where they start.
       avg(r.grid_position - r.finish_position)            AS mean_places_gained,
       max(tm.tire_mgmt_s_per_lap)                         AS tire_mgmt_s_per_lap,
       max(tm.races_measured)                              AS tire_mgmt_races
FROM race_rows r
LEFT JOIN tire_management tm
       ON tm.season_year = r.season_year AND tm.driver_id = r.driver_id
GROUP BY r.season_year, r.driver
"""

TEAM_SEASON = """
CREATE MATERIALIZED VIEW features.mv_team_season AS
SELECT e.season_year,
       t.name                                          AS team,
       count(*)                                        AS entries,
       count(DISTINCT sr.driver_id)                    AS drivers,
       sum(sr.points::float)                           AS points,
       count(*) FILTER (WHERE sr.position = 1)          AS wins,
       count(*) FILTER (WHERE sr.position <= 3)         AS podiums,
       avg(sr.position::float)                         AS mean_finish,
       avg(sr.grid_position::float)                    AS mean_grid,
       count(*) FILTER (WHERE NOT (sr.status = 'Finished' OR sr.status LIKE '%Lap%'))
                                                       AS dnfs
FROM raw.session_results sr
JOIN raw.sessions s ON s.id = sr.session_id AND s.session_type = 'R'
JOIN raw.events   e ON e.id = s.event_id
JOIN raw.teams    t ON t.id = sr.team_id
WHERE sr.position IS NOT NULL
GROUP BY e.season_year, t.name
"""

CIRCUIT_SEASON = """
CREATE MATERIALIZED VIEW features.mv_circuit_season AS
WITH pace AS (
    SELECT sl.season_year, sl.circuit,
           count(*)                                                     AS clean_laps,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY sl.lap_time_ms)  AS median_lap_ms,
           min(sl.lap_time_ms)                                          AS fastest_lap_ms,
           avg(sl.air_temp)                                             AS air_temp,
           avg(sl.track_temp)                                           AS track_temp
    FROM features.stint_laps sl
    GROUP BY sl.season_year, sl.circuit
),
stops AS (
    SELECT e.season_year, c.name AS circuit,
           count(*)                                        AS pit_stops,
           avg(p.duration_ms_est)                          AS mean_stationary_ms
    FROM raw.pit_stops p
    JOIN raw.sessions s ON s.id = p.session_id AND s.session_type = 'R'
    JOIN raw.events   e ON e.id = s.event_id
    JOIN raw.circuits c ON c.id = e.circuit_id
    GROUP BY e.season_year, c.name
),
safety AS (
    SELECT e.season_year, c.name AS circuit,
           count(DISTINCT l.lap_number)
               FILTER (WHERE l.track_status && ARRAY[4,6,7]::smallint[]) AS sc_laps
    FROM raw.laps l
    JOIN raw.sessions s ON s.id = l.session_id AND s.session_type = 'R'
    JOIN raw.events   e ON e.id = s.event_id
    JOIN raw.circuits c ON c.id = e.circuit_id
    GROUP BY e.season_year, c.name
)
SELECT p.season_year, p.circuit, p.clean_laps, p.median_lap_ms, p.fastest_lap_ms,
       p.air_temp, p.track_temp,
       s.pit_stops, s.mean_stationary_ms,
       sa.sc_laps
FROM pace p
LEFT JOIN stops  s  ON s.season_year  = p.season_year AND s.circuit  = p.circuit
LEFT JOIN safety sa ON sa.season_year = p.season_year AND sa.circuit = p.circuit
"""

# CONCURRENTLY refresh REQUIRES a unique index on each materialized view.
INDEXES = [
    "CREATE UNIQUE INDEX uq_mv_driver_season ON features.mv_driver_season (season_year, driver)",
    "CREATE UNIQUE INDEX uq_mv_team_season ON features.mv_team_season (season_year, team)",
    "CREATE UNIQUE INDEX uq_mv_circuit_season "
    "ON features.mv_circuit_season (season_year, circuit)",
]

MATVIEWS = ("mv_driver_season", "mv_team_season", "mv_circuit_season")


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS features")
    op.execute(DRIVER_SEASON)
    op.execute(TEAM_SEASON)
    op.execute(CIRCUIT_SEASON)
    for statement in INDEXES:
        op.execute(statement)


def downgrade() -> None:
    for name in MATVIEWS:
        op.execute(f"DROP MATERIALIZED VIEW IF EXISTS features.{name}")
