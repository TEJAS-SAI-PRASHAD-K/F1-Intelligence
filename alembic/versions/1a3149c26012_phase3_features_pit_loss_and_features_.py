"""phase3: features.pit_loss and features.race_conditions

Two views supplying the EMPIRICAL inputs the pit-strategy simulator needs. Both
contain facts measured from lap data and nothing else. Anything involving a
modeling assumption -- the fuel coefficient, base pace, viable stint lengths --
stays in Python, for the same reason Phase 2 kept fuel out of
features.stint_laps: an assumption belongs where it can be documented, tested
and varied.

HOW PIT LOSS IS DEFINED, AND WHY NOT THE OBVIOUS WAY
----------------------------------------------------
raw.pit_stops.duration_ms_est is the time from crossing the pit ENTRY line to
crossing the pit EXIT line. That is NOT the time a stop costs you, because it
omits the braking before the entry line and the acceleration after the exit
line. Measured against this data it understates pit loss by roughly 2-3s.

The quantity a strategist actually needs is "how much time did pitting cost me
versus staying out", so this view uses:

    pit_loss = (in_lap + out_lap) - 2 x the driver's normal green-flag pace

That captures deceleration, pit lane transit, the stationary stop and
acceleration back to speed, all at once.

Validation: this gives a median of 23.3s across 2,715 green-flag stops, inside
the 18-25s range quoted for modern F1. The entry-to-exit figure gives ~20-22s.

AND THE FILTER LESSON, FOR THE THIRD TIME
-----------------------------------------
Phase 1: is_accurate plus a green-flag filter still admitted a 148s restart lap.
Phase 2: dry compounds still admitted 20s laps from a drying track.
Phase 3: including stops made under a safety car gives a p90 pit loss of 71.9s.

A stop under the safety car has a slow in-lap by definition, so the subtraction
attributes the safety car's cost to the pit stop. Restricting to stops whose
in-lap AND out-lap were both strictly green brings p90 from 71.9s down to 32.0s.
The median only moves from 24.6s to 23.3s -- which is exactly why the median was
chosen, and exactly why looking only at the median would have hidden the problem.

Revision ID: 1a3149c26012
Revises: 819c4ea913e1
"""

from collections.abc import Sequence

from alembic import op

revision: str = "1a3149c26012"
down_revision: str | None = "819c4ea913e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# A driver needs this many clean green laps in a race before their median is a
# trustworthy "normal pace" reference to subtract.
MIN_GREEN_LAPS_FOR_PACE = 10

PIT_LOSS_VIEW = f"""
CREATE OR REPLACE VIEW features.pit_loss AS
WITH normal_pace AS (
    -- Each driver's own median green-flag pace in that race, EXCLUDING in-laps
    -- and out-laps (which are the thing being measured).
    SELECT l.session_id,
           l.driver_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY l.lap_time_ms) AS normal_ms
    FROM raw.laps l
    JOIN raw.sessions s ON s.id = l.session_id AND s.session_type = 'R'
    WHERE l.is_accurate
      AND l.track_status = ARRAY[1]::smallint[]
      AND l.lap_time_ms IS NOT NULL
      AND l.pit_in_ms  IS NULL
      AND l.pit_out_ms IS NULL
    GROUP BY l.session_id, l.driver_id
    HAVING count(*) >= {MIN_GREEN_LAPS_FOR_PACE}
),
stops AS (
    SELECT c.name AS circuit,
           e.season_year,
           il.session_id,
           il.driver_id,
           il.lap_number,
           (il.lap_time_ms + ol.lap_time_ms - 2 * np.normal_ms) AS pit_loss_ms
    FROM raw.laps il
    -- The out-lap is the NEXT lap by the same driver, identified by pit_out_ms.
    JOIN raw.laps ol
      ON  ol.session_id = il.session_id
      AND ol.driver_id  = il.driver_id
      AND ol.lap_number = il.lap_number + 1
      AND ol.pit_out_ms IS NOT NULL
    JOIN normal_pace np
      ON  np.session_id = il.session_id
      AND np.driver_id  = il.driver_id
    JOIN raw.sessions s ON s.id = il.session_id
    JOIN raw.events   e ON e.id = s.event_id
    JOIN raw.circuits c ON c.id = e.circuit_id
    WHERE il.pit_in_ms    IS NOT NULL
      AND il.lap_time_ms  IS NOT NULL
      AND ol.lap_time_ms  IS NOT NULL
      -- BOTH laps strictly green. See the module docstring: safety-car stops
      -- push p90 pit loss from 32s to 72s.
      AND il.track_status = ARRAY[1]::smallint[]
      AND ol.track_status = ARRAY[1]::smallint[]
)
SELECT circuit,
       count(*)                                                      AS n_stops,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY pit_loss_ms)       AS pit_loss_ms,
       percentile_cont(0.1) WITHIN GROUP (ORDER BY pit_loss_ms)       AS pit_loss_p10_ms,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY pit_loss_ms)       AS pit_loss_p90_ms,
       stddev_samp(pit_loss_ms)                                       AS pit_loss_sd_ms
FROM stops
GROUP BY circuit
"""

RACE_CONDITIONS_VIEW = """
CREATE OR REPLACE VIEW features.race_conditions AS
WITH per_race AS (
    SELECT s.id              AS session_id,
           c.name            AS circuit,
           e.season_year,
           s.total_laps,
           -- Distinct LAPS on which any car saw SC or VSC. Counting laps rather
           -- than rows avoids 20 cars inflating a single safety car period.
           count(DISTINCT l.lap_number)
               FILTER (WHERE l.track_status && ARRAY[4, 6, 7]::smallint[]) AS sc_laps,
           bool_or(l.track_status && ARRAY[5]::smallint[])                 AS had_red_flag
    FROM raw.sessions s
    JOIN raw.events   e ON e.id = s.event_id
    JOIN raw.circuits c ON c.id = e.circuit_id
    JOIN raw.laps     l ON l.session_id = s.id
    WHERE s.session_type = 'R'
      AND s.total_laps IS NOT NULL
    GROUP BY s.id, c.name, e.season_year, s.total_laps
)
SELECT circuit,
       count(*)                                           AS n_races,
       -- Mode would be better but percentile is simpler and agrees here:
       -- race distance at a circuit is fixed by regulation.
       percentile_cont(0.5) WITHIN GROUP (ORDER BY total_laps)::int AS race_laps,
       avg((sc_laps > 0)::int)                            AS sc_probability,
       avg(sc_laps)                                       AS mean_sc_laps,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY sc_laps)
           FILTER (WHERE sc_laps > 0)                     AS median_sc_laps_when_deployed,
       avg(had_red_flag::int)                             AS red_flag_probability
FROM per_race
GROUP BY circuit
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS features")
    op.execute(PIT_LOSS_VIEW)
    op.execute(RACE_CONDITIONS_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS features.race_conditions")
    op.execute("DROP VIEW IF EXISTS features.pit_loss")
