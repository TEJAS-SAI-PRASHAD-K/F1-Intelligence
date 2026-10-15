"""Phase 5 endpoints: dashboard aggregates.

Every aggregate here reads a MATERIALIZED view refreshed after ingest, rather
than computing over 171k laps per request. See the Phase 5 migration for why
that is preferred over an application-level cache.

The one exception is `/race/{event_id}`, which reads live: it touches a single
race, so the query is already cheap and materialising it would mean a row per
race for no gain.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, text
from sqlalchemy.orm import Session as OrmSession

from app.db.base import engine
from app.db.deps import get_db
from app.db.models import Driver, Event, ModelVersion
from app.schemas.dashboard import (
    CircuitSeasonOut,
    CircuitStrengthOut,
    DriverDashboardOut,
    DriverSeasonSummaryOut,
    DriverStandingOut,
    ModelsDashboardOut,
    ModelSummaryOut,
    PredictionVsActualOut,
    RaceDashboardOut,
    RaceResultOut,
    SeasonDashboardOut,
    StintChartEntryOut,
    TeamStandingOut,
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

SEASON_NOTES = [
    "Points are RACE points only. Sprint sessions were not ingested, so totals "
    "are lower than official championship points in sprint years.",
    "tire_mgmt_s_per_lap is measured RELATIVE to the field median in the same "
    "race, which cancels the fuel effect without needing the fuel model.",
]

MODEL_PHASES = {
    "tire_degradation": "Phase 2 - tire degradation",
    "race_prediction": "Phase 4 - race prediction",
    "race_prediction_podium": "Phase 4 - podium probability",
}


def _f(value) -> float | None:
    return None if value is None else float(value)


@router.get("/season/{year}", response_model=SeasonDashboardOut)
def season_dashboard(year: int, db: OrmSession = Depends(get_db)):
    """Standings, constructor table and per-circuit pace for one season."""
    drivers = db.execute(
        text(
            "SELECT * FROM features.mv_driver_season WHERE season_year = :y "
            "ORDER BY points DESC, wins DESC"
        ),
        {"y": year},
    ).mappings().all()

    if not drivers:
        available = db.execute(
            text("SELECT DISTINCT season_year FROM features.mv_driver_season ORDER BY 1")
        ).scalars().all()
        raise HTTPException(
            404,
            f"No data for season {year}. Available: {', '.join(map(str, available))}. "
            "If a season was just ingested, refresh the views: "
            "python -m app.ingestion.cli refresh-views",
        )

    teams = db.execute(
        text(
            "SELECT * FROM features.mv_team_season WHERE season_year = :y "
            "ORDER BY points DESC"
        ),
        {"y": year},
    ).mappings().all()

    circuits = db.execute(
        text(
            "SELECT * FROM features.mv_circuit_season WHERE season_year = :y ORDER BY circuit"
        ),
        {"y": year},
    ).mappings().all()

    races = db.scalar(
        text("SELECT count(*) FROM raw.events e JOIN raw.sessions s ON s.event_id = e.id "
             "AND s.session_type = 'R' WHERE e.season_year = :y"),
        {"y": year},
    )

    return SeasonDashboardOut(
        season_year=year,
        races=int(races or 0),
        drivers=[
            DriverStandingOut(
                driver=r["driver"], full_name=r["full_name"], team=r["team"],
                races=r["races"], points=_f(r["points"]) or 0.0, wins=r["wins"],
                podiums=r["podiums"], points_finishes=r["points_finishes"], dnfs=r["dnfs"],
                mean_finish=round(_f(r["mean_finish"]) or 0, 3),
                mean_grid=round(_f(r["mean_grid"]) or 0, 3),
                mean_places_gained=round(_f(r["mean_places_gained"]) or 0, 3),
                tire_mgmt_s_per_lap=(
                    round(_f(r["tire_mgmt_s_per_lap"]), 5)
                    if r["tire_mgmt_s_per_lap"] is not None else None
                ),
                tire_mgmt_races=r["tire_mgmt_races"],
            )
            for r in drivers
        ],
        teams=[
            TeamStandingOut(
                team=r["team"], entries=r["entries"], drivers=r["drivers"],
                points=_f(r["points"]) or 0.0, wins=r["wins"], podiums=r["podiums"],
                mean_finish=round(_f(r["mean_finish"]) or 0, 3),
                mean_grid=round(_f(r["mean_grid"]) or 0, 3), dnfs=r["dnfs"],
            )
            for r in teams
        ],
        circuits=[
            CircuitSeasonOut(
                circuit=r["circuit"], clean_laps=r["clean_laps"],
                median_lap_s=round((_f(r["median_lap_ms"]) or 0) / 1000, 3),
                fastest_lap_s=round((_f(r["fastest_lap_ms"]) or 0) / 1000, 3),
                air_temp=(round(_f(r["air_temp"]), 1) if r["air_temp"] is not None else None),
                track_temp=(
                    round(_f(r["track_temp"]), 1) if r["track_temp"] is not None else None
                ),
                pit_stops=r["pit_stops"],
                mean_stationary_s=(
                    round((_f(r["mean_stationary_ms"]) or 0) / 1000, 2)
                    if r["mean_stationary_ms"] is not None else None
                ),
                sc_laps=r["sc_laps"],
            )
            for r in circuits
        ],
        notes=SEASON_NOTES,
    )


@router.get("/driver/{abbreviation}", response_model=DriverDashboardOut)
def driver_dashboard(abbreviation: str, db: OrmSession = Depends(get_db)):
    """A driver's season-by-season record plus their best and worst circuits."""
    code = abbreviation.upper()
    driver = db.execute(select(Driver).where(Driver.abbreviation == code)).scalar_one_or_none()
    if driver is None:
        raise HTTPException(404, f"No driver with abbreviation {code!r}")

    seasons = db.execute(
        text(
            "SELECT * FROM features.mv_driver_season WHERE driver = :d ORDER BY season_year"
        ),
        {"d": code},
    ).mappings().all()
    if not seasons:
        raise HTTPException(404, f"No race data for {code}")

    circuits = db.execute(
        text(
            """
            SELECT circuit,
                   count(*)                     AS races,
                   avg(finish_position)         AS mean_finish,
                   avg(grid_position)           AS mean_grid,
                   min(finish_position)         AS best_finish
            FROM features.race_features
            WHERE driver = :d
            GROUP BY circuit
            HAVING count(*) >= 2
            ORDER BY mean_finish
            """
        ),
        {"d": code},
    ).mappings().all()

    def strength(rows) -> list[CircuitStrengthOut]:
        return [
            CircuitStrengthOut(
                circuit=r["circuit"], races=r["races"],
                mean_finish=round(_f(r["mean_finish"]) or 0, 2),
                mean_grid=round(_f(r["mean_grid"]) or 0, 2),
                best_finish=_f(r["best_finish"]) or 0,
            )
            for r in rows
        ]

    return DriverDashboardOut(
        driver=code,
        full_name=driver.full_name,
        country_code=driver.country_code,
        seasons=[
            DriverSeasonSummaryOut(
                season_year=r["season_year"], team=r["team"], races=r["races"],
                points=_f(r["points"]) or 0.0, wins=r["wins"], podiums=r["podiums"],
                dnfs=r["dnfs"], mean_finish=round(_f(r["mean_finish"]) or 0, 3),
                mean_places_gained=round(_f(r["mean_places_gained"]) or 0, 3),
                tire_mgmt_s_per_lap=(
                    round(_f(r["tire_mgmt_s_per_lap"]), 5)
                    if r["tire_mgmt_s_per_lap"] is not None else None
                ),
            )
            for r in seasons
        ],
        best_circuits=strength(circuits[:5]),
        worst_circuits=strength(list(reversed(circuits))[:5]),
        career_races=sum(r["races"] for r in seasons),
        career_wins=sum(r["wins"] for r in seasons),
        career_podiums=sum(r["podiums"] for r in seasons),
    )


@router.get("/race/{event_id}", response_model=RaceDashboardOut)
def race_dashboard(event_id: int, db: OrmSession = Depends(get_db)):
    """Everything about one race: result, stint chart, and prediction vs actual.

    Read live rather than materialised: one race is a cheap query, and a
    materialised view would store a row per race for no benefit.

    `prediction_vs_actual` is IN-SAMPLE -- the registered model saw this race in
    training -- so `prediction_note` says so. It is for inspecting reasoning, not
    measuring accuracy.
    """
    event = db.get(Event, event_id)
    if event is None:
        raise HTTPException(404, f"No event with id {event_id}")

    race = db.execute(
        text(
            "SELECT id, total_laps FROM raw.sessions "
            "WHERE event_id = :e AND session_type = 'R'"
        ),
        {"e": event_id},
    ).mappings().first()
    if race is None:
        raise HTTPException(404, f"Event {event_id} has no race session ingested")

    results = db.execute(
        text(
            """
            SELECT sr.position, d.abbreviation AS driver, t.name AS team,
                   sr.grid_position, sr.points, sr.status
            FROM raw.session_results sr
            JOIN raw.drivers d ON d.id = sr.driver_id
            LEFT JOIN raw.teams t ON t.id = sr.team_id
            WHERE sr.session_id = :s
            ORDER BY sr.position NULLS LAST
            """
        ),
        {"s": race["id"]},
    ).mappings().all()

    # Stint shape from SQL, but the degradation SLOPE from Phase 2's Stage A,
    # which fuel-corrects first. A raw lap-time-vs-tire-age slope is the exact
    # mistake Phase 2 exists to prevent: the fuel effect (~0.055 s/lap of gain)
    # nearly cancels degradation, so uncorrected slopes come out near zero or
    # negative. Reusing stage_a keeps one definition of "degradation" in the
    # project instead of a second, wrong one here.
    stints = db.execute(
        text(
            """
            SELECT sl.driver, sl.stint_number, sl.compound,
                   min(sl.lap_number) AS lap_start, max(sl.lap_number) AS lap_end,
                   count(*)           AS laps,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY sl.lap_time_ms) AS median_ms
            FROM features.stint_laps sl
            WHERE sl.session_id = :s
            GROUP BY sl.driver, sl.stint_number, sl.compound
            ORDER BY sl.driver, sl.stint_number
            """
        ),
        {"s": race["id"]},
    ).mappings().all()

    slopes: dict[tuple[str, int], float] = {}
    try:
        from app.ml.tire import predict as tire_predict

        fits = tire_predict.observed_stint_fits(engine, int(race["id"]))
        if not fits.empty:
            slopes = {
                (row.driver, int(row.stint_number)): float(row.slope)
                for row in fits.itertuples(index=False)
            }
    except Exception:  # noqa: BLE001
        slopes = {}

    prediction: list[PredictionVsActualOut] = []
    note: str | None = None
    try:
        from app.ml.prediction import predict as race_predict

        predicted = race_predict.predict_event(engine, event_id, db=db)
        prediction = [
            PredictionVsActualOut(
                driver=d.driver, predicted_rank=d.predicted_rank,
                actual_position=d.actual_position, grid_position=d.grid_position,
                podium_probability=d.podium_probability,
                rank_error=(
                    int(abs(d.predicted_rank - d.actual_position))
                    if d.actual_position is not None else None
                ),
            )
            for d in predicted.drivers
        ]
        note = predicted.warnings[0] if predicted.warnings else None
    except Exception:  # noqa: BLE001
        note = (
            "No race-prediction model registered. Train one with: "
            "python -m app.ml.prediction.train"
        )

    return RaceDashboardOut(
        event_id=event_id,
        season_year=event.season_year,
        round=event.round,
        event_name=event.event_name,
        circuit=event.circuit.name if event.circuit else "Unknown",
        event_date=event.event_date,
        total_laps=race["total_laps"],
        results=[
            RaceResultOut(
                position=_f(r["position"]), driver=r["driver"], team=r["team"],
                grid_position=_f(r["grid_position"]), points=_f(r["points"]),
                status=r["status"],
                places_gained=(
                    _f(r["grid_position"]) - _f(r["position"])
                    if r["position"] is not None and r["grid_position"] is not None else None
                ),
            )
            for r in results
        ],
        stints=[
            StintChartEntryOut(
                driver=r["driver"], stint_number=r["stint_number"], compound=r["compound"],
                lap_start=r["lap_start"], lap_end=r["lap_end"], laps=r["laps"],
                median_green_lap_s=round((_f(r["median_ms"]) or 0) / 1000, 3),
                degradation_s_per_lap=(
                    round(slopes[(r["driver"], r["stint_number"])], 5)
                    if (r["driver"], r["stint_number"]) in slopes else None
                ),
            )
            for r in stints
        ],
        prediction_vs_actual=prediction,
        prediction_note=note,
    )


@router.get("/models", response_model=ModelsDashboardOut)
def models_dashboard(db: OrmSession = Depends(get_db)):
    """Every registered model, its provenance and its headline metric.

    The point of a registry is being able to answer "what is deployed, trained on
    what, and how good was it" without reading code. Each model's headline metric
    is rendered in its own units, because an MAE in milliseconds of lap time and
    an MAE in finishing positions are not comparable numbers.
    """
    rows = db.execute(select(ModelVersion).order_by(ModelVersion.trained_at.desc())).scalars().all()

    out: list[ModelSummaryOut] = []
    for row in rows:
        headline = None
        if row.metrics_json:
            try:
                metrics = json.loads(row.metrics_json)
                if "mae_positions" in metrics:
                    headline = f"{metrics['mae_positions']:.3f} positions MAE (walk-forward)"
                elif "brier" in metrics:
                    headline = f"Brier {metrics['brier']:.4f}"
                elif isinstance(metrics.get("test"), dict):
                    headline = f"{metrics['test']['mae_ms'] / 1000:.4f}s MAE (holdout)"
            except (ValueError, KeyError, TypeError):
                headline = None

        out.append(
            ModelSummaryOut(
                name=row.name, version=row.version, backend=row.backend,
                is_active=row.is_active, trained_at=row.trained_at,
                train_season_min=row.train_season_min, train_season_max=row.train_season_max,
                n_train_rows=row.n_train_rows, headline_metric=headline, notes=row.notes,
            )
        )

    return ModelsDashboardOut(models=out, phases=MODEL_PHASES)
