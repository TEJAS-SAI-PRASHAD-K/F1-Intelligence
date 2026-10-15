"""Response models for the dashboard endpoints."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class DriverStandingOut(BaseModel):
    driver: str
    full_name: str | None = None
    team: str | None = None
    races: int
    points: float = Field(
        ..., description="RACE points only. Sprint sessions are not ingested."
    )
    wins: int
    podiums: int
    points_finishes: int
    dnfs: int
    mean_finish: float
    mean_grid: float
    mean_places_gained: float = Field(
        ..., description="Positive means the driver gains places relative to their grid slot"
    )
    tire_mgmt_s_per_lap: float | None = Field(
        None,
        description=(
            "Degradation slope relative to the FIELD MEDIAN in the same race. "
            "Negative = preserves tires better than the field. Measured relative "
            "to the field so the fuel effect cancels without any assumption."
        ),
    )
    tire_mgmt_races: int | None = None


class TeamStandingOut(BaseModel):
    team: str
    entries: int
    drivers: int
    points: float
    wins: int
    podiums: int
    mean_finish: float
    mean_grid: float
    dnfs: int


class CircuitSeasonOut(BaseModel):
    circuit: str
    clean_laps: int
    median_lap_s: float
    fastest_lap_s: float
    air_temp: float | None = None
    track_temp: float | None = None
    pit_stops: int | None = None
    mean_stationary_s: float | None = None
    sc_laps: int | None = None


class SeasonDashboardOut(BaseModel):
    season_year: int
    races: int
    drivers: list[DriverStandingOut]
    teams: list[TeamStandingOut]
    circuits: list[CircuitSeasonOut]
    notes: list[str] = Field(default_factory=list)


class DriverSeasonSummaryOut(BaseModel):
    season_year: int
    team: str | None = None
    races: int
    points: float
    wins: int
    podiums: int
    dnfs: int
    mean_finish: float
    mean_places_gained: float
    tire_mgmt_s_per_lap: float | None = None


class CircuitStrengthOut(BaseModel):
    circuit: str
    races: int
    mean_finish: float
    mean_grid: float
    best_finish: float


class DriverDashboardOut(BaseModel):
    driver: str
    full_name: str | None = None
    country_code: str | None = None
    seasons: list[DriverSeasonSummaryOut]
    best_circuits: list[CircuitStrengthOut]
    worst_circuits: list[CircuitStrengthOut]
    career_races: int
    career_wins: int
    career_podiums: int


class StintChartEntryOut(BaseModel):
    driver: str
    stint_number: int
    compound: str | None = None
    lap_start: int
    lap_end: int
    laps: int
    median_green_lap_s: float | None = None
    degradation_s_per_lap: float | None = Field(
        None,
        description=(
            "OLS slope on FUEL-CORRECTED lap times (Phase 2 Stage A). Null when "
            "the stint has too few clean laps. An uncorrected slope would come "
            "out near zero or negative, because fuel burn nearly cancels wear."
        ),
    )


class RaceResultOut(BaseModel):
    position: float | None = None
    driver: str
    team: str | None = None
    grid_position: float | None = None
    points: float | None = None
    status: str | None = None
    places_gained: float | None = None


class PredictionVsActualOut(BaseModel):
    driver: str
    predicted_rank: int
    actual_position: float | None = None
    grid_position: float | None = None
    podium_probability: float
    rank_error: int | None = None


class RaceDashboardOut(BaseModel):
    event_id: int
    season_year: int
    round: int
    event_name: str
    circuit: str
    event_date: date | None = None
    total_laps: int | None = None
    results: list[RaceResultOut]
    stints: list[StintChartEntryOut]
    prediction_vs_actual: list[PredictionVsActualOut] = Field(default_factory=list)
    prediction_note: str | None = None


class ModelSummaryOut(BaseModel):
    name: str
    version: str
    backend: str
    is_active: bool
    trained_at: datetime | None = None
    train_season_min: int | None = None
    train_season_max: int | None = None
    n_train_rows: int | None = None
    headline_metric: str | None = Field(
        None, description="The model's own primary metric, as a readable string"
    )
    notes: str | None = None


class ModelsDashboardOut(BaseModel):
    models: list[ModelSummaryOut]
    phases: dict[str, str] = Field(
        default_factory=dict, description="Which phase each model family belongs to"
    )
