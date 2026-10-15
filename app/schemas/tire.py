"""Request and response models for the tire-degradation endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DegradationRequest(BaseModel):
    circuit: str = Field(..., description="Circuit name as stored, e.g. 'Monza'")
    compound: str = Field(..., description="SOFT, MEDIUM or HARD")
    max_tyre_life: int = Field(30, ge=2, le=60, description="Laps of tire age to project")
    race_laps: int = Field(57, ge=20, le=100)
    stint_start_lap: int = Field(
        1, ge=1, description="Race lap the stint begins on."
    )
    reference_lap: int | None = Field(
        None,
        ge=1,
        description=(
            "Race lap at which lap_number is pinned for the whole curve. "
            "Defaults to the stint midpoint. Held constant so the curve shows "
            "the PARTIAL effect of tire age: sweeping lap_number alongside "
            "tyre_life makes the curve non-monotonic, because the monotonic "
            "constraint binds tyre_life only with other features fixed."
        ),
    )
    air_temp: float | None = Field(25.0, ge=-10, le=60)
    track_temp: float | None = Field(38.0, ge=-10, le=80)
    driver: str | None = Field(None, description="Optional driver abbreviation, e.g. VER")
    team: str | None = None
    stint_number: int = Field(1, ge=1, le=8)
    is_fresh_tyre: bool = True


class DegradationPoint(BaseModel):
    tyre_life: float
    # Interpretable: how much slower than the same tire when new.
    delta_vs_fresh_ms: float
    delta_vs_fresh_s: float
    # Raw model output: deviation from the stint's fuel-corrected median.
    delta_vs_stint_median_ms: float


class DegradationResponse(BaseModel):
    circuit: str
    compound: str
    degradation_rate_s_per_lap: float = Field(
        ..., description="Slope of a straight line through the curve"
    )
    total_loss_s: float = Field(
        ..., description="Predicted loss from new to max_tyre_life, in seconds"
    )
    points: list[DegradationPoint]
    model_version: str
    model_backend: str
    unseen_categories: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Categories not present in training, mapped to NaN. Non-empty means "
            "the prediction fell back toward the global average for those fields."
        ),
    )


class StintFitOut(BaseModel):
    """An observed Stage A fit: what a real stint actually did."""

    driver: str
    stint_number: int
    compound: str | None = None
    laps: int
    tyre_life_min: float
    tyre_life_max: float
    # Degradation rate measured by OLS on fuel-corrected lap times.
    slope_s_per_lap: float
    intercept_s: float
    r2: float


class ModelMetricsOut(BaseModel):
    mae_s: float
    rmse_s: float
    r2: float
    n: int


class ModelInfoOut(BaseModel):
    name: str
    version: str
    backend: str
    trained_at: datetime | None = None
    train_season_min: int | None = None
    train_season_max: int | None = None
    n_train_rows: int | None = None
    holdout: ModelMetricsOut | None = None
    baseline_mae_s: float | None = Field(
        None, description="MAE of 'this lap == its stint median'. The model must beat it."
    )
    lift_vs_baseline: float | None = Field(
        None, description="Fractional improvement over the baseline on the holdout season"
    )
    control_histgbr_mae_s: float | None = Field(
        None,
        description=(
            "Holdout MAE of the sklearn HistGradientBoosting control model. If this "
            "beats the primary XGBoost model, suspect over-tuning or an eval-set leak."
        ),
    )
    notes: str | None = None
