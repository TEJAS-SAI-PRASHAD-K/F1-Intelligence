"""Request and response models for the pit-strategy endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.ml.strategy.inputs import DegradationSource


class StintSpec(BaseModel):
    compound: str = Field(..., description="SOFT, MEDIUM or HARD")
    laps: int = Field(..., ge=1, le=100)


class SimulateRequest(BaseModel):
    circuit: str = Field(..., description="Circuit name as stored, e.g. 'Sakhir'")
    race_laps: int | None = Field(
        None, ge=20, le=100, description="Defaults to the circuit's observed race distance"
    )
    max_stops: int = Field(3, ge=1, le=4)
    top_k: int = Field(8, ge=1, le=30)
    n_scenarios: int = Field(
        2000, ge=0, le=20000, description="Monte Carlo runs; 0 for deterministic only"
    )
    per_stop_count: bool = Field(
        True,
        description=(
            "Return the best strategy for each stop count rather than a flat top-k. "
            "A flat top-k is usually variants of one stop count differing by <0.3s, "
            "which hides the decision that matters."
        ),
    )
    degradation_source: DegradationSource = Field(
        DegradationSource.STAGE_A,
        description=(
            "stage_a = measured per-stint slopes (default). model = the Phase 2 "
            "pooled model, which under-predicts the measured rate by ~32% and so "
            "biases toward longer stints."
        ),
    )
    seed: int = Field(42, description="Monte Carlo seed, for reproducible scenarios")


class CompareRequest(BaseModel):
    circuit: str
    race_laps: int | None = Field(None, ge=20, le=100)
    strategies: list[list[StintSpec]] = Field(
        ..., min_length=2, max_length=6, description="Two to six strategies to compare"
    )
    n_scenarios: int = Field(2000, ge=1, le=20000)
    degradation_source: DegradationSource = DegradationSource.STAGE_A
    seed: int = 42


class StrategyOut(BaseModel):
    label: str
    description: str
    n_stops: int
    stints: list[StintSpec]
    stop_laps: list[int]

    deterministic_total_s: float
    deterministic_delta_s: float = Field(
        ..., description="Seconds behind the best deterministic strategy"
    )
    degradation_cost_s: float
    pit_cost_s: float

    # Present only when n_scenarios > 0.
    mc_median_s: float | None = None
    mc_delta_s: float | None = None
    mc_sd_s: float | None = None
    mc_p10_s: float | None = None
    mc_p90_s: float | None = None
    sc_luck_rate: float | None = Field(
        None, description="Fraction of scenarios where a stop landed under a safety car"
    )
    win_rate_vs_best: float | None = Field(
        None, description="Paired P(this strategy beats the best one) on shared scenarios"
    )
    margin_sd_s: float | None = Field(
        None,
        description=(
            "SD of the per-scenario MARGIN against the best strategy. Far smaller "
            "than mc_sd_s because shared randomness cancels -- this is the number "
            "that tells you whether a gap is real."
        ),
    )


class InputsOut(BaseModel):
    circuit: str
    race_laps: int
    base_pace_s: float
    pit_loss_s: float
    pit_loss_sd_s: float
    degradation_s_per_lap: dict[str, float]
    max_stint_laps: dict[str, int]
    sc_probability: float
    median_sc_laps: int
    degradation_source: str
    n_stops_observed: int
    n_races_observed: int
    notes: list[str] = Field(
        default_factory=list,
        description="Data-quality caveats, e.g. a fallback used for missing data",
    )


class SimulateResponse(BaseModel):
    inputs: InputsOut
    n_candidates_evaluated: int
    n_scenarios: int
    strategies: list[StrategyOut]
    limitations: list[str]


class CompareResponse(BaseModel):
    inputs: InputsOut
    n_scenarios: int
    strategies: list[StrategyOut]
    win_matrix: list[list[float]] = Field(
        ..., description="win_matrix[i][j] = P(strategy i beats strategy j), paired"
    )
    invalid: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Rejected strategies keyed by index, with the reasons",
    )
    limitations: list[str]
