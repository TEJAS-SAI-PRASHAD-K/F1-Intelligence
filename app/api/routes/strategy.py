"""Phase 3 endpoints: pit stop strategy simulation."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from app.db.base import engine
from app.ml.strategy import inputs as strategy_inputs
from app.ml.strategy import monte_carlo as mc
from app.ml.strategy import simulator as sim
from app.ml.strategy.enumerate_strategies import (
    Stint,
    Strategy,
    iter_candidates,
    validate,
)
from app.ml.strategy.inputs import StrategyInputs
from app.ml.tire import dataset as ds
from app.schemas.strategy import (
    CompareRequest,
    CompareResponse,
    InputsOut,
    SimulateRequest,
    SimulateResponse,
    StintSpec,
    StrategyOut,
)

router = APIRouter(prefix="/strategy", tags=["strategy"])

# Stated on every response. These are honest scope limits, not disclaimers --
# a user who does not know them will over-trust the numbers.
LIMITATIONS = [
    "Single-car model: no competitors, so no track position, traffic, undercut or overcut. "
    "Strategies are ranked by RACE TIME, not finishing position, and where overtaking is "
    "hard those differ.",
    "Degradation is linear in tire age. Real tires can fall off a cliff, which this "
    "under-penalises for very long stints.",
    "Safety cars affect the ranking only through pit timing; their lap-time penalty is "
    "identical for every strategy and cancels out.",
    "Dry races only. Wet-compound stints are excluded from the degradation data.",
]

# build_dataset reads ~100k rows and does several groupbys (~3s). The result is
# identical for every request until new data is ingested, so it is cached for
# the process with a short TTL rather than rebuilt per call.
_CACHE_TTL_SECONDS = 600.0
_laps_cache: dict[str, object] = {"frame": None, "at": 0.0}


def _cached_laps():
    now = time.monotonic()
    if _laps_cache["frame"] is None or now - float(_laps_cache["at"]) > _CACHE_TTL_SECONDS:
        _laps_cache["frame"] = ds.build_dataset(engine)
        _laps_cache["at"] = now
    return _laps_cache["frame"]


def clear_cache() -> None:
    """Drop the cached lap frame. Used by tests and after a re-ingest."""
    _laps_cache["frame"] = None
    _laps_cache["at"] = 0.0


def _build_inputs(circuit: str, race_laps: int | None, source) -> StrategyInputs:
    try:
        return strategy_inputs.build_inputs(
            engine, circuit, race_laps=race_laps, source=source, laps=_cached_laps()
        )
    except LookupError as exc:
        available = strategy_inputs.available_circuits(engine)
        raise HTTPException(
            404,
            f"No data for circuit {circuit!r}. Available circuits: {', '.join(available)}",
        ) from exc


def _inputs_out(inputs: StrategyInputs) -> InputsOut:
    return InputsOut(
        circuit=inputs.circuit,
        race_laps=inputs.race_laps,
        base_pace_s=round(inputs.base_pace_ms / 1000.0, 3),
        pit_loss_s=round(inputs.pit_loss_ms / 1000.0, 2),
        pit_loss_sd_s=round(inputs.pit_loss_sd_ms / 1000.0, 2),
        degradation_s_per_lap={
            k: round(v, 5) for k, v in inputs.degradation_s_per_lap.items()
        },
        max_stint_laps=inputs.max_stint_laps,
        sc_probability=round(inputs.sc_probability, 3),
        median_sc_laps=inputs.median_sc_laps,
        degradation_source=str(inputs.source),
        n_stops_observed=inputs.n_stops_observed,
        n_races_observed=inputs.n_races_observed,
        notes=inputs.notes,
    )


def _strategy_out(
    result: sim.RaceResult,
    det_best_ms: float,
    distribution: mc.StrategyDistribution | None = None,
    mc_best_ms: float | None = None,
    delta: mc.PairedDelta | None = None,
) -> StrategyOut:
    strategy = result.strategy
    out = StrategyOut(
        label=strategy.label(),
        description=strategy.describe(),
        n_stops=strategy.n_stops,
        stints=[StintSpec(compound=s.compound, laps=s.laps) for s in strategy.stints],
        stop_laps=list(strategy.stop_laps),
        deterministic_total_s=round(result.total_s, 3),
        deterministic_delta_s=round((result.total_ms - det_best_ms) / 1000.0, 3),
        degradation_cost_s=round(result.degradation_ms / 1000.0, 2),
        pit_cost_s=round(result.pit_ms / 1000.0, 2),
    )
    if distribution is not None:
        out.mc_median_s = round(distribution.median_ms / 1000.0, 3)
        out.mc_sd_s = round(distribution.sd_ms / 1000.0, 2)
        out.mc_p10_s = round(distribution.p10_ms / 1000.0, 2)
        out.mc_p90_s = round(distribution.p90_ms / 1000.0, 2)
        out.sc_luck_rate = round(distribution.sc_luck_rate, 3)
        if mc_best_ms is not None:
            out.mc_delta_s = round((distribution.median_ms - mc_best_ms) / 1000.0, 3)
    if delta is not None:
        out.win_rate_vs_best = round(delta.win_rate, 3)
        out.margin_sd_s = round(delta.sd_delta_ms / 1000.0, 2)
    return out


@router.post("/simulate", response_model=SimulateResponse)
def simulate_strategies(request: SimulateRequest):
    """Rank pit strategies for a circuit.

    Two layers, and the difference between them is the point:

    `deterministic_*` is what happens if the race goes to plan. It comes from an
    exhaustive search over every legal strategy, scored by a closed form.

    `mc_*` is what happens across thousands of races with random safety cars and
    pit fumbles. Read `win_rate_vs_best` and `margin_sd_s` rather than `mc_sd_s`:
    a single strategy's sd is dominated by the safety-car penalty that every
    strategy pays equally, while the paired margin isolates the real difference.

    `per_stop_count=true` (the default) returns the best 1-stop, 2-stop and
    3-stop rather than a flat top-k, because a flat list is usually variants of
    one stop count differing by under a second.
    """
    inputs = _build_inputs(request.circuit, request.race_laps, request.degradation_source)

    n_candidates = sum(1 for _ in iter_candidates(inputs, max_stops=request.max_stops))
    if n_candidates == 0:
        raise HTTPException(
            422,
            f"No legal strategy exists for {inputs.circuit} over {inputs.race_laps} laps "
            f"with at most {request.max_stops} stops. Observed stint limits: "
            f"{inputs.max_stint_laps}",
        )

    if request.per_stop_count:
        results = list(sim.rank_by_stop_count(inputs, max_stops=request.max_stops).values())
    else:
        results = sim.rank_strategies(
            inputs, max_stops=request.max_stops, top_k=request.top_k
        )

    det_best_ms = min(r.total_ms for r in results)

    if request.n_scenarios == 0:
        return SimulateResponse(
            inputs=_inputs_out(inputs),
            n_candidates_evaluated=n_candidates,
            n_scenarios=0,
            strategies=[_strategy_out(r, det_best_ms) for r in results],
            limitations=LIMITATIONS,
        )

    distributions, _ = mc.compare_strategies(
        [r.strategy for r in results],
        inputs,
        n_scenarios=request.n_scenarios,
        seed=request.seed,
    )
    best_dist = min(distributions, key=lambda d: d.median_ms)
    mc_best_ms = best_dist.median_ms

    out: list[StrategyOut] = []
    for result, distribution in zip(results, distributions, strict=True):
        delta = (
            None if distribution is best_dist else mc.paired_delta(distribution, best_dist)
        )
        out.append(_strategy_out(result, det_best_ms, distribution, mc_best_ms, delta))

    out.sort(key=lambda s: s.mc_median_s if s.mc_median_s is not None else 0.0)
    return SimulateResponse(
        inputs=_inputs_out(inputs),
        n_candidates_evaluated=n_candidates,
        n_scenarios=request.n_scenarios,
        strategies=out,
        limitations=LIMITATIONS,
    )


@router.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    """Compare strategies you specify, head to head.

    Illegal strategies are REJECTED with reasons rather than silently adjusted:
    a strategy whose stints do not sum to the race distance, or that uses one
    compound (which F1 regulations forbid in a dry race), is not something to
    quietly fix on the caller's behalf.

    All surviving strategies are run on the SAME random scenarios, so
    `win_matrix` is a paired comparison.
    """
    inputs = _build_inputs(request.circuit, request.race_laps, request.degradation_source)

    valid: list[Strategy] = []
    invalid: dict[str, list[str]] = {}
    for index, spec in enumerate(request.strategies):
        strategy = Strategy(
            tuple(Stint(compound=s.compound.upper(), laps=s.laps) for s in spec)
        )
        problems = validate(strategy, inputs)
        if problems:
            invalid[str(index)] = problems
        else:
            valid.append(strategy)

    if len(valid) < 2:
        raise HTTPException(
            422,
            {
                "detail": "Need at least two legal strategies to compare.",
                "invalid": invalid,
                "race_laps": inputs.race_laps,
            },
        )

    distributions, wins = mc.compare_strategies(
        valid, inputs, n_scenarios=request.n_scenarios, seed=request.seed
    )
    results = [sim.simulate(s, inputs) for s in valid]
    det_best_ms = min(r.total_ms for r in results)
    best_dist = min(distributions, key=lambda d: d.median_ms)

    out: list[StrategyOut] = []
    for result, distribution in zip(results, distributions, strict=True):
        delta = (
            None if distribution is best_dist else mc.paired_delta(distribution, best_dist)
        )
        out.append(
            _strategy_out(result, det_best_ms, distribution, best_dist.median_ms, delta)
        )

    return CompareResponse(
        inputs=_inputs_out(inputs),
        n_scenarios=request.n_scenarios,
        strategies=out,
        win_matrix=[[round(v, 4) for v in row] for row in wins.tolist()],
        invalid=invalid,
        limitations=LIMITATIONS,
    )


@router.get("/inputs/{circuit}", response_model=InputsOut)
def circuit_inputs(circuit: str, race_laps: int | None = None):
    """The measured inputs driving a circuit's simulation.

    Exposed because every recommendation follows from these six numbers. If a
    ranking looks wrong, this is where to look first -- usually `pit_loss_s` or
    a degradation rate resting on too few stints, which `notes` will say.
    """
    inputs = _build_inputs(circuit, race_laps, strategy_inputs.DegradationSource.STAGE_A)
    return _inputs_out(inputs)
