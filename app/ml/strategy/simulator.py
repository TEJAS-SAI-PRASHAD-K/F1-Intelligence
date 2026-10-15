"""Deterministic race-time simulation for a pit strategy.

THE MODEL
---------
Total race time is the sum of every lap plus the cost of every stop:

    race_time = sum over laps of lap_time(lap) + n_stops * pit_loss

and a single lap, on a tire `age` laps old at race lap `lap`, is:

    lap_time = base_pace                       # fresh tire, empty tank
             + degradation(compound) * (age-1) # tires wear
             + fuel_penalty(lap, race_laps)    # fuel still aboard slows the car

`base_pace` is the empty-tank, fresh-tire reference from `inputs.compute_base_pace`,
and `fuel_penalty` is Phase 2's validated fuel model -- reused rather than
reinvented, so the simulator and the tire model cannot disagree about fuel.

TWO IMPLEMENTATIONS, ON PURPOSE
-------------------------------
`simulate_lap_by_lap` walks every lap. It is the honest, extensible version: it
can be given a safety car, per-lap noise, or a traffic penalty.

`deterministic_cost` evaluates a closed form. Summing the degradation term over
a stint of length L gives an arithmetic series:

    sum over age=1..L of slope*(age-1) = slope * L*(L-1)/2

so a whole strategy scores in O(stints) instead of O(laps), which is what makes
exhaustively ranking 2.4 million Monaco strategies practical.

They must agree to the floating-point bit, and
`tests/test_strategy_simulator.py` asserts exactly that. The closed form is the
fast path; the loop is the specification.

WHAT THE CLOSED FORM REVEALS
----------------------------
    cost(strategy) = sum of slope(c) * L*(L-1)/2  +  stops * pit_loss

Three things follow immediately, without running anything:

1. The strategy-dependent cost contains NO term for stint ORDER. Running soft
   first or last scores identically. Order only matters once a safety car or
   track position breaks the symmetry.
2. The L*(L-1)/2 term is convex in L, so the optimal split of laps between two
   stints on the same compound is an even one -- long stints are punished
   quadratically.
3. An extra stop costs a fixed `pit_loss` and buys a quadratic reduction in
   degradation. That trade is the entire game, and it is why abrasive circuits
   (Sakhir, ~0.14 s/lap) want more stops than smooth ones (Monaco, ~0.01 s/lap).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from app.ml.fuel import fuel_penalty_ms
from app.ml.strategy.enumerate_strategies import (
    MIN_STINT_LAPS,
    Strategy,
    build_strategy,
    iter_candidates,
)
from app.ml.strategy.inputs import DRY_COMPOUNDS, StrategyInputs


@dataclass
class LapRecord:
    lap_number: int
    stint_index: int
    compound: str
    tyre_age: int
    lap_time_ms: float
    is_pit_lap: bool


@dataclass
class RaceResult:
    strategy: Strategy
    total_ms: float
    racing_ms: float
    pit_ms: float
    degradation_ms: float
    fuel_ms: float
    laps: list[LapRecord] | None = None

    @property
    def total_s(self) -> float:
        return self.total_ms / 1000.0


def fuel_total_ms(race_laps: int) -> float:
    """Total fuel penalty across a whole race.

    Depends only on race distance, so it is identical for every strategy and
    cancels out of any ranking. Computed anyway so reported race times are
    absolute rather than relative.
    """
    return sum(fuel_penalty_ms(lap, race_laps) for lap in range(1, race_laps + 1))


def degradation_cost_ms(slope_s_per_lap: float, stint_laps: int) -> float:
    """Total time lost to tire wear over one stint, in ms.

    sum over age=1..L of slope*(age-1) = slope * L*(L-1)/2.

    Note L*(L-1)/2, not L*L/2: a fresh tire (age 1) has lost nothing, so the
    series starts at zero. Getting this off by one would overstate every
    strategy's degradation by exactly one lap's worth.
    """
    return slope_s_per_lap * 1000.0 * stint_laps * (stint_laps - 1) / 2.0


def deterministic_cost(
    compounds: tuple[str, ...], lengths: tuple[int, ...], inputs: StrategyInputs
) -> float:
    """Strategy-dependent cost in ms: degradation plus pit loss.

    Deliberately EXCLUDES base pace and fuel, which are identical for every
    strategy at a given race distance. Omitting them keeps the fast path cheap
    and makes the numbers directly comparable between strategies.
    """
    total = 0.0
    for compound, laps in zip(compounds, lengths, strict=True):
        total += degradation_cost_ms(inputs.degradation(compound), laps)
    total += (len(lengths) - 1) * inputs.pit_loss_ms
    return total


def simulate_lap_by_lap(
    strategy: Strategy, inputs: StrategyInputs, *, keep_laps: bool = False
) -> RaceResult:
    """Walk every lap of the race. The specification implementation."""
    records: list[LapRecord] = [] if keep_laps else []
    lap_number = 0
    racing_ms = 0.0
    degradation_ms = 0.0
    fuel_ms = 0.0

    for stint_index, stint in enumerate(strategy.stints):
        slope_ms = inputs.degradation(stint.compound) * 1000.0
        for age in range(1, stint.laps + 1):
            lap_number += 1
            deg = slope_ms * (age - 1)
            fuel = fuel_penalty_ms(lap_number, inputs.race_laps)
            lap_ms = inputs.base_pace_ms + deg + fuel

            racing_ms += lap_ms
            degradation_ms += deg
            fuel_ms += fuel

            if keep_laps:
                records.append(
                    LapRecord(
                        lap_number=lap_number,
                        stint_index=stint_index,
                        compound=stint.compound,
                        tyre_age=age,
                        lap_time_ms=lap_ms,
                        is_pit_lap=(age == stint.laps and stint_index < len(strategy.stints) - 1),
                    )
                )

    pit_ms = strategy.n_stops * inputs.pit_loss_ms
    return RaceResult(
        strategy=strategy,
        total_ms=racing_ms + pit_ms,
        racing_ms=racing_ms,
        pit_ms=pit_ms,
        degradation_ms=degradation_ms,
        fuel_ms=fuel_ms,
        laps=records if keep_laps else None,
    )


def simulate(strategy: Strategy, inputs: StrategyInputs) -> RaceResult:
    """Fast path: closed form, with the same fields as the lap-by-lap version."""
    deg_ms = sum(
        degradation_cost_ms(inputs.degradation(s.compound), s.laps) for s in strategy.stints
    )
    fuel_ms = fuel_total_ms(inputs.race_laps)
    base_ms = inputs.base_pace_ms * inputs.race_laps
    pit_ms = strategy.n_stops * inputs.pit_loss_ms

    return RaceResult(
        strategy=strategy,
        total_ms=base_ms + deg_ms + fuel_ms + pit_ms,
        racing_ms=base_ms + deg_ms + fuel_ms,
        pit_ms=pit_ms,
        degradation_ms=deg_ms,
        fuel_ms=fuel_ms,
    )


def _plan_key(
    compounds: tuple[str, ...], lengths: tuple[int, ...]
) -> tuple[tuple[str, int], ...]:
    """Canonical key for strategies the deterministic model cannot distinguish.

    Because the cost contains no ordering term, every permutation of the same
    (compound, length) assignment scores identically. Without collapsing them a
    top-10 is mostly permutations of one plan: at Monza the first four results
    were "SOFT 28 / HARD 25" and "HARD 25 / SOFT 28" and their neighbours, all
    within 0.02s, which tells a user nothing.

    Sorting makes the key order-insensitive, so one representative survives per
    distinct plan. The Monte Carlo layer DOES distinguish orderings -- a safety
    car arrives at a specific lap -- so ordering is compared there, where it
    actually changes the answer.
    """
    return tuple(sorted(zip(compounds, lengths, strict=True)))


def rank_strategies(
    inputs: StrategyInputs,
    *,
    max_stops: int = 3,
    top_k: int = 10,
    min_stint: int = MIN_STINT_LAPS,
    compounds: tuple[str, ...] = DRY_COMPOUNDS,
    deduplicate: bool = True,
) -> list[RaceResult]:
    """Exhaustively score every legal strategy; return the best `top_k` plans.

    Memory is O(candidates seen) when deduplicating, because one best-cost entry
    is kept per distinct plan; set `deduplicate=False` for a bounded-heap pass
    that keeps raw permutations.

    Scoring uses the closed form. The winners are then re-run through `simulate`
    so their reported breakdown fields are complete.
    """
    if not deduplicate:
        heap: list[tuple[float, int, tuple[str, ...], tuple[int, ...]]] = []
        tiebreak = 0
        for sequence, lengths in iter_candidates(
            inputs, max_stops=max_stops, min_stint=min_stint, compounds=compounds
        ):
            cost = deterministic_cost(sequence, lengths, inputs)
            tiebreak += 1
            # Negated: heapq is a min-heap and we want to evict the WORST.
            entry = (-cost, tiebreak, sequence, lengths)
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
        winners = sorted(heap, key=lambda e: -e[0])
        return [
            simulate(build_strategy(seq, lengths), inputs) for _, _, seq, lengths in winners
        ]

    best: dict[tuple[tuple[str, int], ...], tuple[float, tuple[str, ...], tuple[int, ...]]] = {}
    for sequence, lengths in iter_candidates(
        inputs, max_stops=max_stops, min_stint=min_stint, compounds=compounds
    ):
        cost = deterministic_cost(sequence, lengths, inputs)
        key = _plan_key(sequence, lengths)
        current = best.get(key)
        if current is None or cost < current[0]:
            best[key] = (cost, sequence, lengths)

    ranked = heapq.nsmallest(top_k, best.values(), key=lambda v: v[0])
    return [simulate(build_strategy(seq, lengths), inputs) for _, seq, lengths in ranked]


def rank_by_stop_count(
    inputs: StrategyInputs,
    *,
    max_stops: int = 3,
    min_stint: int = MIN_STINT_LAPS,
    compounds: tuple[str, ...] = DRY_COMPOUNDS,
) -> dict[int, RaceResult]:
    """The single best strategy for each number of stops.

    A plain top-10 is dominated by near-identical variants of whichever stop
    count wins -- at Sakhir all ten were 2-stops differing by under 0.3s, which
    answers a question nobody asked. The decision a strategist actually faces is
    "one stop or two", so this returns the best representative of each, making
    the stop-count trade-off visible.

    It is also the comparison worth taking to the Monte Carlo: strategies with
    different stop counts have genuinely different safety-car exposure, where
    variants of one stop count do not.
    """
    best: dict[int, tuple[float, tuple[str, ...], tuple[int, ...]]] = {}
    for sequence, lengths in iter_candidates(
        inputs, max_stops=max_stops, min_stint=min_stint, compounds=compounds
    ):
        n_stops = len(lengths) - 1
        cost = deterministic_cost(sequence, lengths, inputs)
        current = best.get(n_stops)
        if current is None or cost < current[0]:
            best[n_stops] = (cost, sequence, lengths)

    return {
        n_stops: simulate(build_strategy(seq, lengths), inputs)
        for n_stops, (_, seq, lengths) in sorted(best.items())
    }


def optimal_stint_split(n_stints: int, race_laps: int, slope_s_per_lap: float) -> list[int]:
    """Best way to divide a race into `n_stints` on ONE compound.

    Not used by the ranking -- it exists because it has an exact answer, which
    makes it a test oracle. Since cost is slope * sum of L*(L-1)/2 and that is
    convex, the optimum is the most even split possible. `slope` does not appear
    in the answer at all: it scales every option identically.

    Confirming the brute-force search agrees with this on a single-compound
    problem is how we know the search is correct, not merely plausible.
    """
    base, remainder = divmod(race_laps, n_stints)
    return [base + 1] * remainder + [base] * (n_stints - remainder)
