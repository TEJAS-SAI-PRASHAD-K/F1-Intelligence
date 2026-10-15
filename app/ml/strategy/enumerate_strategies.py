"""Enumerate every legal pit strategy for a race.

A strategy is an ordered sequence of (compound, stint_length) whose lengths sum
to the race distance.

THE RULES ARE REAL REGULATIONS, NOT MODELING CHOICES
----------------------------------------------------
1. At least two different dry compounds must be used in a dry race. This is an
   actual F1 sporting regulation, and it is what forces a minimum of one stop --
   not any assumption about tire wear.
2. Stint lengths are bounded below by MIN_STINT_LAPS (a 2-lap stint is a
   mistake or damage repair, not a strategy) and above by what teams have
   actually achieved on that compound at that circuit.

WHY BRUTE FORCE, AND WHY IT GENERATES TUPLES RATHER THAN OBJECTS
----------------------------------------------------------------
With pruning the space is small enough to enumerate exactly, and an exact answer
that is easy to debug beats a heuristic here. The deterministic cost has a closed
form (see simulator.py), so scoring one candidate is O(stints), not O(laps).

But the space is larger than it first looks. Monaco is 78 laps and its tires
barely degrade, so stints up to 45 laps are legal, giving **2.46 million** legal
strategies -- of which 2.43 million are 3-stop. Building a `Strategy` dataclass
for each took 17 seconds, essentially all of it object construction for
candidates that were about to be thrown away.

So `iter_candidates` yields plain `(compounds, lengths)` tuples and
`simulator.rank_strategies` materialises `Strategy` objects only for the top few.
Same exhaustive search, ~10x faster.

ORDER MATTERS, BUT NOT FOR THE DETERMINISTIC COST
-------------------------------------------------
Worth understanding, because it is a useful check on the model. Deterministic
cost is sum over stints of slope(compound) * L*(L-1)/2 plus stops * pit_loss,
which contains no term for WHICH stint came first -- so (SOFT 20, HARD 33) and
(HARD 33, SOFT 20) score identically.

Order only starts to matter once something breaks that symmetry: a safety car
arriving at a particular lap, starting-tire rules, or track position. The Monte
Carlo layer introduces exactly that, which is why strategies are enumerated as
ordered sequences even though the deterministic ranking cannot tell them apart.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product

from app.ml.strategy.inputs import DRY_COMPOUNDS, MIN_STINT_LAPS, StrategyInputs


@dataclass(frozen=True)
class Stint:
    compound: str
    laps: int


@dataclass(frozen=True)
class Strategy:
    """An ordered sequence of stints covering the full race distance."""

    stints: tuple[Stint, ...]

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def total_laps(self) -> int:
        return sum(s.laps for s in self.stints)

    @property
    def compounds_used(self) -> set[str]:
        return {s.compound for s in self.stints}

    @property
    def stop_laps(self) -> tuple[int, ...]:
        """Race laps on which a stop happens (end of each stint but the last)."""
        laps: list[int] = []
        cumulative = 0
        for stint in self.stints[:-1]:
            cumulative += stint.laps
            laps.append(cumulative)
        return tuple(laps)

    def label(self) -> str:
        return " / ".join(f"{s.compound[0]}{s.laps}" for s in self.stints)

    def describe(self) -> str:
        parts = " -> ".join(f"{s.compound} x{s.laps}" for s in self.stints)
        return f"{self.n_stops}-stop: {parts}"


def _lap_compositions(
    total: int, n_parts: int, minimum: int, maxima: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """Every way to split `total` laps into `n_parts` stints within bounds.

    `maxima` is per-part, because a HARD tire can run longer than a SOFT one at
    the same circuit. Recursion with an early feasibility check: if even the
    maximum remaining capacity cannot reach `total`, the branch is abandoned
    rather than explored.
    """
    results: list[tuple[int, ...]] = []

    def recurse(remaining: int, index: int, acc: list[int]) -> None:
        if index == n_parts - 1:
            # Last stint takes whatever is left, if that is legal.
            if minimum <= remaining <= maxima[index]:
                results.append((*acc, remaining))
            return

        # Feasibility: the stints after this one can absorb at most this much.
        max_rest = sum(maxima[index + 1 :])
        min_rest = minimum * (n_parts - index - 1)

        lo = max(minimum, remaining - max_rest)
        hi = min(maxima[index], remaining - min_rest)
        for length in range(lo, hi + 1):
            acc.append(length)
            recurse(remaining - length, index + 1, acc)
            acc.pop()

    recurse(total, 0, [])
    return results


def iter_candidates(
    inputs: StrategyInputs,
    *,
    max_stops: int = 3,
    min_stint: int = MIN_STINT_LAPS,
    compounds: tuple[str, ...] = DRY_COMPOUNDS,
) -> Iterator[tuple[tuple[str, ...], tuple[int, ...]]]:
    """Yield every legal `(compounds, lengths)` pair, without building objects.

    A generator so a caller can score millions of candidates while holding only
    a top-K heap in memory.
    """
    race_laps = inputs.race_laps

    for n_stints in range(2, max_stops + 2):
        if min_stint * n_stints > race_laps:
            continue
        for sequence in product(compounds, repeat=n_stints):
            # The two-compound regulation.
            if len(set(sequence)) < 2:
                continue
            maxima = tuple(inputs.max_stint(c) for c in sequence)
            if sum(maxima) < race_laps:
                continue
            for lengths in _lap_compositions(race_laps, n_stints, min_stint, maxima):
                yield sequence, lengths


def build_strategy(compounds: tuple[str, ...], lengths: tuple[int, ...]) -> Strategy:
    """Materialise a `Strategy` from the tuple form `iter_candidates` yields."""
    return Strategy(
        tuple(
            Stint(compound=c, laps=n) for c, n in zip(compounds, lengths, strict=True)
        )
    )


def enumerate_strategies(
    inputs: StrategyInputs,
    *,
    max_stops: int = 3,
    min_stint: int = MIN_STINT_LAPS,
    compounds: tuple[str, ...] = DRY_COMPOUNDS,
) -> list[Strategy]:
    """Eager version of `iter_candidates`, for tests and small races.

    Prefer `iter_candidates` in anything that scores a full space -- this builds
    one object per candidate, which is 2.46 million of them at Monaco.
    """
    return [
        build_strategy(seq, lengths)
        for seq, lengths in iter_candidates(
            inputs, max_stops=max_stops, min_stint=min_stint, compounds=compounds
        )
    ]


def validate(
    strategy: Strategy, inputs: StrategyInputs, min_stint: int = MIN_STINT_LAPS
) -> list[str]:
    """Reasons a strategy is illegal, or an empty list if it is fine.

    Used by the /compare endpoint, where a caller supplies their own strategy
    and deserves to be told exactly what is wrong with it rather than getting a
    silently adjusted answer.
    """
    problems: list[str] = []

    if strategy.total_laps != inputs.race_laps:
        problems.append(
            f"stint lengths sum to {strategy.total_laps}, but {inputs.circuit} "
            f"is {inputs.race_laps} laps"
        )
    if len(strategy.stints) < 2:
        problems.append("a dry race requires at least one stop")
    if len(strategy.compounds_used) < 2:
        problems.append(
            "F1 regulations require at least two different dry compounds in a dry race"
        )
    for i, stint in enumerate(strategy.stints, start=1):
        if stint.compound not in inputs.degradation_s_per_lap:
            problems.append(f"stint {i}: unknown dry compound {stint.compound!r}")
            continue
        if stint.laps < min_stint:
            problems.append(f"stint {i}: {stint.laps} laps is below the {min_stint}-lap minimum")
        limit = inputs.max_stint(stint.compound)
        if stint.laps > limit:
            problems.append(
                f"stint {i}: {stint.laps} laps on {stint.compound} exceeds the "
                f"{limit}-lap maximum observed at {inputs.circuit}"
            )
    return problems
