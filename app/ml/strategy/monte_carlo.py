"""Monte Carlo layer: race strategy under uncertainty.

The deterministic simulator answers "which strategy is fastest if everything
goes to plan". Real races do not, and the differences are large enough to
reverse the ranking. Three sources of randomness are modelled:

  1. PIT LOSS VARIANCE  -- crews fumble. Measured per circuit from the
     interdecile range of real stops, typically about +/-2.5s.
  2. LAP TIME NOISE     -- traffic, a missed apex, wind.
  3. SAFETY CARS        -- the big one. 68.6% of races in the ingested data
     have at least one, lasting a median of 5-9 laps depending on the circuit.

COMMON RANDOM NUMBERS -- the technique that makes this work
----------------------------------------------------------
Scenarios are drawn ONCE and every strategy is evaluated against the SAME ones.

This matters more than it sounds. We do not care about the absolute race time of
a strategy; we care about the DIFFERENCE between two strategies. If each strategy
got its own random draws, that difference would carry the variance of both draws,
and separating a genuine 0.5s edge from noise would need vastly more samples.
Pairing on scenarios cancels the shared randomness, so the comparison is far
sharper for the same compute. This is "common random numbers", and it is why
`compare_strategies` can report P(A beats B) from a few thousand runs.

WHY THE SAFETY CAR ONLY MATTERS THROUGH PIT TIMING
--------------------------------------------------
Worth being precise about, because it is easy to over-model.

When a safety car is deployed on lap k for d laps, every car on track runs those
laps slowly -- so EVERY strategy pays the same penalty, and it cancels out of any
comparison. The safety car changes the ranking through exactly one channel: a
stop made inside the window costs far less, because the whole field is already
slow. That is the "free pit stop".

So the SC lap-time penalty is computed (race times should be realistic) but it is
the stop-timing interaction that actually decides anything.

WHAT IS NOT MODELLED
--------------------
No competitors, so no track position, no traffic, no undercut or overcut. Those
need a multi-car simulation, and faking them in a single-car model would produce
confident numbers with no basis. The limitation is real: it means this ranks
strategies by RACE TIME, not by finishing position, and in a race where
overtaking is hard those differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ml.fuel import fuel_penalty_ms
from app.ml.strategy.enumerate_strategies import Strategy
from app.ml.strategy.inputs import StrategyInputs
from app.ml.strategy.simulator import degradation_cost_ms, fuel_total_ms

# Lap times under a safety car, as a multiple of green-flag pace. Cars circulate
# well below racing speed behind the SC; ~1.45x is the usual observed figure.
SC_LAP_FACTOR = 1.45

# What a pit stop costs when made under the safety car, as a fraction of its
# green-flag cost. The field is already slowed, so the time lost RELATIVE to
# staying out shrinks. ~0.45 is the standard approximation in F1 strategy work:
# a safety-car stop is worth roughly half a normal one. It is an assumption, and
# `sensitivity_to_sc_factor` exists to show how much the answer depends on it.
SC_PIT_LOSS_FACTOR = 0.45

# Per-lap noise. Lap-to-lap scatter within a clean stint measured on the Phase 2
# data has an MAE near 0.49s, implying an sd around 0.6s.
LAP_NOISE_SD_MS = 600.0

# Safety cars cluster early -- first-lap incidents, cold tires, close racing.
# Drawing the deployment lap from a Beta(2, 3) over the race distance puts more
# mass in the first half without forbidding a late one.
SC_LAP_BETA_A = 2.0
SC_LAP_BETA_B = 3.0


@dataclass
class Scenario:
    """One random draw of race conditions, shared across all strategies."""

    sc_lap: int | None
    sc_duration: int
    # Indexed by stop number; longer than any strategy needs so every strategy
    # reads the SAME multiplier for its Nth stop.
    pit_multipliers: np.ndarray
    lap_noise_ms: np.ndarray

    def sc_window(self) -> set[int]:
        if self.sc_lap is None:
            return set()
        return set(range(self.sc_lap, self.sc_lap + self.sc_duration))


@dataclass
class StrategyDistribution:
    strategy: Strategy
    total_ms: np.ndarray
    median_ms: float
    mean_ms: float
    sd_ms: float
    p10_ms: float
    p90_ms: float
    # Fraction of scenarios in which at least one stop landed under a safety car.
    sc_luck_rate: float
    deterministic_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def median_s(self) -> float:
        return self.median_ms / 1000.0


def draw_scenarios(
    inputs: StrategyInputs, n: int, *, max_stops: int = 4, seed: int = 42
) -> list[Scenario]:
    """Draw `n` race scenarios. Call once, reuse for every strategy."""
    rng = np.random.default_rng(seed)
    race_laps = inputs.race_laps

    has_sc = rng.random(n) < inputs.sc_probability
    # Beta over the race distance, so deployment clusters early.
    sc_fraction = rng.beta(SC_LAP_BETA_A, SC_LAP_BETA_B, size=n)
    sc_laps = 1 + (sc_fraction * (race_laps - 1)).astype(int)
    # Duration jitters around the circuit's observed median, floored at 2.
    durations = np.maximum(
        2, rng.poisson(max(inputs.median_sc_laps, 2), size=n)
    )

    pit_sd = max(inputs.pit_loss_sd_ms, 300.0)
    scenarios: list[Scenario] = []
    for i in range(n):
        multipliers = 1.0 + rng.normal(0.0, pit_sd / inputs.pit_loss_ms, size=max_stops + 1)
        scenarios.append(
            Scenario(
                sc_lap=int(sc_laps[i]) if has_sc[i] else None,
                sc_duration=int(durations[i]),
                # Clip so a draw cannot make a stop take negative time.
                pit_multipliers=np.clip(multipliers, 0.3, 3.0),
                lap_noise_ms=rng.normal(0.0, LAP_NOISE_SD_MS, size=race_laps),
            )
        )
    return scenarios


def simulate_scenario(
    strategy: Strategy,
    inputs: StrategyInputs,
    scenario: Scenario,
    *,
    sc_pit_loss_factor: float = SC_PIT_LOSS_FACTOR,
) -> tuple[float, bool]:
    """Race time for one strategy in one scenario.

    Returns `(total_ms, got_sc_luck)`, where the flag records whether any stop
    fell inside the safety-car window -- the only channel through which the
    safety car changes the ranking.
    """
    base_ms = inputs.base_pace_ms * inputs.race_laps
    fuel_ms = fuel_total_ms(inputs.race_laps)
    deg_ms = sum(
        degradation_cost_ms(inputs.degradation(s.compound), s.laps) for s in strategy.stints
    )
    noise_ms = float(scenario.lap_noise_ms.sum())

    window = scenario.sc_window()

    # Every strategy pays this identically, so it cannot change the ranking --
    # it is here only so reported race times are realistic.
    sc_penalty_ms = 0.0
    for lap in window:
        if 1 <= lap <= inputs.race_laps:
            sc_penalty_ms += inputs.base_pace_ms * (SC_LAP_FACTOR - 1.0)

    pit_ms = 0.0
    got_luck = False
    for stop_index, stop_lap in enumerate(strategy.stop_laps):
        multiplier = float(scenario.pit_multipliers[stop_index])
        cost = inputs.pit_loss_ms * multiplier
        if stop_lap in window:
            cost *= sc_pit_loss_factor
            got_luck = True
        pit_ms += cost

    return base_ms + fuel_ms + deg_ms + noise_ms + sc_penalty_ms + pit_ms, got_luck


def run_monte_carlo(
    strategy: Strategy,
    inputs: StrategyInputs,
    scenarios: list[Scenario],
    *,
    sc_pit_loss_factor: float = SC_PIT_LOSS_FACTOR,
) -> StrategyDistribution:
    """Evaluate one strategy across every scenario."""
    totals = np.empty(len(scenarios), dtype=float)
    luck = 0
    for i, scenario in enumerate(scenarios):
        totals[i], got = simulate_scenario(
            strategy, inputs, scenario, sc_pit_loss_factor=sc_pit_loss_factor
        )
        luck += int(got)

    from app.ml.strategy.simulator import simulate

    return StrategyDistribution(
        strategy=strategy,
        total_ms=totals,
        median_ms=float(np.median(totals)),
        mean_ms=float(totals.mean()),
        sd_ms=float(totals.std(ddof=1)) if len(totals) > 1 else 0.0,
        p10_ms=float(np.percentile(totals, 10)),
        p90_ms=float(np.percentile(totals, 90)),
        sc_luck_rate=luck / len(scenarios) if scenarios else 0.0,
        deterministic_ms=simulate(strategy, inputs).total_ms,
    )


def compare_strategies(
    strategies: list[Strategy],
    inputs: StrategyInputs,
    *,
    n_scenarios: int = 2000,
    seed: int = 42,
    sc_pit_loss_factor: float = SC_PIT_LOSS_FACTOR,
) -> tuple[list[StrategyDistribution], np.ndarray]:
    """Evaluate every strategy on shared scenarios and compute win probabilities.

    Returns the distributions plus a win matrix where `W[i][j]` is the fraction
    of scenarios in which strategy i finished ahead of strategy j. Because the
    scenarios are shared, that is a PAIRED comparison -- the same safety car and
    the same pit fumbles hit both strategies, so the probability reflects the
    strategies rather than the luck of two separate draws.
    """
    max_stops = max((s.n_stops for s in strategies), default=1)
    scenarios = draw_scenarios(inputs, n_scenarios, max_stops=max_stops + 1, seed=seed)

    distributions = [
        run_monte_carlo(s, inputs, scenarios, sc_pit_loss_factor=sc_pit_loss_factor)
        for s in strategies
    ]

    n = len(strategies)
    wins = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wins[i][j] = float(
                (distributions[i].total_ms < distributions[j].total_ms).mean()
            )
    return distributions, wins


@dataclass
class PairedDelta:
    """Head-to-head margin between two strategies, scenario by scenario."""

    median_delta_ms: float
    mean_delta_ms: float
    sd_delta_ms: float
    p10_delta_ms: float
    p90_delta_ms: float
    win_rate: float


def paired_delta(
    challenger: StrategyDistribution, reference: StrategyDistribution
) -> PairedDelta:
    """Per-scenario margin, which is the number that actually means something.

    The standard deviation of a single strategy's race time here is 130-185s,
    almost all of it the safety-car penalty -- which EVERY strategy pays
    identically. Quoting that sd suggests the strategies are indistinguishable
    when in fact one wins 78% of the time.

    Because both were run on the same scenarios, subtracting them per scenario
    cancels everything they share: the safety-car penalty, the lap noise, the
    fuel and base pace. What survives is the genuine difference. Measured at
    Sakhir: each strategy's race time has an sd of 131s, while the sd of the
    MARGIN between two of them is 8.7s -- a 15x reduction, from the same runs.

    This is the payoff of common random numbers, and it is worth seeing the two
    numbers side by side once to believe it.
    """
    delta = reference.total_ms - challenger.total_ms  # positive => challenger faster
    return PairedDelta(
        median_delta_ms=float(np.median(delta)),
        mean_delta_ms=float(delta.mean()),
        sd_delta_ms=float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        p10_delta_ms=float(np.percentile(delta, 10)),
        p90_delta_ms=float(np.percentile(delta, 90)),
        win_rate=float((delta > 0).mean()),
    )


def sensitivity_to_sc_factor(
    strategies: list[Strategy],
    inputs: StrategyInputs,
    *,
    factors: tuple[float, ...] = (0.25, 0.45, 0.65, 1.0),
    n_scenarios: int = 1000,
    seed: int = 42,
) -> dict[float, list[float]]:
    """How much does the ranking depend on SC_PIT_LOSS_FACTOR being right?

    That constant is the least defensible number in the model -- an
    approximation, not a measurement. Rather than defend it, vary it and see
    whether the recommendation survives. A conclusion that flips between 0.45
    and 0.65 should be reported as uncertain; one that holds across the range
    can be stated plainly.
    """
    out: dict[float, list[float]] = {}
    for factor in factors:
        distributions, _ = compare_strategies(
            strategies,
            inputs,
            n_scenarios=n_scenarios,
            seed=seed,
            sc_pit_loss_factor=factor,
        )
        out[factor] = [d.median_ms for d in distributions]
    return out


def lap_times_for_scenario(
    strategy: Strategy, inputs: StrategyInputs, scenario: Scenario
) -> list[float]:
    """Per-lap times for one scenario, for plotting a single race.

    Not used in ranking; it exists so a notebook can show what one simulated
    race actually looked like, including the safety-car period.
    """
    window = scenario.sc_window()
    times: list[float] = []
    lap_number = 0
    for stint in strategy.stints:
        slope_ms = inputs.degradation(stint.compound) * 1000.0
        for age in range(1, stint.laps + 1):
            lap_number += 1
            lap_ms = (
                inputs.base_pace_ms
                + slope_ms * (age - 1)
                + fuel_penalty_ms(lap_number, inputs.race_laps)
                + float(scenario.lap_noise_ms[lap_number - 1])
            )
            if lap_number in window:
                lap_ms *= SC_LAP_FACTOR
            times.append(lap_ms)
    return times
