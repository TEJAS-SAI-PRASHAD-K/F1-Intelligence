"""Tests for the pit-strategy simulator.

The structure of these tests is the point: the deterministic cost has a closed
form AND a lap-by-lap implementation, so they can be checked against each other.
Two independent implementations of one quantity is the strongest test available
short of real-world validation.
"""

from __future__ import annotations

import pytest

from app.ml.strategy import monte_carlo as mc
from app.ml.strategy import simulator as sim
from app.ml.strategy.enumerate_strategies import (
    Stint,
    Strategy,
    build_strategy,
    enumerate_strategies,
    iter_candidates,
    validate,
)
from app.ml.strategy.inputs import StrategyInputs


@pytest.fixture
def inputs() -> StrategyInputs:
    """A synthetic circuit with round numbers, so expected values are computable.

    Degradation differs clearly by compound (0.12 / 0.06 / 0.03 s/lap) so tests
    can assert the simulator uses the compound rather than ignoring it.
    """
    return StrategyInputs(
        circuit="TestPark",
        race_laps=50,
        base_pace_ms=90_000.0,
        pit_loss_ms=22_000.0,
        pit_loss_sd_ms=2_000.0,
        degradation_s_per_lap={"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.03},
        max_stint_laps={"SOFT": 25, "MEDIUM": 35, "HARD": 45},
        sc_probability=0.7,
        median_sc_laps=5,
    )


class TestDegradationCost:
    def test_fresh_tire_has_lost_nothing(self):
        """A one-lap stint has zero degradation: age 1 is the reference.
        L*(L-1)/2 rather than L*L/2 -- off by one here would overstate every
        strategy's degradation by a lap's worth."""
        assert sim.degradation_cost_ms(0.06, 1) == 0.0

    def test_arithmetic_series(self):
        """Sum over age=1..L of slope*(age-1) = slope * L*(L-1)/2."""
        # 10 laps at 0.06 s/lap -> 0.06 * 45 = 2.7s
        assert sim.degradation_cost_ms(0.06, 10) == pytest.approx(2_700.0)

    def test_cost_is_convex_in_stint_length(self):
        """Doubling a stint MORE than doubles its degradation cost. This is the
        quadratic term that makes extra stops worth their pit loss."""
        short = sim.degradation_cost_ms(0.06, 10)
        long = sim.degradation_cost_ms(0.06, 20)
        assert long > 2 * short

    def test_zero_slope_costs_nothing(self):
        assert sim.degradation_cost_ms(0.0, 40) == 0.0


class TestClosedFormMatchesLapByLap:
    """The central correctness property of the simulator."""

    @pytest.mark.parametrize(
        "spec",
        [
            (("SOFT", 20), ("HARD", 30)),
            (("MEDIUM", 17), ("MEDIUM", 17), ("HARD", 16)),
            (("SOFT", 12), ("MEDIUM", 13), ("HARD", 13), ("HARD", 12)),
        ],
    )
    def test_two_implementations_agree(self, inputs, spec):
        strategy = Strategy(tuple(Stint(c, n) for c, n in spec))
        closed = sim.simulate(strategy, inputs)
        looped = sim.simulate_lap_by_lap(strategy, inputs)
        assert closed.total_ms == pytest.approx(looped.total_ms, abs=1e-6)
        assert closed.degradation_ms == pytest.approx(looped.degradation_ms, abs=1e-6)
        assert closed.fuel_ms == pytest.approx(looped.fuel_ms, abs=1e-6)
        assert closed.pit_ms == pytest.approx(looped.pit_ms, abs=1e-9)

    def test_lap_by_lap_runs_every_lap(self, inputs):
        strategy = Strategy((Stint("SOFT", 20), Stint("HARD", 30)))
        result = sim.simulate_lap_by_lap(strategy, inputs, keep_laps=True)
        assert len(result.laps) == inputs.race_laps
        assert [lap.lap_number for lap in result.laps] == list(range(1, 51))

    def test_tyre_age_resets_at_each_stop(self, inputs):
        strategy = Strategy((Stint("SOFT", 20), Stint("HARD", 30)))
        laps = sim.simulate_lap_by_lap(strategy, inputs, keep_laps=True).laps
        assert laps[0].tyre_age == 1
        assert laps[19].tyre_age == 20
        assert laps[20].tyre_age == 1, "a new tire starts at age 1, not 21"
        assert laps[20].compound == "HARD"


class TestOrderInvariance:
    def test_deterministic_cost_ignores_stint_order(self, inputs):
        """The closed form contains no ordering term, so permutations tie.

        This is not a quirk to work around -- it is why `rank_strategies`
        deduplicates, and why ordering is only compared in the Monte Carlo,
        where a safety car at a specific lap breaks the symmetry.
        """
        a = Strategy((Stint("SOFT", 20), Stint("HARD", 30)))
        b = Strategy((Stint("HARD", 30), Stint("SOFT", 20)))
        assert sim.simulate(a, inputs).total_ms == pytest.approx(
            sim.simulate(b, inputs).total_ms
        )

    def test_monte_carlo_does_distinguish_order(self, inputs):
        """Because the stop falls on a different lap, so safety-car exposure
        differs. If this ever tied, the safety-car model would be inert."""
        a = Strategy((Stint("SOFT", 12), Stint("HARD", 38)))
        b = Strategy((Stint("HARD", 38), Stint("SOFT", 12)))
        scenarios = mc.draw_scenarios(inputs, 400, seed=7)
        da = mc.run_monte_carlo(a, inputs, scenarios)
        db = mc.run_monte_carlo(b, inputs, scenarios)
        assert a.stop_laps != b.stop_laps
        assert da.sc_luck_rate != db.sc_luck_rate


class TestAnalyticOptimum:
    def test_even_split_is_optimal_on_one_compound(self):
        """Convex cost means the best split is the most even one, and the slope
        does not enter the answer -- it scales every option identically."""
        assert sim.optimal_stint_split(2, 50, 0.06) == [25, 25]
        assert sim.optimal_stint_split(3, 50, 0.06) == [17, 17, 16]
        assert sim.optimal_stint_split(3, 50, 0.99) == [17, 17, 16]

    def test_brute_force_agrees_with_the_analytic_optimum(self, inputs):
        """The search is validated against a known answer, not just inspected.

        Restricting both compounds to the same degradation rate makes the
        problem single-compound, where the optimum is provably the even split.
        """
        flat = StrategyInputs(
            circuit="Flat",
            race_laps=50,
            base_pace_ms=90_000.0,
            pit_loss_ms=22_000.0,
            pit_loss_sd_ms=1_000.0,
            degradation_s_per_lap={"SOFT": 0.06, "MEDIUM": 0.06, "HARD": 0.06},
            max_stint_laps={"SOFT": 40, "MEDIUM": 40, "HARD": 40},
            sc_probability=0.0,
            median_sc_laps=0,
        )
        best = sim.rank_by_stop_count(flat, max_stops=2)
        one_stop_lengths = sorted(s.laps for s in best[1].strategy.stints)
        assert one_stop_lengths == [25, 25]
        two_stop_lengths = sorted(s.laps for s in best[2].strategy.stints)
        assert two_stop_lengths == [16, 17, 17]


class TestEnumeration:
    def test_every_strategy_covers_the_race_distance(self, inputs):
        for strategy in enumerate_strategies(inputs, max_stops=2):
            assert strategy.total_laps == inputs.race_laps

    def test_two_compound_regulation_is_enforced(self, inputs):
        """An actual F1 sporting regulation, and what forces a minimum of one
        stop -- not any assumption about tire wear."""
        for strategy in enumerate_strategies(inputs, max_stops=2):
            assert len(strategy.compounds_used) >= 2

    def test_stint_limits_are_respected(self, inputs):
        for strategy in enumerate_strategies(inputs, max_stops=2):
            for stint in strategy.stints:
                assert 5 <= stint.laps <= inputs.max_stint(stint.compound)

    def test_generator_and_eager_versions_agree(self, inputs):
        eager = enumerate_strategies(inputs, max_stops=2)
        streamed = [
            build_strategy(seq, lengths)
            for seq, lengths in iter_candidates(inputs, max_stops=2)
        ]
        assert eager == streamed

    def test_stop_laps_are_cumulative(self):
        strategy = Strategy((Stint("SOFT", 20), Stint("MEDIUM", 15), Stint("HARD", 15)))
        assert strategy.stop_laps == (20, 35)
        assert strategy.n_stops == 2

    def test_impossible_race_yields_nothing(self, inputs):
        """A race longer than the combined maximum stint life has no legal
        single-stop strategy; the enumerator must return empty rather than
        inventing one."""
        tight = StrategyInputs(
            circuit="Tight",
            race_laps=90,
            base_pace_ms=90_000.0,
            pit_loss_ms=22_000.0,
            pit_loss_sd_ms=1_000.0,
            degradation_s_per_lap={"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
            max_stint_laps={"SOFT": 20, "MEDIUM": 20, "HARD": 20},
            sc_probability=0.0,
            median_sc_laps=0,
        )
        assert enumerate_strategies(tight, max_stops=1) == []


class TestValidate:
    def test_accepts_a_legal_strategy(self, inputs):
        assert validate(Strategy((Stint("SOFT", 20), Stint("HARD", 30))), inputs) == []

    def test_rejects_wrong_distance(self, inputs):
        problems = validate(Strategy((Stint("SOFT", 20), Stint("HARD", 25))), inputs)
        assert any("sum to 45" in p for p in problems)

    def test_rejects_single_compound(self, inputs):
        problems = validate(Strategy((Stint("SOFT", 25), Stint("SOFT", 25))), inputs)
        assert any("two different dry compounds" in p for p in problems)

    def test_rejects_stint_beyond_observed_life(self, inputs):
        problems = validate(Strategy((Stint("SOFT", 40), Stint("HARD", 10))), inputs)
        assert any("exceeds the 25-lap maximum" in p for p in problems)

    def test_rejects_no_stop(self, inputs):
        problems = validate(Strategy((Stint("SOFT", 50),)), inputs)
        assert any("at least one stop" in p for p in problems)


def _uniform_inputs(slope: float, pit_loss_s: float = 22.0, race_laps: int = 50):
    """One circuit, one degradation rate on every compound.

    Uniform rates make the optimum analytically predictable, which is what lets
    the threshold below be derived rather than guessed.
    """
    return StrategyInputs(
        circuit="Uniform",
        race_laps=race_laps,
        base_pace_ms=90_000.0,
        pit_loss_ms=pit_loss_s * 1000.0,
        pit_loss_sd_ms=1_000.0,
        degradation_s_per_lap={"SOFT": slope, "MEDIUM": slope, "HARD": slope},
        # Capped at 30 rather than 45: with 45-lap stints allowed, enumerating
        # 4-stop strategies over 50 laps takes ~40s and adds nothing, since the
        # optimum is always a near-even split well inside the cap.
        max_stint_laps={"SOFT": 30, "MEDIUM": 30, "HARD": 30},
        sc_probability=0.0,
        median_sc_laps=0,
    )


class TestStrategyTradeoff:
    """The central trade-off: a stop costs a fixed pit loss and buys a quadratic
    reduction in degradation."""

    def test_extra_stop_must_pay_for_itself(self, inputs):
        """A second stop is NOT automatically better on a worn-tire circuit.

        With this fixture (HARD 0.03 s/lap, pit loss 22s over 50 laps) a second
        stop saves only ~10s of degradation while costing 22s of pit time, so
        one stop wins. Asserting the opposite was this suite's own first bug --
        the intuition "more degradation means more stops" is only true once the
        degradation saved exceeds the pit loss paid.
        """
        best = sim.rank_by_stop_count(inputs, max_stops=3)
        assert best[1].total_ms < best[2].total_ms
        extra_pit = best[2].pit_ms - best[1].pit_ms
        degradation_saved = best[1].degradation_ms - best[2].degradation_ms
        assert degradation_saved < extra_pit

    def test_threshold_matches_the_derivation(self):
        """Derived, then verified by brute force.

        For 50 laps at one rate `s`: a 1-stop splits 25/25 costing
        s*(2 * 25*24/2) = 600s, a 2-stop splits 17/17/16 costing
        s*(136+136+120) = 392s. The second stop therefore saves 208*s seconds
        and costs one pit loss of 22s, so it pays when s > 22/208 = 0.1058.
        """
        threshold = 22.0 / 208.0

        below = sim.rank_by_stop_count(_uniform_inputs(threshold * 0.8), max_stops=2)
        assert below[1].total_ms < below[2].total_ms, "below threshold: 1-stop wins"

        above = sim.rank_by_stop_count(_uniform_inputs(threshold * 1.2), max_stops=2)
        assert above[2].total_ms < above[1].total_ms, "above threshold: 2-stop wins"

    def test_optimal_stop_count_never_falls_as_wear_rises(self):
        """The monotone property, which holds regardless of any threshold."""
        counts = []
        for slope in (0.01, 0.05, 0.10, 0.20, 0.40):
            best = sim.rank_by_stop_count(_uniform_inputs(slope), max_stops=3)
            counts.append(min(best, key=lambda n: best[n].total_ms))
        assert counts == sorted(counts), f"stop count should rise with wear, got {counts}"

    def test_smooth_circuit_wants_fewer_stops(self):
        """Monaco's real behaviour: near-zero degradation means a stop buys
        almost nothing, so the fewest legal stops wins."""
        smooth = StrategyInputs(
            circuit="Smooth",
            race_laps=60,
            base_pace_ms=80_000.0,
            pit_loss_ms=22_000.0,
            pit_loss_sd_ms=1_000.0,
            degradation_s_per_lap={"SOFT": 0.005, "MEDIUM": 0.003, "HARD": 0.002},
            max_stint_laps={"SOFT": 40, "MEDIUM": 45, "HARD": 45},
            sc_probability=0.0,
            median_sc_laps=0,
        )
        best = sim.rank_by_stop_count(smooth, max_stops=3)
        assert best[1].total_ms < best[2].total_ms < best[3].total_ms

    def test_pit_loss_drives_the_stop_count(self, inputs):
        """Same degradation, a far more expensive pit lane -> fewer stops."""
        expensive = StrategyInputs(**{**inputs.__dict__, "pit_loss_ms": 60_000.0})
        assert sim.rank_by_stop_count(expensive, max_stops=3)[1].total_ms < (
            sim.rank_by_stop_count(expensive, max_stops=3)[2].total_ms
        )
