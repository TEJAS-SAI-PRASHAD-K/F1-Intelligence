"""Tests for the Monte Carlo layer.

The properties worth pinning here are mostly about the TECHNIQUE rather than
the numbers: common random numbers, what the safety car can and cannot change,
and that the reported spread is the one that means something.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.strategy import monte_carlo as mc
from app.ml.strategy.enumerate_strategies import Stint, Strategy
from app.ml.strategy.inputs import StrategyInputs


@pytest.fixture
def inputs() -> StrategyInputs:
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


@pytest.fixture
def no_sc_inputs(inputs) -> StrategyInputs:
    return StrategyInputs(**{**inputs.__dict__, "sc_probability": 0.0})


ONE_STOP = Strategy((Stint("SOFT", 20), Stint("HARD", 30)))
TWO_STOP = Strategy((Stint("SOFT", 16), Stint("MEDIUM", 17), Stint("HARD", 17)))


class TestScenarios:
    def test_safety_car_rate_matches_the_input_probability(self, inputs):
        scenarios = mc.draw_scenarios(inputs, 4000, seed=1)
        rate = sum(s.sc_lap is not None for s in scenarios) / len(scenarios)
        assert rate == pytest.approx(inputs.sc_probability, abs=0.03)

    def test_no_safety_car_when_probability_is_zero(self, no_sc_inputs):
        scenarios = mc.draw_scenarios(no_sc_inputs, 200, seed=1)
        assert all(s.sc_lap is None for s in scenarios)

    def test_deployment_clusters_early(self, inputs):
        """Beta(2,3) over the race distance: incidents cluster at the start, from
        first-lap contact and cold tires, without forbidding a late one."""
        scenarios = mc.draw_scenarios(inputs, 4000, seed=1)
        laps = [s.sc_lap for s in scenarios if s.sc_lap is not None]
        assert np.mean(laps) < inputs.race_laps / 2

    def test_scenarios_are_reproducible(self, inputs):
        a = mc.draw_scenarios(inputs, 50, seed=11)
        b = mc.draw_scenarios(inputs, 50, seed=11)
        assert [s.sc_lap for s in a] == [s.sc_lap for s in b]
        assert np.allclose(a[0].lap_noise_ms, b[0].lap_noise_ms)

    def test_pit_multipliers_cannot_make_a_stop_free(self, inputs):
        """Clipped, because an unclipped normal draw can go negative and a stop
        that gives time back is not physical."""
        scenarios = mc.draw_scenarios(inputs, 500, seed=3)
        assert all((s.pit_multipliers > 0).all() for s in scenarios)

    def test_window_spans_the_duration(self, inputs):
        scenario = mc.Scenario(
            sc_lap=10, sc_duration=4,
            pit_multipliers=np.ones(5), lap_noise_ms=np.zeros(50),
        )
        assert scenario.sc_window() == {10, 11, 12, 13}


class TestCommonRandomNumbers:
    """The technique that makes the comparison work."""

    def test_every_strategy_sees_the_same_scenarios(self, inputs):
        scenarios = mc.draw_scenarios(inputs, 300, seed=5)
        a = mc.run_monte_carlo(ONE_STOP, inputs, scenarios)
        b = mc.run_monte_carlo(TWO_STOP, inputs, scenarios)
        assert len(a.total_ms) == len(b.total_ms) == 300

    def test_paired_margin_has_far_less_variance_than_either_strategy(self, inputs):
        """The whole point of pairing, as a test.

        A single strategy's race time is dominated by the safety-car penalty,
        which EVERY strategy pays identically. Subtracting per scenario cancels
        it, so the margin's sd is an order of magnitude smaller -- which is why
        a 2s edge can be resolved from a few thousand runs.
        """
        distributions, _ = mc.compare_strategies(
            [ONE_STOP, TWO_STOP], inputs, n_scenarios=2000, seed=5
        )
        delta = mc.paired_delta(distributions[1], distributions[0])
        assert delta.sd_delta_ms < distributions[0].sd_ms / 5

    def test_win_rates_are_complementary(self, inputs):
        _, wins = mc.compare_strategies([ONE_STOP, TWO_STOP], inputs, n_scenarios=800, seed=5)
        assert wins[0][1] + wins[1][0] == pytest.approx(1.0, abs=0.01)

    def test_a_strategy_cannot_beat_itself(self, inputs):
        _, wins = mc.compare_strategies([ONE_STOP, ONE_STOP], inputs, n_scenarios=200, seed=5)
        assert wins[0][1] == 0.0 and wins[1][0] == 0.0


class TestSafetyCarEffect:
    def test_safety_car_slows_every_strategy(self, inputs, no_sc_inputs):
        """The lap-time penalty is real but shared."""
        with_sc = mc.run_monte_carlo(
            ONE_STOP, inputs, mc.draw_scenarios(inputs, 1500, seed=9)
        )
        without = mc.run_monte_carlo(
            ONE_STOP, no_sc_inputs, mc.draw_scenarios(no_sc_inputs, 1500, seed=9)
        )
        assert with_sc.median_ms > without.median_ms

    def test_a_stop_under_the_safety_car_is_cheaper(self, inputs):
        """The ONLY channel through which a safety car changes the ranking.

        ONE strategy under TWO scenarios, not two strategies under one. The
        first version of this test compared S20/H30 against S10/H40, which
        differ in degradation by 7.05s as well as in safety-car luck, so the
        observed 5.05s gap was 12.10s of pit saving minus 7.05s of degradation
        and matched neither number. Holding the strategy fixed and moving only
        the safety car isolates the effect exactly.
        """
        strategy = Strategy((Stint("SOFT", 20), Stint("HARD", 30)))  # stops on lap 20
        assert strategy.stop_laps == (20,)

        covers_the_stop = mc.Scenario(
            sc_lap=18, sc_duration=5,  # window 18-22, contains lap 20
            pit_multipliers=np.ones(5), lap_noise_ms=np.zeros(50),
        )
        misses_the_stop = mc.Scenario(
            sc_lap=40, sc_duration=5,  # window 40-44, same LENGTH so the
            pit_multipliers=np.ones(5), lap_noise_ms=np.zeros(50),  # shared
        )  # lap-time penalty is identical

        lucky_ms, lucky_flag = mc.simulate_scenario(strategy, inputs, covers_the_stop)
        unlucky_ms, unlucky_flag = mc.simulate_scenario(strategy, inputs, misses_the_stop)

        assert lucky_flag is True
        assert unlucky_flag is False
        # Both scenarios have a 5-lap safety car, so the shared lap-time penalty
        # cancels and the whole difference is the discounted pit stop.
        expected_saving = inputs.pit_loss_ms * (1 - mc.SC_PIT_LOSS_FACTOR)
        assert unlucky_ms - lucky_ms == pytest.approx(expected_saving, abs=1.0)

    def test_more_stops_means_more_safety_car_exposure(self, inputs):
        """More stops are more lottery tickets: each is another chance to land
        under a safety car. It partly offsets the deterministic cost of the
        extra stop, and at Sakhir it recovers a few tenths."""
        scenarios = mc.draw_scenarios(inputs, 3000, seed=13)
        one = mc.run_monte_carlo(ONE_STOP, inputs, scenarios)
        two = mc.run_monte_carlo(TWO_STOP, inputs, scenarios)
        assert two.sc_luck_rate > one.sc_luck_rate

    def test_no_luck_without_a_safety_car(self, no_sc_inputs):
        scenarios = mc.draw_scenarios(no_sc_inputs, 300, seed=13)
        assert mc.run_monte_carlo(TWO_STOP, no_sc_inputs, scenarios).sc_luck_rate == 0.0


class TestSensitivity:
    def test_ranking_survives_the_least_defensible_constant(self, inputs):
        """SC_PIT_LOSS_FACTOR is an approximation, not a measurement, so the
        honest move is to vary it and check the conclusion holds rather than
        defend the value."""
        result = mc.sensitivity_to_sc_factor(
            [ONE_STOP, TWO_STOP], inputs, factors=(0.25, 0.45, 0.65, 1.0), n_scenarios=600
        )
        winners = {
            factor: int(np.argmin(medians)) for factor, medians in result.items()
        }
        assert len(set(winners.values())) == 1, f"ranking flipped across factors: {winners}"


class TestDistributionFields:
    def test_percentiles_are_ordered(self, inputs):
        d = mc.run_monte_carlo(ONE_STOP, inputs, mc.draw_scenarios(inputs, 1000, seed=2))
        assert d.p10_ms < d.median_ms < d.p90_ms

    def test_deterministic_is_recorded_for_comparison(self, inputs):
        """A safety car can only ADD time, so the Monte Carlo median must sit
        at or above the deterministic time."""
        d = mc.run_monte_carlo(ONE_STOP, inputs, mc.draw_scenarios(inputs, 1000, seed=2))
        assert d.median_ms >= d.deterministic_ms

    def test_lap_times_cover_the_race(self, inputs):
        scenario = mc.draw_scenarios(inputs, 1, seed=4)[0]
        times = mc.lap_times_for_scenario(ONE_STOP, inputs, scenario)
        assert len(times) == inputs.race_laps
