"""Tests for the Phase 3 strategy endpoints.

These run against the real ingested data rather than fixtures, because the
endpoints' whole job is to combine measured inputs -- pit loss, degradation,
safety-car rates -- and a synthetic circuit would test none of that.
"""

from __future__ import annotations

import pytest

from app.api.routes import strategy as strategy_route
from app.db.base import engine
from app.ml.strategy import inputs as strategy_inputs


@pytest.fixture(scope="module")
def has_data() -> bool:
    """Skip if the strategy views are empty, so a fresh clone fails clearly."""
    try:
        return len(strategy_inputs.available_circuits(engine)) > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def _require_data(has_data):
    if not has_data:
        pytest.skip("no ingested race data; run the ingestion CLI first")


class TestInputsEndpoint:
    def test_returns_measured_inputs(self, client):
        body = client.get("/api/v1/strategy/inputs/Monza").json()
        assert body["circuit"] == "Monza"
        assert 40 < body["race_laps"] < 80
        # Monza race laps are ~83-85s.
        assert 75 < body["base_pace_s"] < 95
        # Modern F1 pit loss sits in the high teens to mid thirties.
        assert 15 < body["pit_loss_s"] < 40
        assert set(body["degradation_s_per_lap"]) == {"SOFT", "MEDIUM", "HARD"}
        assert 0.0 <= body["sc_probability"] <= 1.0

    def test_unknown_circuit_lists_the_valid_ones(self, client):
        """A 404 that tells you what IS available beats one that only says no."""
        response = client.get("/api/v1/strategy/inputs/Nürburgring-Nordschleife")
        assert response.status_code == 404
        assert "Available circuits" in response.json()["detail"]

    def test_degradation_is_never_negative(self, client):
        """A negative measured rate means track evolution outran tire wear in
        that sample. Left alone it would make the optimiser want infinitely
        long stints, so build_inputs clamps it at zero."""
        for circuit in ("Monza", "Monaco", "Sakhir"):
            body = client.get(f"/api/v1/strategy/inputs/{circuit}").json()
            assert all(v >= 0 for v in body["degradation_s_per_lap"].values())


class TestSimulateEndpoint:
    def test_deterministic_only(self, client):
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Monza", "n_scenarios": 0, "max_stops": 2},
        ).json()
        assert body["n_scenarios"] == 0
        assert body["n_candidates_evaluated"] > 1000
        assert len(body["strategies"]) >= 2
        assert body["strategies"][0]["mc_median_s"] is None

    def test_best_strategy_has_zero_delta(self, client):
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Monza", "n_scenarios": 0, "max_stops": 2},
        ).json()
        assert min(s["deterministic_delta_s"] for s in body["strategies"]) == 0.0

    def test_every_strategy_covers_the_race_distance(self, client):
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Sakhir", "n_scenarios": 0, "max_stops": 3},
        ).json()
        race_laps = body["inputs"]["race_laps"]
        for s in body["strategies"]:
            assert sum(st["laps"] for st in s["stints"]) == race_laps

    def test_two_compound_rule_holds_in_every_result(self, client):
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Sakhir", "n_scenarios": 0, "max_stops": 3},
        ).json()
        for s in body["strategies"]:
            assert len({st["compound"] for st in s["stints"]}) >= 2

    def test_per_stop_count_returns_one_of_each(self, client):
        """A flat top-k is dominated by variants of one stop count differing by
        under a second, which hides the decision that matters."""
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Sakhir", "n_scenarios": 0, "max_stops": 3,
                  "per_stop_count": True},
        ).json()
        counts = [s["n_stops"] for s in body["strategies"]]
        assert sorted(counts) == [1, 2, 3]

    def test_monte_carlo_fields_appear(self, client):
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Sakhir", "n_scenarios": 400, "max_stops": 2},
        ).json()
        for s in body["strategies"]:
            assert s["mc_median_s"] is not None
            assert s["mc_p10_s"] <= s["mc_median_s"] <= s["mc_p90_s"]
            assert 0.0 <= s["sc_luck_rate"] <= 1.0

    def test_margin_sd_is_much_smaller_than_strategy_sd(self, client):
        """The payoff of common random numbers, visible through the API.

        A single strategy's sd is dominated by the safety-car penalty that every
        strategy pays. The paired margin cancels it, so margin_sd_s is the
        number that tells you whether a gap is real.
        """
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Sakhir", "n_scenarios": 1500, "max_stops": 3},
        ).json()
        others = [s for s in body["strategies"] if s["margin_sd_s"] is not None]
        assert others, "expected at least one non-best strategy"
        for s in others:
            assert s["margin_sd_s"] < s["mc_sd_s"] / 3

    def test_results_are_reproducible_for_a_seed(self, client):
        payload = {"circuit": "Monza", "n_scenarios": 300, "max_stops": 2, "seed": 99}
        a = client.post("/api/v1/strategy/simulate", json=payload).json()
        b = client.post("/api/v1/strategy/simulate", json=payload).json()
        assert [s["mc_median_s"] for s in a["strategies"]] == [
            s["mc_median_s"] for s in b["strategies"]
        ]

    def test_limitations_are_always_stated(self, client):
        """Scope limits, not disclaimers: a user who does not know this is a
        single-car model will over-trust the ranking."""
        body = client.post(
            "/api/v1/strategy/simulate", json={"circuit": "Monza", "n_scenarios": 0}
        ).json()
        assert any("Single-car" in line for line in body["limitations"])

    @pytest.mark.parametrize(
        "payload",
        [
            {"circuit": "Monza", "max_stops": 0},
            {"circuit": "Monza", "max_stops": 9},
            {"circuit": "Monza", "n_scenarios": -1},
            {"circuit": "Monza", "race_laps": 5},
        ],
    )
    def test_out_of_range_input_is_rejected(self, client, payload):
        assert client.post("/api/v1/strategy/simulate", json=payload).status_code == 422


class TestRealWorldAgreement:
    """The strongest available check: does it reproduce known race strategies?"""

    def test_monza_favours_one_stop(self, client):
        """Monza 2024 was won on a one-stop, and that was the story of the race:
        low degradation plus a costly pit lane."""
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Monza", "n_scenarios": 0, "max_stops": 3},
        ).json()
        best = min(body["strategies"], key=lambda s: s["deterministic_delta_s"])
        assert best["n_stops"] == 1

    def test_sakhir_favours_two_stops(self, client):
        """Bahrain is the most abrasive surface on the calendar and is
        classically a two-stop race."""
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Sakhir", "n_scenarios": 0, "max_stops": 3},
        ).json()
        best = min(body["strategies"], key=lambda s: s["deterministic_delta_s"])
        assert best["n_stops"] == 2

    def test_monaco_favours_one_stop(self, client):
        """Monaco is low-energy: tires barely degrade, so a stop buys almost
        nothing."""
        body = client.post(
            "/api/v1/strategy/simulate",
            json={"circuit": "Monaco", "n_scenarios": 0, "max_stops": 3},
        ).json()
        best = min(body["strategies"], key=lambda s: s["deterministic_delta_s"])
        assert best["n_stops"] == 1


class TestCompareEndpoint:
    def test_compares_two_legal_strategies(self, client):
        laps = client.get("/api/v1/strategy/inputs/Monza").json()["race_laps"]
        body = client.post(
            "/api/v1/strategy/compare",
            json={
                "circuit": "Monza",
                "n_scenarios": 400,
                "strategies": [
                    [{"compound": "SOFT", "laps": 25}, {"compound": "HARD", "laps": laps - 25}],
                    [
                        {"compound": "SOFT", "laps": 17},
                        {"compound": "MEDIUM", "laps": 18},
                        {"compound": "HARD", "laps": laps - 35},
                    ],
                ],
            },
        ).json()
        assert len(body["strategies"]) == 2
        assert body["win_matrix"][0][1] + body["win_matrix"][1][0] == pytest.approx(1.0, abs=0.01)

    def test_illegal_strategy_is_rejected_with_reasons(self, client):
        """Not silently adjusted. A strategy whose stints do not sum to the race
        distance is not something to quietly fix for the caller."""
        laps = client.get("/api/v1/strategy/inputs/Monza").json()["race_laps"]
        response = client.post(
            "/api/v1/strategy/compare",
            json={
                "circuit": "Monza",
                "n_scenarios": 100,
                "strategies": [
                    [{"compound": "SOFT", "laps": 25}, {"compound": "HARD", "laps": laps - 25}],
                    [{"compound": "SOFT", "laps": 10}, {"compound": "HARD", "laps": 10}],
                ],
            },
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "1" in detail["invalid"]
        assert any("sum to 20" in p for p in detail["invalid"]["1"])

    def test_single_compound_strategy_is_rejected(self, client):
        laps = client.get("/api/v1/strategy/inputs/Monza").json()["race_laps"]
        response = client.post(
            "/api/v1/strategy/compare",
            json={
                "circuit": "Monza",
                "n_scenarios": 100,
                "strategies": [
                    [{"compound": "SOFT", "laps": 25}, {"compound": "HARD", "laps": laps - 25}],
                    [{"compound": "SOFT", "laps": 25}, {"compound": "SOFT", "laps": laps - 25}],
                ],
            },
        )
        assert response.status_code == 422
        assert any(
            "two different dry compounds" in p
            for p in response.json()["detail"]["invalid"]["1"]
        )

    def test_needs_at_least_two_strategies(self, client):
        assert (
            client.post(
                "/api/v1/strategy/compare",
                json={"circuit": "Monza", "strategies": [[{"compound": "SOFT", "laps": 53}]]},
            ).status_code
            == 422
        )


class TestCache:
    def test_cache_can_be_cleared(self):
        strategy_route.clear_cache()
        assert strategy_route._laps_cache["frame"] is None
