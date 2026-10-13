"""Tests for the Phase 2 tire endpoints.

A real (tiny) XGBoost model is trained and registered so the endpoints are
exercised end to end, including the JSON serialisation path and the
unseen-category fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml import registry
from app.ml.tire import dataset as ds
from app.ml.tire import predict as tire_predict
from app.ml.tire.train import MODEL_NAME, build_stage_c_xgboost


@pytest.fixture
def registered_model(db, tmp_path, monkeypatch):
    """Train and register a small real model on synthetic data.

    Degradation is built in deliberately (0.08 s/lap on SOFT, 0.03 on HARD) so
    the endpoint's output can be checked for the right SHAPE rather than just
    for a 200.
    """
    monkeypatch.setattr(registry, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)

    rng = np.random.default_rng(0)
    rows = []
    for circuit in ("Monza", "Monaco"):
        for compound, rate in (("SOFT", 80.0), ("HARD", 30.0)):
            for stint in (1, 2):
                for life in range(1, 26):
                    rows.append(
                        {
                            "tyre_life": float(life),
                            "tyre_life_sq": float(life) ** 2,
                            "lap_number": float(stint * 25 + life),
                            "race_laps": 53.0,
                            "stint_number": float(stint),
                            "air_temp": 25.0,
                            "track_temp": 38.0,
                            "is_fresh_tyre_int": 1,
                            "compound": compound,
                            "circuit": circuit,
                            "driver": "VER",
                            "team": "Red Bull Racing",
                            ds.TARGET: rate * life + rng.normal(0, 20),
                        }
                    )
    frame = pd.DataFrame(rows)
    for col in ds.CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype("category")

    model = build_stage_c_xgboost(ds.FEATURES, early_stopping=False)
    model.set_params(n_estimators=40)
    model.fit(ds.feature_matrix(frame), ds.target_vector(frame))

    vocab = {
        "compound": ["HARD", "SOFT"],
        "circuit": ["Monaco", "Monza"],
        "driver": ["VER"],
        "team": ["Red Bull Racing"],
    }
    registry.save_model(
        db,
        name=MODEL_NAME,
        version="test_v1",
        backend=registry.BACKEND_XGBOOST,
        model=model,
        categories=vocab,
        feature_names=ds.FEATURES,
        metrics={
            "test": {"mae_ms": 220.0, "rmse_ms": 310.0, "r2": 0.42, "n": 1000},
            "baseline_test_mae_ms": 500.0,
            "lift_vs_baseline": 0.56,
            "control_histgbr_test_mae_ms": 228.0,
        },
        train_season_min=2022,
        train_season_max=2023,
        n_train_rows=1000,
        notes="holdout season 2024",
    )
    db.flush()
    tire_predict.clear_cache()
    yield
    tire_predict.clear_cache()


class TestDegradationEndpointWithoutModel:
    def test_returns_503_when_nothing_is_registered(self, client, monkeypatch):
        """A missing model is an operational state, not a crash. 503 plus the
        command to fix it is more useful than a 500."""
        tire_predict.clear_cache()
        response = client.post(
            "/api/v1/tire/degradation", json={"circuit": "Monza", "compound": "SOFT"}
        )
        assert response.status_code == 503
        assert "python -m app.ml.tire.train" in response.json()["detail"]


class TestDegradationEndpoint:
    def test_returns_a_curve(self, client, registered_model):
        response = client.post(
            "/api/v1/tire/degradation",
            json={"circuit": "Monza", "compound": "SOFT", "max_tyre_life": 20},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["points"]) == 20
        assert body["model_backend"] == "xgboost"
        assert body["model_version"] == "test_v1"

    def test_fresh_tyre_is_the_zero_reference(self, client, registered_model):
        """`delta_vs_fresh` is defined as prediction(L) - prediction(1), so the
        first point must be exactly zero. That is what makes the number
        interpretable without knowing the stint median."""
        body = client.post(
            "/api/v1/tire/degradation",
            json={"circuit": "Monza", "compound": "SOFT", "max_tyre_life": 15},
        ).json()
        assert body["points"][0]["delta_vs_fresh_ms"] == 0.0
        assert body["points"][0]["delta_vs_fresh_s"] == 0.0

    def test_curve_never_improves_with_age(self, client, registered_model):
        """The monotonic constraint, observed through the API. A tire getting
        faster as it ages is physically impossible, and `monotone_constraints`
        in XGBoost enforces it rather than hoping the data teaches it."""
        body = client.post(
            "/api/v1/tire/degradation",
            json={"circuit": "Monza", "compound": "SOFT", "max_tyre_life": 25},
        ).json()
        deltas = [p["delta_vs_fresh_ms"] for p in body["points"]]
        assert all(b >= a - 1e-6 for a, b in zip(deltas, deltas[1:]))  # noqa: B905

    def test_degradation_rate_is_positive(self, client, registered_model):
        body = client.post(
            "/api/v1/tire/degradation",
            json={"circuit": "Monza", "compound": "SOFT", "max_tyre_life": 25},
        ).json()
        assert body["degradation_rate_s_per_lap"] > 0
        assert body["total_loss_s"] > 0

    def test_softer_compound_degrades_faster(self, client, registered_model):
        """Built into the fixture (SOFT 80 ms/lap vs HARD 30), so the model
        must recover the ordering. Checks the compound feature is actually
        used rather than ignored."""
        def rate(compound: str) -> float:
            return client.post(
                "/api/v1/tire/degradation",
                json={"circuit": "Monza", "compound": compound, "max_tyre_life": 25},
            ).json()["degradation_rate_s_per_lap"]

        assert rate("SOFT") > rate("HARD")

    def test_compound_is_upper_cased(self, client, registered_model):
        response = client.post(
            "/api/v1/tire/degradation", json={"circuit": "Monza", "compound": "soft"}
        )
        assert response.status_code == 200
        assert response.json()["compound"] == "SOFT"

    def test_unseen_circuit_degrades_gracefully_and_is_reported(
        self, client, registered_model
    ):
        """The second XGBoost trap. An unseen category is a HARD ERROR in
        XGBoost, so the pinned vocabulary maps it to NaN instead. The response
        must still succeed AND say which fields fell back, so a caller is not
        silently handed a global-average curve."""
        body = client.post(
            "/api/v1/tire/degradation",
            json={"circuit": "Las Vegas", "compound": "SOFT", "max_tyre_life": 10},
        ).json()
        assert body["unseen_categories"] == {"circuit": 10}
        assert len(body["points"]) == 10

    def test_known_circuit_reports_no_unseen_categories(self, client, registered_model):
        body = client.post(
            "/api/v1/tire/degradation",
            json={"circuit": "Monza", "compound": "SOFT", "driver": "VER"},
        ).json()
        assert body["unseen_categories"] == {}

    @pytest.mark.parametrize(
        "payload",
        [
            {"circuit": "Monza", "compound": "SOFT", "max_tyre_life": 1},
            {"circuit": "Monza", "compound": "SOFT", "max_tyre_life": 999},
            {"circuit": "Monza", "compound": "SOFT", "race_laps": 5},
        ],
    )
    def test_out_of_range_input_is_rejected(self, client, registered_model, payload):
        assert client.post("/api/v1/tire/degradation", json=payload).status_code == 422


class TestModelInfoEndpoint:
    def test_reports_provenance_and_baseline(self, client, registered_model):
        """An MAE on its own is unreadable. Reporting the baseline next to it is
        what makes 0.22s mean something."""
        body = client.get("/api/v1/tire/model-info").json()
        assert body["backend"] == "xgboost"
        assert body["version"] == "test_v1"
        assert body["train_season_min"] == 2022
        assert body["holdout"]["mae_s"] == pytest.approx(0.22)
        assert body["baseline_mae_s"] == pytest.approx(0.5)
        assert body["lift_vs_baseline"] == pytest.approx(0.56)

    def test_reports_the_histgbr_control(self, client, registered_model):
        """Keeping the control model's number visible is how you notice a
        tuned XGBoost quietly losing to a near-default sklearn model."""
        body = client.get("/api/v1/tire/model-info").json()
        assert body["control_histgbr_mae_s"] == pytest.approx(0.228)

    def test_503_without_a_model(self, client):
        assert client.get("/api/v1/tire/model-info").status_code == 503


class TestObservedStintsEndpoint:
    def test_404_for_unknown_session(self, client, seeded):
        assert client.get("/api/v1/tire/stints/999999").status_code == 404

    def test_returns_a_list_for_a_known_session(self, client, seeded):
        """The seeded fixture's laps are MEDIUM/HARD with few green laps, so
        an empty list is a valid answer -- what matters is that it does not
        error."""
        response = client.get(f"/api/v1/tire/stints/{seeded['session_id']}")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
