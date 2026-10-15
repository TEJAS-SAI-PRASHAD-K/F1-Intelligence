"""Tests for the Phase 5 dashboard endpoints.

These run against the REAL database rather than the usual rolled-back test
transaction, and that is a deliberate exception worth explaining.

Every other suite uses `conftest.client`, whose `get_db` override hands routes a
session bound to `f1_intelligence_test`. That database holds only ORM tables --
`Base.metadata.create_all` does not create views -- so `features.mv_driver_season`
does not exist there, and it would be empty even if it did.

The dashboard's entire job is aggregating 171k real laps. Testing it against an
empty schema would exercise nothing. So `dashboard_client` below overrides
`get_db` with a real-database session instead. Every dashboard endpoint is
read-only, so this cannot leave anything behind.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db.base import SessionLocal


@pytest.fixture
def dashboard_client(monkeypatch):
    """TestClient whose routes get a REAL-database session. Read-only."""
    from fastapi.testclient import TestClient

    from app.db.deps import get_db
    from app.main import app

    def _real_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _real_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def has_views() -> bool:
    try:
        with SessionLocal() as db:
            n = db.execute(text("SELECT count(*) FROM features.mv_driver_season")).scalar()
        return bool(n)
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def _require_views(has_views):
    if not has_views:
        pytest.skip(
            "dashboard views are empty; run: python -m app.ingestion.cli refresh-views"
        )


@pytest.fixture(scope="module")
def latest_season() -> int:
    with SessionLocal() as db:
        return int(
            db.execute(text("SELECT max(season_year) FROM features.mv_driver_season")).scalar()
        )


class TestSeasonDashboard:
    def test_returns_standings_ordered_by_points(self, dashboard_client, latest_season):
        body = dashboard_client.get(f"/api/v1/dashboard/season/{latest_season}").json()
        points = [d["points"] for d in body["drivers"]]
        assert points == sorted(points, reverse=True)
        assert len(body["drivers"]) >= 18

    def test_includes_teams_and_circuits(self, dashboard_client, latest_season):
        body = dashboard_client.get(f"/api/v1/dashboard/season/{latest_season}").json()
        assert len(body["teams"]) >= 8
        assert len(body["circuits"]) >= 10

    def test_champion_has_the_most_wins_or_close_to_it(self, dashboard_client, latest_season):
        """A sanity check against reality: the points leader should be at or
        near the top of the win count too."""
        body = dashboard_client.get(f"/api/v1/dashboard/season/{latest_season}").json()
        leader = body["drivers"][0]
        most_wins = max(d["wins"] for d in body["drivers"])
        assert leader["wins"] >= most_wins - 1

    def test_wins_and_podiums_are_consistent(self, dashboard_client, latest_season):
        """Every win is also a podium, so wins can never exceed podiums."""
        body = dashboard_client.get(f"/api/v1/dashboard/season/{latest_season}").json()
        for d in body["drivers"]:
            assert d["wins"] <= d["podiums"] <= d["points_finishes"] <= d["races"]

    def test_states_the_sprint_points_caveat(self, dashboard_client, latest_season):
        """Sprint sessions were not ingested, so totals are below official
        championship points in sprint years. A number that looks like official
        points but is not has to say so."""
        body = dashboard_client.get(f"/api/v1/dashboard/season/{latest_season}").json()
        assert any("Sprint" in n for n in body["notes"])

    def test_tire_management_is_relative_to_the_field(self, dashboard_client, latest_season):
        """Measured as a driver's slope minus the FIELD MEDIAN in the same race,
        which cancels the fuel effect without needing the fuel model. So values
        straddle zero by construction."""
        body = dashboard_client.get(f"/api/v1/dashboard/season/{latest_season}").json()
        values = [
            d["tire_mgmt_s_per_lap"]
            for d in body["drivers"]
            if d["tire_mgmt_s_per_lap"] is not None
        ]
        assert values, "expected at least one driver with a tire-management figure"
        assert min(values) < 0 < max(values)

    def test_unknown_season_lists_what_is_available(self, dashboard_client):
        response = dashboard_client.get("/api/v1/dashboard/season/1950")
        assert response.status_code == 404
        assert "Available" in response.json()["detail"]


class TestDriverDashboard:
    def test_returns_career_summary(self, dashboard_client):
        body = dashboard_client.get("/api/v1/dashboard/driver/VER").json()
        assert body["driver"] == "VER"
        assert body["career_races"] > 50
        assert body["career_wins"] > 20
        assert len(body["seasons"]) >= 4

    def test_seasons_are_chronological(self, dashboard_client):
        body = dashboard_client.get("/api/v1/dashboard/driver/VER").json()
        years = [s["season_year"] for s in body["seasons"]]
        assert years == sorted(years)

    def test_best_circuits_beat_worst_circuits(self, dashboard_client):
        body = dashboard_client.get("/api/v1/dashboard/driver/VER").json()
        if body["best_circuits"] and body["worst_circuits"]:
            assert (
                body["best_circuits"][0]["mean_finish"]
                < body["worst_circuits"][0]["mean_finish"]
            )

    def test_case_insensitive(self, dashboard_client):
        assert dashboard_client.get("/api/v1/dashboard/driver/ver").status_code == 200

    def test_unknown_driver_is_404(self, dashboard_client):
        assert dashboard_client.get("/api/v1/dashboard/driver/ZZZ").status_code == 404


class TestRaceDashboard:
    @pytest.fixture(scope="class")
    def monza_event_id(self) -> int:
        with SessionLocal() as db:
            row = db.execute(
                text(
                    "SELECT id FROM raw.events WHERE season_year = 2024 AND round = 16"
                )
            ).scalar()
        if row is None:
            pytest.skip("2024 Monza not ingested")
        return int(row)

    def test_returns_results_stints_and_metadata(self, dashboard_client, monza_event_id):
        body = dashboard_client.get(f"/api/v1/dashboard/race/{monza_event_id}").json()
        assert body["circuit"] == "Monza"
        assert body["total_laps"] == 53
        assert len(body["results"]) >= 18
        assert len(body["stints"]) >= 20

    def test_results_ordered_by_finishing_position(self, dashboard_client, monza_event_id):
        body = dashboard_client.get(f"/api/v1/dashboard/race/{monza_event_id}").json()
        positions = [r["position"] for r in body["results"] if r["position"] is not None]
        assert positions == sorted(positions)

    def test_places_gained_is_grid_minus_finish(self, dashboard_client, monza_event_id):
        body = dashboard_client.get(f"/api/v1/dashboard/race/{monza_event_id}").json()
        for r in body["results"]:
            if r["places_gained"] is not None:
                assert r["places_gained"] == pytest.approx(
                    r["grid_position"] - r["position"]
                )

    def test_stint_degradation_is_fuel_corrected(self, dashboard_client, monza_event_id):
        """Uncorrected slopes come out near zero or negative, because fuel burn
        (~0.055 s/lap of gain) nearly cancels tire wear. The dashboard reuses
        Phase 2's Stage A rather than recomputing a raw slope in SQL, so most
        measurable stints should show POSITIVE degradation."""
        body = dashboard_client.get(f"/api/v1/dashboard/race/{monza_event_id}").json()
        slopes = [
            s["degradation_s_per_lap"]
            for s in body["stints"]
            if s["degradation_s_per_lap"] is not None
        ]
        assert slopes, "expected at least one stint with a measurable slope"
        assert sum(s > 0 for s in slopes) / len(slopes) > 0.6

    def test_stint_laps_are_consistent(self, dashboard_client, monza_event_id):
        body = dashboard_client.get(f"/api/v1/dashboard/race/{monza_event_id}").json()
        for s in body["stints"]:
            assert s["lap_start"] <= s["lap_end"]
            assert s["laps"] <= s["lap_end"] - s["lap_start"] + 1

    def test_unknown_event_is_404(self, dashboard_client):
        assert dashboard_client.get("/api/v1/dashboard/race/999999").status_code == 404


class TestModelsDashboard:
    def test_lists_registered_models(self, dashboard_client):
        """The point of a registry: answer 'what is deployed, trained on what,
        and how good was it' without reading code."""
        body = dashboard_client.get("/api/v1/dashboard/models").json()
        assert isinstance(body["models"], list)
        assert body["phases"]

    def test_headline_metrics_are_rendered_in_their_own_units(self, dashboard_client):
        """Lap-time MAE in seconds and finishing-position MAE are not comparable
        numbers, so each model renders its own."""
        body = dashboard_client.get("/api/v1/dashboard/models").json()
        rendered = [m["headline_metric"] for m in body["models"] if m["headline_metric"]]
        if rendered:
            assert any("MAE" in text for text in rendered)
