"""Test fixtures.

Tests run against a real Postgres database (`f1_intelligence_test`), not
SQLite. That is a deliberate consequence of the single-database decision: the
tests exercise the same engine, the same array types and the same ON CONFLICT
semantics as production, so a test passing here means something.

Each test runs inside a transaction that is rolled back afterwards, so the
suite is fast and leaves no residue.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.db.base import SCHEMA_FEATURES, SCHEMA_ML, SCHEMA_RAW, Base
from app.db.deps import get_db
from app.db.models import Driver, Event, Lap, Season, Session, SessionResult, Team
from app.main import app


@pytest.fixture(scope="session")
def engine():
    eng = create_engine(get_settings().test_database_url, future=True)
    with eng.begin() as conn:
        for schema in (SCHEMA_RAW, SCHEMA_FEATURES, SCHEMA_ML):
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine) -> OrmSession:
    """Session bound to a transaction that is always rolled back."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db) -> TestClient:
    """TestClient wired to the rolled-back session."""
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def seeded(db) -> dict[str, int]:
    """A minimal but realistic session: 2 drivers, 2 stints, a safety car.

    Lap times loosely mimic Monza: ~84s, degrading within a stint.
    """
    db.add(Season(year=2024))
    db.flush()

    event = Event(season_year=2024, round=16, event_name="Italian Grand Prix",
                  event_date=dt.date(2024, 9, 1), event_format="conventional")
    db.add(event)
    db.flush()

    session = Session(event_id=event.id, session_type="R",
                      session_date=dt.datetime(2024, 9, 1, 13, 0, tzinfo=dt.UTC),
                      total_laps=53)
    db.add(session)

    team = Team(name="Ferrari", season_year=2024)
    ver = Driver(abbreviation="VER", driver_number=1, full_name="Max Verstappen")
    lec = Driver(abbreviation="LEC", driver_number=16, full_name="Charles Leclerc")
    db.add_all([team, ver, lec])
    db.flush()

    laps: list[Lap] = []
    for driver in (ver, lec):
        for lap_number in range(1, 11):
            stint = 1 if lap_number <= 5 else 2
            tyre_life = lap_number if stint == 1 else lap_number - 5
            # Lap 7 ran under safety car -- must be excluded from pace stats.
            status = [1, 4] if lap_number == 7 else [1]
            laps.append(
                Lap(
                    session_id=session.id,
                    driver_id=driver.id,
                    team_id=team.id,
                    lap_number=lap_number,
                    lap_time_ms=84_000 + tyre_life * 100 + (20_000 if lap_number == 7 else 0),
                    compound="MEDIUM" if stint == 1 else "HARD",
                    tyre_life=float(tyre_life),
                    is_fresh_tyre=True,
                    stint_number=stint,
                    track_status=status,
                    position=1.0 if driver is lec else 2.0,
                    is_accurate=True,
                )
            )
    db.add_all(laps)

    db.add_all([
        SessionResult(session_id=session.id, driver_id=lec.id, team_id=team.id,
                      position=1.0, grid_position=4.0, status="Finished", points=25.0),
        SessionResult(session_id=session.id, driver_id=ver.id, team_id=team.id,
                      position=6.0, grid_position=7.0, status="Finished", points=8.0),
    ])
    db.flush()

    return {"event_id": event.id, "session_id": session.id,
            "ver_id": ver.id, "lec_id": lec.id}
