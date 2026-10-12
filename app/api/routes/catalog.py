"""Read endpoints for the ingested catalogue: seasons, events, drivers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import joinedload

from app.db.deps import get_db
from app.db.models import Driver, Event, Season, Session
from app.schemas import DriverOut, EventOut, SeasonOut, SessionOut

router = APIRouter(tags=["catalog"])


@router.get("/seasons", response_model=list[SeasonOut])
def list_seasons(db: OrmSession = Depends(get_db)):
    return db.scalars(select(Season).order_by(Season.year.desc())).all()


@router.get("/seasons/{year}/events", response_model=list[EventOut])
def list_events(year: int, db: OrmSession = Depends(get_db)):
    events = db.scalars(
        select(Event)
        .options(joinedload(Event.circuit))
        .where(Event.season_year == year)
        .order_by(Event.round)
    ).all()
    if not events:
        raise HTTPException(404, f"No events ingested for {year}")
    return events


@router.get("/events/{event_id}/sessions", response_model=list[SessionOut])
def list_sessions(event_id: int, db: OrmSession = Depends(get_db)):
    sessions = db.scalars(
        select(Session).where(Session.event_id == event_id).order_by(Session.session_date)
    ).all()
    if not sessions:
        raise HTTPException(404, f"No sessions for event {event_id}")
    return sessions


@router.get("/drivers", response_model=list[DriverOut])
def list_drivers(db: OrmSession = Depends(get_db)):
    return db.scalars(select(Driver).order_by(Driver.abbreviation)).all()
