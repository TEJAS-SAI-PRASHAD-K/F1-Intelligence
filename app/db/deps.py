"""FastAPI dependency for database sessions."""

from collections.abc import Iterator

from sqlalchemy.orm import Session

from app.db.base import SessionLocal


def get_db() -> Iterator[Session]:
    """Request-scoped session. FastAPI closes it when the request ends."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
