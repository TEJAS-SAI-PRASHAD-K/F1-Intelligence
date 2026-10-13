"""FastAPI application.

Interactive docs at http://localhost:8000/docs once running.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.routes import catalog, health, sessions, tire
from app.config import get_settings

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="F1 Intelligence API",
        version="0.1.0",
        description=(
            "Backend for F1 race prediction, tire degradation modeling and pit "
            "strategy simulation.\n\n"
            "**Phase 1 (current): data foundation.** Read-only access to ingested "
            "FastF1 timing data.\n\n"
            "Two things to know when reading lap data:\n"
            "- `track_status` is an array of flags, not a number. `[1]` is clear; "
            "`[1,4]` means the lap was partly clear and partly under safety car.\n"
            "- `compound` does not distinguish C1-C5. SOFT/MEDIUM/HARD are relative "
            "to each weekend's allocation, so a Monza soft and a Monaco soft are "
            "different rubber."
        ),
    )

    app.include_router(health.router)
    app.include_router(catalog.router, prefix=API_PREFIX)
    app.include_router(sessions.router, prefix=API_PREFIX)
    app.include_router(tire.router, prefix=API_PREFIX)
    return app


app = create_app()
