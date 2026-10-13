"""Serve tire-degradation predictions from the registered model.

The model predicts a lap's deviation from its own stint's fuel-corrected median,
which is the right TRAINING target (it removes driver, car and circuit) but an
awkward thing to hand a user. "This lap is 240 ms off its stint median" only
means something if you know the median.

So this module reframes the output around a reference every user already
understands -- a fresh tire:

    delta_vs_fresh_ms(L) = prediction(L) - prediction(1)

That answers the question actually being asked: how much slower is a tire at L
laps old than it was new? Phase 3's simulator needs exactly this shape, since it
adds a degradation penalty on top of a base pace.

WHY `lap_number` IS HELD FIXED ALONG THE CURVE
----------------------------------------------
`lap_number` is a real feature (it is the 3rd most important, and dropping it
costs 0.6% MAE) because it absorbs residual fuel and track-evolution effects the
physics correction in app/ml/fuel.py did not fully remove.

But it must be held CONSTANT while sweeping `tyre_life`. Advancing both together
-- the obvious thing to do, since that is what a real stint does -- produced a
curve that went NEGATIVE at 5 laps of tire age: the monotonic constraint binds
`tyre_life` *holding other features fixed*, so an unconstrained `lap_number`
moving at the same time can pull the prediction back down. The result looked like
a tire getting faster as it aged.

Holding it fixed gives the PARTIAL effect of tire age, which is what
"degradation curve" means, is monotonic by construction, and is what Phase 3
wants -- the simulator applies its own fuel model, so a curve that already
contained fuel effects would double-count them.

The cost is that the ends of the curve sit slightly outside the training
distribution (25 laps of tire age never occurs at a mid-stint lap number). The
reference defaults to the stint's midpoint to keep that extrapolation balanced
rather than one-sided.

The loaded ARTIFACT is cached per process, keyed by (name, version). Reading an
XGBoost booster back from JSON costs tens of milliseconds -- small, but pointless
to repeat on every request. The registry ROW is always read fresh through the
caller's session, so a retrain becomes visible without a restart and callers can
supply their own transaction.

`db` is an optional parameter rather than something this module creates. A
FastAPI route already holds a request-scoped session, so opening a second one
would be wasteful; and code that manufactures its own session cannot be tested
inside a transaction that rolls back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session as OrmSession

from app.db.base import SessionLocal
from app.ml import registry
from app.ml.categories import apply_categories
from app.ml.tire import dataset as ds
from app.ml.tire.train import MODEL_NAME

# Reference values used when a caller does not supply them. Each is a median
# over the training data rather than a guess.
DEFAULT_RACE_LAPS = 57
DEFAULT_AIR_TEMP = 25.0
DEFAULT_TRACK_TEMP = 38.0


@dataclass
class DegradationCurve:
    circuit: str
    compound: str
    tyre_life: list[float]
    delta_vs_fresh_ms: list[float]
    delta_vs_stint_median_ms: list[float]
    degradation_rate_s_per_lap: float
    model_version: str
    model_backend: str
    unseen_categories: dict[str, int]


# (name, version) -> LoadedModel. Not an lru_cache because the lookup needs a
# Session, which is not hashable.
_ARTIFACT_CACHE: dict[tuple[str, str], registry.LoadedModel] = {}


def _load(name: str, db: OrmSession | None = None) -> registry.LoadedModel:
    """Resolve the active model, reusing a cached artifact when the version matches.

    The registry row is read every call so a retrain is picked up without a
    restart; only the heavier artifact load is cached.
    """
    if db is not None:
        return _load_with_session(name, db)
    with SessionLocal() as own_db:
        return _load_with_session(name, own_db)


def _load_with_session(name: str, db: OrmSession) -> registry.LoadedModel:
    from sqlalchemy import select

    from app.db.models import ModelVersion

    version = db.execute(
        select(ModelVersion.version).where(
            ModelVersion.name == name, ModelVersion.is_active.is_(True)
        )
    ).scalar_one_or_none()
    if version is None:
        raise LookupError(f"no active model registered as {name!r}")

    cached = _ARTIFACT_CACHE.get((name, version))
    if cached is not None:
        return cached

    loaded = registry.load_model(db, name, version=version)
    _ARTIFACT_CACHE[(name, version)] = loaded
    return loaded


def clear_cache() -> None:
    """Drop cached artifacts. Call after retraining within a live process."""
    _ARTIFACT_CACHE.clear()


def predict_curve(
    circuit: str,
    compound: str,
    *,
    max_tyre_life: int = 30,
    race_laps: int = DEFAULT_RACE_LAPS,
    stint_start_lap: int = 1,
    air_temp: float | None = DEFAULT_AIR_TEMP,
    track_temp: float | None = DEFAULT_TRACK_TEMP,
    driver: str | None = None,
    team: str | None = None,
    stint_number: int = 1,
    is_fresh_tyre: bool = True,
    reference_lap: int | None = None,
    model_name: str = MODEL_NAME,
    db: OrmSession | None = None,
) -> DegradationCurve:
    """Predicted degradation curve over tire age.

    `lap_number` is pinned at `reference_lap` for every point on the curve --
    see the module docstring for why sweeping it alongside `tyre_life` produces
    a non-monotonic curve. It defaults to the stint's midpoint
    (`stint_start_lap + max_tyre_life // 2`), which keeps the unavoidable
    extrapolation balanced across the curve instead of piling it at one end.

    Unseen categories (a new circuit, a rookie driver) are mapped to NaN rather
    than raising, and the count is returned so a caller can warn. XGBoost treats
    an unknown category as a hard error otherwise -- see app/ml/categories.py.
    """
    loaded = _load(model_name, db=db)

    tyre_life = np.arange(1, max_tyre_life + 1, dtype=float)
    if reference_lap is None:
        reference_lap = stint_start_lap + max_tyre_life // 2

    frame = pd.DataFrame(
        {
            "tyre_life": tyre_life,
            "tyre_life_sq": tyre_life**2,
            # Constant, NOT stint_start_lap + tyre_life - 1. See the docstring.
            "lap_number": float(reference_lap),
            "race_laps": float(race_laps),
            "stint_number": float(stint_number),
            "air_temp": air_temp if air_temp is not None else np.nan,
            "track_temp": track_temp if track_temp is not None else np.nan,
            "is_fresh_tyre_int": int(bool(is_fresh_tyre)),
            "compound": compound,
            "circuit": circuit,
            "driver": driver,
            "team": team,
        }
    )

    frame, unseen = apply_categories(frame, loaded.categories)

    # Reindex to the saved training order. A reordered frame yields plausible,
    # wrong numbers rather than an error, so this is not optional.
    X = frame[loaded.feature_names]
    predicted = np.asarray(loaded.model.predict(X), dtype=float)

    delta_vs_fresh = predicted - predicted[0]

    # Slope of a straight line through the curve: one interpretable number
    # summarising it, directly comparable to Stage A's per-stint slopes.
    rate_s_per_lap = (
        float(np.polyfit(tyre_life, delta_vs_fresh / 1000.0, 1)[0])
        if len(tyre_life) > 1
        else 0.0
    )

    return DegradationCurve(
        circuit=circuit,
        compound=compound,
        tyre_life=tyre_life.tolist(),
        delta_vs_fresh_ms=[round(v, 1) for v in delta_vs_fresh.tolist()],
        delta_vs_stint_median_ms=[round(v, 1) for v in predicted.tolist()],
        degradation_rate_s_per_lap=round(rate_s_per_lap, 5),
        model_version=loaded.version,
        model_backend=loaded.backend,
        unseen_categories={k: v for k, v in unseen.items() if v},
    )


def observed_stint_fits(engine, session_id: int) -> pd.DataFrame:
    """Stage A fits for one session's stints, straight from the data.

    This is the empirical counterpart to `predict_curve`: what the stints in
    that race actually did, rather than what the model expects. Comparing the
    two is how you sanity-check a prediction against reality.
    """
    from app.ml.tire import stage_a

    df = ds.build_dataset(engine)
    subset = df[df["session_id"] == session_id]
    if subset.empty:
        return pd.DataFrame()
    return stage_a.fit_all_stints(subset, min_laps=5, use_fuel_correction=True)
