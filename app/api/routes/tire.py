"""Phase 2 endpoints: tire degradation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session as OrmSession

from app.db.base import engine
from app.db.deps import get_db
from app.db.models import Session as RaceSession
from app.ml import registry
from app.ml.tire import predict as tire_predict
from app.ml.tire.train import MODEL_NAME
from app.schemas.tire import (
    DegradationPoint,
    DegradationRequest,
    DegradationResponse,
    ModelInfoOut,
    ModelMetricsOut,
    StintFitOut,
)

router = APIRouter(prefix="/tire", tags=["tire"])


def _no_model_error() -> HTTPException:
    return HTTPException(
        503,
        "No tire-degradation model is registered yet. "
        "Train one with: python -m app.ml.tire.train",
    )


@router.post("/degradation", response_model=DegradationResponse)
def degradation_curve(
    request: DegradationRequest, db: OrmSession = Depends(get_db)
):
    """Predicted lap-time loss as a tire ages.

    `delta_vs_fresh_s` is the number to read: how much slower this tire is at a
    given age than it was new. The raw model target (`delta_vs_stint_median_ms`)
    is also returned for transparency, but it is only interpretable if you know
    the stint median it is relative to.

    A non-empty `unseen_categories` means a circuit, driver or team was absent
    from training and was mapped to NaN, so the prediction fell back toward the
    global average for that field. Treat such a curve with suspicion.
    """
    try:
        curve = tire_predict.predict_curve(
            circuit=request.circuit,
            compound=request.compound.upper(),
            max_tyre_life=request.max_tyre_life,
            race_laps=request.race_laps,
            stint_start_lap=request.stint_start_lap,
            air_temp=request.air_temp,
            track_temp=request.track_temp,
            driver=request.driver.upper() if request.driver else None,
            team=request.team,
            stint_number=request.stint_number,
            is_fresh_tyre=request.is_fresh_tyre,
            reference_lap=request.reference_lap,
            db=db,
        )
    except (LookupError, FileNotFoundError) as exc:
        raise _no_model_error() from exc

    points = [
        DegradationPoint(
            tyre_life=life,
            delta_vs_fresh_ms=fresh,
            delta_vs_fresh_s=round(fresh / 1000.0, 4),
            delta_vs_stint_median_ms=median,
        )
        for life, fresh, median in zip(
            curve.tyre_life,
            curve.delta_vs_fresh_ms,
            curve.delta_vs_stint_median_ms,
            strict=True,
        )
    ]

    return DegradationResponse(
        circuit=curve.circuit,
        compound=curve.compound,
        degradation_rate_s_per_lap=curve.degradation_rate_s_per_lap,
        total_loss_s=round(curve.delta_vs_fresh_ms[-1] / 1000.0, 4),
        points=points,
        model_version=curve.model_version,
        model_backend=curve.model_backend,
        unseen_categories=curve.unseen_categories,
    )


@router.get("/stints/{session_id}", response_model=list[StintFitOut])
def observed_stints(
    session_id: int,
    min_laps: int = Query(5, ge=2, description="Minimum clean laps to fit a slope"),
    db: OrmSession = Depends(get_db),
):
    """Measured degradation for one race, by per-stint OLS (Stage A).

    This is the empirical counterpart to `/degradation`: what the stints in that
    race actually did, not what the model expects. `r2` is included because a
    slope without a goodness-of-fit is not interpretable -- a confident-looking
    rate with r2 near zero is noise.

    Lap times are fuel-corrected before fitting. Without that, the slopes come
    out near zero or negative, because the fuel effect (~0.055 s/lap of gain)
    almost exactly cancels typical degradation.
    """
    if db.get(RaceSession, session_id) is None:
        raise HTTPException(404, f"Session {session_id} not found")

    fits = tire_predict.observed_stint_fits(engine, session_id)
    if fits.empty:
        return []

    fits = fits[fits["n"] >= min_laps]
    return [
        StintFitOut(
            driver=row.driver,
            stint_number=int(row.stint_number),
            compound=row.compound,
            laps=int(row.n),
            tyre_life_min=float(row.tyre_life_min),
            tyre_life_max=float(row.tyre_life_max),
            slope_s_per_lap=round(float(row.slope), 5),
            intercept_s=round(float(row.intercept), 4),
            r2=round(float(row.r2), 4),
        )
        for row in fits.itertuples(index=False)
    ]


@router.get("/model-info", response_model=ModelInfoOut)
def model_info(db: OrmSession = Depends(get_db)):
    """The active model's provenance and honest holdout numbers.

    `baseline_mae_s` and `lift_vs_baseline` are reported alongside the model's
    own MAE deliberately. An MAE on its own is unreadable: 0.45s sounds good or
    bad depending entirely on what doing nothing would have scored.
    """
    try:
        rows = registry.list_models(db, MODEL_NAME)
    except Exception as exc:  # noqa: BLE001
        raise _no_model_error() from exc

    active = next((r for r in rows if r.is_active), None)
    if active is None:
        raise _no_model_error()

    import json

    metrics = json.loads(active.metrics_json) if active.metrics_json else {}
    test = metrics.get("test") or {}
    baseline_ms = metrics.get("baseline_test_mae_ms")
    control_ms = metrics.get("control_histgbr_test_mae_ms")

    holdout = (
        ModelMetricsOut(
            mae_s=round(test["mae_ms"] / 1000.0, 5),
            rmse_s=round(test["rmse_ms"] / 1000.0, 5),
            r2=round(test["r2"], 5),
            n=int(test["n"]),
        )
        if test
        else None
    )

    return ModelInfoOut(
        name=active.name,
        version=active.version,
        backend=active.backend,
        trained_at=active.trained_at,
        train_season_min=active.train_season_min,
        train_season_max=active.train_season_max,
        n_train_rows=active.n_train_rows,
        holdout=holdout,
        baseline_mae_s=round(baseline_ms / 1000.0, 5) if baseline_ms else None,
        lift_vs_baseline=(
            round(metrics["lift_vs_baseline"], 5)
            if metrics.get("lift_vs_baseline") is not None
            else None
        ),
        control_histgbr_mae_s=round(control_ms / 1000.0, 5) if control_ms else None,
        notes=active.notes,
    )
