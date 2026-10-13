"""Stages B and C: pooled models for tire degradation.

    python -m app.ml.tire.train
    python -m app.ml.tire.train --test-season 2024 --no-save

WHY POOLING CHANGES WHAT IS KNOWABLE
------------------------------------
Stage A fits one stint at a time, where tyre_life and lap_number are perfectly
collinear (corr = 1.000) so the fuel effect is unidentifiable and must be
imposed from physics.

Pooling breaks that collinearity. Across stints the same tire age occurs at wildly
different fuel loads -- on 2024 Monza a tyre_life of 5 appears at lap 4 and at
lap 46, sd 13.9 laps -- so a pooled model can see both and estimate them
separately. That is why `lap_number` is kept as a feature even though the target
is already fuel-corrected: it lets the model correct MY correction, absorbing
whatever residual fuel or track-evolution effect the physics assumption got
wrong.

VALIDATION
----------
Two different splits, for two different jobs:

  GroupKFold(session_id)  -- model selection. Grouping by session is essential:
                             laps from one session are not independent, and a
                             random split would put near-duplicate laps from the
                             same stint on both sides of the fold.

  time split by season    -- the number we REPORT. Train on past seasons, test
                             on the most recent. This answers the question that
                             matters: would the model have worked on a season it
                             never saw?

Reporting the CV number as if it were a generalisation estimate is the standard
way to overstate a model, so both are printed and clearly labelled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.db.base import SessionLocal, engine
from app.ml import registry
from app.ml.categories import extract_categories
from app.ml.tire import dataset as ds

MODEL_NAME = "tire_degradation"

N_FOLDS = 5

# Feature order is pinned on disk by the registry; a reordered frame at predict
# time produces plausible wrong numbers rather than an error.
MONOTONIC_INCREASING = ("tyre_life", "tyre_life_sq")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE is the headline because it is directly interpretable in seconds.

    RMSE is reported alongside it: RMSE penalises large errors more, so
    RMSE >> MAE means a few laps are badly wrong rather than all laps being
    slightly wrong.
    """
    return {
        "mae_ms": float(mean_absolute_error(y_true, y_pred)),
        "mae_s": float(mean_absolute_error(y_true, y_pred) / 1000.0),
        "rmse_ms": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def residuals_by_tyre_life(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Mean signed residual per tire-age bucket.

    An overall MAE can hide systematic bias. If the model is consistently
    optimistic on old tires, Phase 3's simulator -- which cares most about the
    end of a stint, where the pit-stop decision is made -- inherits that bias.
    A signed mean per bucket is what exposes it; MAE cannot, because it discards
    the sign.
    """
    buckets = pd.cut(
        df["tyre_life"], bins=[0, 5, 10, 15, 20, 25, 30, 40, 100], right=False
    )
    frame = pd.DataFrame(
        {"bucket": buckets, "residual_ms": y_true - y_pred, "abs_ms": np.abs(y_true - y_pred)}
    )
    out = frame.groupby("bucket", observed=True).agg(
        n=("residual_ms", "size"),
        mean_residual_ms=("residual_ms", "mean"),
        mae_ms=("abs_ms", "mean"),
    )
    return out.reset_index()


# ---------------------------------------------------------------------------
# Stage B: regularised linear model
# ---------------------------------------------------------------------------


def build_stage_b() -> Pipeline:
    """Ridge on one-hot categoricals plus scaled numerics.

    A linear model cannot consume a pandas `category` dtype, so categoricals are
    one-hot encoded here. `handle_unknown="ignore"` makes an unseen circuit
    encode as all-zeros, which is a graceful degradation rather than XGBoost's
    hard error -- a real advantage of the linear model worth knowing about.

    Numerics are standardised because Ridge penalises the SQUARED magnitude of
    coefficients. Without scaling, `race_laps` (~44-78) and `tyre_life` (~1-40)
    would be penalised on different effective scales and the regularisation
    would be arbitrary.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ds.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ds.CATEGORICAL_FEATURES,
            ),
        ]
    )
    # alpha=1.0 is a deliberate, modest default. Tuning it is a notebook
    # exercise; the point of Stage B is an honest, interpretable reference.
    return Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])


# ---------------------------------------------------------------------------
# Stage C: XGBoost (primary) and HistGBR (control)
# ---------------------------------------------------------------------------


def build_stage_c_xgboost(feature_names: list[str], early_stopping: bool) -> Any:
    """XGBoost with native categoricals and a monotonic constraint.

    Three settings carry real meaning:

    `enable_categorical=True` with `tree_method="hist"` lets XGBoost split on
    `category` dtype columns directly, with no one-hot encoding. (`tree_method`
    defaults to "auto", which IS "hist", but it is set explicitly so the
    requirement is visible.)

    `monotone_constraints` forces the prediction to be non-decreasing in
    `tyre_life` and `tyre_life_sq`. This encodes the physics that a tire never
    gets faster as it ages, rather than hoping the model infers it from noisy
    data. Constraints are how domain knowledge enters a model, and both
    tyre_life terms must be constrained together -- constraining one while
    leaving its square free would let the model route around the constraint.

    HYPERPARAMETERS WERE TUNED, AND IT MATTERED FAR MORE THAN MODEL CHOICE.
    Measured by GroupKFold on this dataset:

        n_estimators=600, max_depth=6, reg_lambda=5   CV 0.5373s  train 0.3750s
        n_estimators=120, max_depth=3, reg_lambda=20  CV 0.4779s  train 0.4570s

    The first config -- a perfectly ordinary starting point, and close to
    XGBoost's own defaults for depth -- is WORSE than a plain Ridge regression
    and barely better than doing nothing, with a 0.166s train/holdout gap. The
    second closes that gap to 0.028s and wins outright.

    The lesson generalises: with ~50k rows of mostly-noise target, a deep forest
    memorises. XGBoost's shipped defaults are NOT conservative (reg_lambda=1,
    subsample=1, colsample_bytree=1, no depth limit in some versions), so
    regularisation is not an optional refinement here, it is the whole game.
    """
    import xgboost as xgb

    constraints = {
        name: (1 if name in MONOTONIC_INCREASING else 0) for name in feature_names
    }

    params: dict[str, Any] = dict(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=20.0,
        reg_alpha=0.1,
        tree_method="hist",
        enable_categorical=True,
        monotone_constraints=constraints,
        random_state=42,
        n_jobs=-1,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 50
    return xgb.XGBRegressor(**params)


def build_stage_c_histgbr(feature_names: list[str]) -> HistGradientBoostingRegressor:
    """sklearn's control model.

    A second, independent implementation of the same idea. If a tuned XGBoost
    LOSES to a near-default HistGBR, that is a red flag -- usually over-tuning
    against the folds or a leak in the eval set -- and it is far easier to notice
    with a reference model sitting next to it than in isolation.

    That is not hypothetical: this control is what exposed BOTH boosters being
    over-parameterised on the first full training run.

    NOTE `early_stopping=False`, which is deliberate and important.
    sklearn's internal early stopping carves out `validation_fraction` at
    RANDOM, not by session. On this dataset a random split puts near-duplicate
    laps from the same stint on both sides, so the internal validation score
    keeps improving, early stopping never fires, and the model runs its full
    budget straight into overfitting. Measured: with the leaky internal split it
    scored 0.5254s holdout (+0.4% over baseline, r2 = -0.013); with it disabled
    and `max_iter` tuned by GroupKFold instead, 0.4856s (+8.0%).

    It is the exact leak this project warns about, and sklearn's API offers no
    way to group its internal split -- so the only correct option is to turn it
    off and choose the budget externally.

    `categorical_features="from_dtype"` picks up the same `category` columns.
    """
    monotonic = [1 if n in MONOTONIC_INCREASING else 0 for n in feature_names]
    return HistGradientBoostingRegressor(
        # Tuned by GroupKFold. More iterations monotonically improved TRAIN MAE
        # while making CV and holdout worse -- the signature of overfitting.
        max_iter=60,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=10.0,
        categorical_features="from_dtype",
        monotonic_cst=monotonic,
        early_stopping=False,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------


def cross_validate_grouped(
    build_fn, train: pd.DataFrame, n_folds: int = N_FOLDS
) -> dict[str, Any]:
    """GroupKFold by session_id.

    Grouping is what makes this honest. Without it, laps 12 and 13 of the same
    stint -- nearly identical rows -- land on opposite sides of a fold, and the
    model is scored on data it has effectively memorised.
    """
    X = ds.feature_matrix(train)
    y = ds.target_vector(train)
    groups = train["session_id"].to_numpy()

    splitter = GroupKFold(n_splits=n_folds)
    fold_scores: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = build_fn()
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # XGBoost with early stopping needs an eval set; it is the held-out
        # fold, which is itself grouped by session.
        if hasattr(model, "early_stopping_rounds") and model.early_stopping_rounds:
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr)

        scores = evaluate(y_val, model.predict(X_val))
        scores["fold"] = fold
        fold_scores.append(scores)

    maes = [s["mae_ms"] for s in fold_scores]
    return {
        "folds": fold_scores,
        "mae_ms_mean": float(np.mean(maes)),
        "mae_ms_std": float(np.std(maes)),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def session_holdout(
    train: pd.DataFrame, fraction: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set out of train by SESSION, for early stopping.

    Splitting by session rather than by row is the same discipline as
    GroupKFold: an eval set sharing stints with the training set would make
    early stopping fire far too late.
    """
    sessions = np.array(sorted(train["session_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)
    n_val = max(1, int(len(sessions) * fraction))
    val_sessions = set(sessions[:n_val].tolist())
    is_val = train["session_id"].isin(val_sessions)
    return train[~is_val].copy(), train[is_val].copy()


def train_all(test_season: int | None = None, save: bool = True) -> dict[str, Any]:
    """Build the dataset, fit Stages B and C, evaluate, and register the winner."""
    print("Loading dataset from features.stint_laps ...")
    df = ds.build_dataset(engine)
    if df.empty:
        raise RuntimeError("features.stint_laps is empty -- ingest some races first")

    train, test = ds.time_split(df, test_season=test_season)
    if train.empty or test.empty:
        raise RuntimeError(
            f"time split produced an empty side: "
            f"train={len(train)}, test={len(test)}. "
            f"Seasons present: {sorted(df.season_year.unique())}"
        )

    test_year = int(test["season_year"].iloc[0])
    train_years = sorted(int(y) for y in train["season_year"].unique())

    print(
        f"  {len(df):,} laps / {df.stint_key.nunique():,} stints\n"
        f"  train: {len(train):,} laps, seasons {train_years}\n"
        f"  test : {len(test):,} laps, season {test_year}"
    )

    baseline_train = ds.baseline_mae_ms(train)
    baseline_test = ds.baseline_mae_ms(test)
    print(
        f"\nBASELINE ('this lap == its stint median')\n"
        f"  train MAE {baseline_train / 1000:.4f}s   test MAE {baseline_test / 1000:.4f}s\n"
        f"  Any model must beat this or it has learned nothing about tire age."
    )

    feature_names = ds.FEATURES
    vocab = extract_categories(train, ds.CATEGORICAL_FEATURES)

    X_train, y_train = ds.feature_matrix(train), ds.target_vector(train)
    X_test, y_test = ds.feature_matrix(test), ds.target_vector(test)

    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}

    specs = [
        ("stage_b_ridge", lambda: build_stage_b(), "sklearn"),
        ("stage_c_xgboost", lambda: build_stage_c_xgboost(feature_names, True), "xgboost"),
        ("stage_c_histgbr", lambda: build_stage_c_histgbr(feature_names), "sklearn"),
    ]

    for label, build_fn, backend in specs:
        print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")

        started = time.perf_counter()
        cv = cross_validate_grouped(build_fn, train)
        cv_secs = time.perf_counter() - started
        print(
            f"  GroupKFold({N_FOLDS}) by session: "
            f"MAE {cv['mae_ms_mean'] / 1000:.4f}s +/- {cv['mae_ms_std'] / 1000:.4f}s "
            f"({cv_secs:.1f}s)"
        )

        # Refit on all training seasons, then score the unseen test season.
        model = build_fn()
        started = time.perf_counter()
        if hasattr(model, "early_stopping_rounds") and model.early_stopping_rounds:
            inner_train, inner_val = session_holdout(train)
            model.fit(
                ds.feature_matrix(inner_train),
                ds.target_vector(inner_train),
                eval_set=[(ds.feature_matrix(inner_val), ds.target_vector(inner_val))],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        fit_secs = time.perf_counter() - started

        train_scores = evaluate(y_train, model.predict(X_train))
        test_scores = evaluate(y_test, model.predict(X_test))

        lift = 1.0 - test_scores["mae_ms"] / baseline_test
        print(
            f"  train MAE {train_scores['mae_s']:.4f}s | "
            f"HOLDOUT {test_year} MAE {test_scores['mae_s']:.4f}s  "
            f"(RMSE {test_scores['rmse_ms'] / 1000:.4f}s, r2 {test_scores['r2']:.3f})"
        )
        verdict = "better" if lift > 0 else "WORSE"
        print(f"  vs baseline: {lift:+.1%} {verdict}  ({fit_secs:.1f}s fit)")

        results[label] = {
            "backend": backend,
            "cv": cv,
            "train": train_scores,
            "test": test_scores,
            "baseline_test_mae_ms": baseline_test,
            "lift_vs_baseline": lift,
            "fit_seconds": fit_secs,
        }
        fitted[label] = model

    # The primary model is XGBoost by project decision; HistGBR is the control.
    primary = "stage_c_xgboost"
    control = "stage_c_histgbr"
    p_mae = results[primary]["test"]["mae_ms"]
    c_mae = results[control]["test"]["mae_ms"]

    print(f"\n{'=' * 64}\nXGBoost vs HistGBR (the honest comparison)\n{'=' * 64}")
    print(f"  XGBoost holdout MAE : {p_mae / 1000:.4f}s")
    print(f"  HistGBR holdout MAE : {c_mae / 1000:.4f}s")
    if c_mae < p_mae:
        print(
            "  HistGBR WINS. Not a disaster -- these two are near-identical\n"
            "  algorithms and ties are expected. But check for over-tuning\n"
            "  against the folds or a leak in the XGBoost eval set."
        )
    else:
        print(f"  XGBoost ahead by {(c_mae - p_mae) / 1000:.4f}s ({1 - p_mae / c_mae:+.1%}).")

    print("\nResiduals by tire age (signed mean -- exposes bias MAE would hide):")
    print(
        residuals_by_tyre_life(test, y_test, fitted[primary].predict(X_test))
        .assign(
            mean_residual_s=lambda d: (d.mean_residual_ms / 1000).round(4),
            mae_s=lambda d: (d.mae_ms / 1000).round(4),
        )[["bucket", "n", "mean_residual_s", "mae_s"]]
        .to_string(index=False)
    )

    if save:
        version = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        with SessionLocal() as db:
            for label in (primary, control, "stage_b_ridge"):
                registry.save_model(
                    db,
                    name=f"{MODEL_NAME}__{label}",
                    version=version,
                    backend=results[label]["backend"],
                    model=fitted[label],
                    categories=vocab,
                    feature_names=feature_names,
                    metrics=results[label],
                    train_season_min=min(train_years),
                    train_season_max=max(train_years),
                    n_train_rows=len(train),
                    notes=f"holdout season {test_year}",
                    make_active=True,
                )
            # The canonical model the API serves.
            registry.save_model(
                db,
                name=MODEL_NAME,
                version=version,
                backend=results[primary]["backend"],
                model=fitted[primary],
                categories=vocab,
                feature_names=feature_names,
                metrics={
                    **results[primary],
                    "control_histgbr_test_mae_ms": c_mae,
                },
                train_season_min=min(train_years),
                train_season_max=max(train_years),
                n_train_rows=len(train),
                notes=f"primary=xgboost, holdout season {test_year}",
                make_active=True,
            )
            db.commit()
        print(f"\nRegistered models at version {version} (active).")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tire degradation models.")
    parser.add_argument("--test-season", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    results = train_all(test_season=args.test_season, save=not args.no_save)
    print("\n" + json.dumps({k: v["test"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
