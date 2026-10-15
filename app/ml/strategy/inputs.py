"""Assemble the empirical inputs the pit-strategy simulator needs.

One `StrategyInputs` object fully determines a circuit's simulation. Everything
in it is either measured from data (`features.pit_loss`,
`features.race_conditions`, Stage A degradation) or a clearly-labelled
assumption with a default.

WHY DEGRADATION COMES FROM STAGE A, NOT THE PHASE 2 MODEL
---------------------------------------------------------
Phase 2 trained a pooled XGBoost model and then discovered it predicts a median
of only ~68% of the degradation rate Stage A measures, compressing the spread
across circuits roughly 4x. The regularisation that minimised per-lap MAE also
flattened the curve, because most lap-to-lap variation inside a stint is noise
and MAE is minimised by shrinking toward zero.

That is fine for predicting one lap and ruinous here, because this simulator
INTEGRATES the curve over ~30 laps. A rate 40% too flat makes long stints look
far cheaper than they are and biases every strategy toward fewer stops.

So the default degradation source is Stage A's measured (circuit, compound)
table. `DegradationSource.MODEL` is available for comparison, and
`notebooks/03_pit_strategy.ipynb` shows what it does to the recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ml.tire import dataset as ds
from app.ml.tire import stage_a

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

# Fallbacks, used only when a circuit has too little data. Each is the median
# across everything ingested rather than a guess.
DEFAULT_PIT_LOSS_MS = 23_300.0
DEFAULT_RACE_LAPS = 57
DEFAULT_SC_PROBABILITY = 0.686
DEFAULT_SC_LAPS = 5

# A stint shorter than this is never a real strategy -- it is a mistake or a
# damage-repair stop, and including it lets the optimiser propose nonsense.
MIN_STINT_LAPS = 5

# Hard ceiling on a dry stint regardless of what the data shows, so a circuit
# with one freak long stint cannot propose a 50-lap run.
MAX_STINT_LAPS_CEILING = 45

# Minimum stints observed before a (circuit, compound) degradation rate is
# trusted; below this we fall back to the compound's global median.
MIN_STINTS_FOR_RATE = 5


class DegradationSource(StrEnum):
    STAGE_A = "stage_a"  # measured per-stint OLS slopes (default)
    MODEL = "model"  # the Phase 2 pooled model -- see module docstring


@dataclass
class StrategyInputs:
    """Everything needed to simulate one circuit."""

    circuit: str
    race_laps: int
    base_pace_ms: float
    pit_loss_ms: float
    # Robust spread, derived from (p90 - p10), NOT stddev. Several circuits have
    # a stddev inflated to 9-11s by a handful of contaminated stops while their
    # interdecile range stays near 5s, so stddev would make the Monte Carlo far
    # noisier than reality.
    pit_loss_sd_ms: float
    degradation_s_per_lap: dict[str, float]
    max_stint_laps: dict[str, int]
    sc_probability: float
    median_sc_laps: int
    source: DegradationSource = DegradationSource.STAGE_A
    n_stops_observed: int = 0
    n_races_observed: int = 0
    notes: list[str] = field(default_factory=list)

    def degradation(self, compound: str) -> float:
        """Degradation in s per lap of tire age for this compound."""
        return self.degradation_s_per_lap[compound]

    def max_stint(self, compound: str) -> int:
        return self.max_stint_laps[compound]


def _robust_sd_ms(p10: float | None, p90: float | None, fallback: float) -> float:
    """Spread from the interdecile range, assuming roughly normal tails.

    p90 - p10 spans about 2.563 standard deviations for a normal distribution,
    so dividing by that recovers an sd that ignores the extreme tails -- which
    is what we want, since the tails here are contamination rather than signal.
    """
    if p10 is None or p90 is None or p90 <= p10:
        return fallback
    return float((p90 - p10) / 2.563)


def load_pit_loss(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM features.pit_loss"), engine)


def load_race_conditions(engine: Engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM features.race_conditions"), engine)


def compute_base_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit reference pace: a fresh tire on an empty tank.

    Taken as the median FUEL-CORRECTED lap time over laps with tyre_life <= 3.
    Both conditions matter and they pull in opposite directions in raw data: a
    fresh tire only occurs at the start of a stint, which is when the car is
    heaviest. Only after fuel correction can "fresh tire" and "empty tank" be
    measured at once, which is why this lives in Python next to the fuel model
    rather than in a SQL view.

    For RANKING strategies at one circuit this value cancels out -- it is added
    to every strategy equally. It matters only for reporting absolute race
    times.
    """
    fresh = laps[laps["tyre_life"] <= 3]
    grouped = fresh.groupby("circuit", observed=True)["lap_time_fuel_corr_ms"]
    return grouped.median().rename("base_pace_ms").reset_index()


def compute_stint_limits(laps: pd.DataFrame) -> pd.DataFrame:
    """Longest viable stint per (circuit, compound), from what teams actually did.

    The p95 of observed stint lengths rather than the max: one freak stint --
    a car nursing damage to the finish, or a rain-interrupted race -- should not
    define what the optimiser may propose.
    """
    per_stint = laps.groupby(
        ["circuit", "compound", "stint_key"], observed=True
    )["tyre_life"].max()
    out = (
        per_stint.groupby(["circuit", "compound"], observed=True)
        .agg(p95=lambda s: s.quantile(0.95), n_stints="count")
        .reset_index()
    )
    out["max_stint_laps"] = (
        out["p95"].clip(lower=MIN_STINT_LAPS + 1, upper=MAX_STINT_LAPS_CEILING).round().astype(int)
    )
    return out


def build_inputs(
    engine: Engine,
    circuit: str,
    *,
    race_laps: int | None = None,
    source: DegradationSource = DegradationSource.STAGE_A,
    laps: pd.DataFrame | None = None,
) -> StrategyInputs:
    """Assemble `StrategyInputs` for one circuit.

    `laps` may be passed in to avoid rebuilding the ~100k-row dataset, which the
    API does via a cached loader.

    Missing data degrades to documented defaults and records WHY in `notes`,
    rather than failing or silently substituting. A caller can then see that a
    recommendation rests on a fallback.
    """
    if laps is None:
        laps = ds.build_dataset(engine)

    notes: list[str] = []

    # --- pit loss ---
    pit = load_pit_loss(engine)
    row = pit[pit["circuit"] == circuit]
    if row.empty:
        pit_loss_ms = DEFAULT_PIT_LOSS_MS
        pit_sd_ms = 2_000.0
        n_stops = 0
        notes.append(
            f"no measured pit stops for {circuit!r}; using the global median "
            f"{DEFAULT_PIT_LOSS_MS / 1000:.1f}s"
        )
    else:
        r = row.iloc[0]
        pit_loss_ms = float(r["pit_loss_ms"])
        pit_sd_ms = _robust_sd_ms(r.get("pit_loss_p10_ms"), r.get("pit_loss_p90_ms"), 2_000.0)
        n_stops = int(r["n_stops"])
        if n_stops < 20:
            notes.append(f"pit loss from only {n_stops} stops; treat as indicative")

    # --- race conditions ---
    cond = load_race_conditions(engine)
    crow = cond[cond["circuit"] == circuit]
    if crow.empty:
        resolved_laps = race_laps or DEFAULT_RACE_LAPS
        sc_probability = DEFAULT_SC_PROBABILITY
        median_sc = DEFAULT_SC_LAPS
        n_races = 0
        notes.append(f"no race history for {circuit!r}; using global safety-car rates")
    else:
        c = crow.iloc[0]
        resolved_laps = race_laps or int(c["race_laps"])
        sc_probability = float(c["sc_probability"])
        median_sc = int(c["median_sc_laps_when_deployed"] or DEFAULT_SC_LAPS)
        n_races = int(c["n_races"])
        if n_races < 4:
            notes.append(
                f"safety-car rate from only {n_races} races; the probability is coarse"
            )

    # --- base pace ---
    base = compute_base_pace(laps)
    brow = base[base["circuit"] == circuit]
    if brow.empty:
        raise LookupError(
            f"no lap data for circuit {circuit!r}. Available: "
            f"{sorted(base['circuit'].unique().tolist())}"
        )
    base_pace_ms = float(brow.iloc[0]["base_pace_ms"])

    # --- degradation ---
    if source is DegradationSource.STAGE_A:
        fits = stage_a.fit_all_stints(laps, use_fuel_correction=True)
        table = stage_a.degradation_lookup(fits, min_stints=MIN_STINTS_FOR_RATE)
        global_by_compound = (
            fits.groupby("compound", observed=True)["slope"].median().to_dict()
        )
        rates: dict[str, float] = {}
        for compound in DRY_COMPOUNDS:
            match = table[(table.circuit == circuit) & (table.compound == compound)]
            if match.empty:
                fallback = float(global_by_compound.get(compound, 0.06))
                rates[compound] = fallback
                notes.append(
                    f"{compound}: no {circuit} data with >={MIN_STINTS_FOR_RATE} stints; "
                    f"using the global median {fallback:+.4f} s/lap"
                )
            else:
                rates[compound] = float(match.iloc[0]["slope_median"])
    else:
        from app.ml.tire import predict as tire_predict

        rates = {}
        for compound in DRY_COMPOUNDS:
            curve = tire_predict.predict_curve(
                circuit, compound, max_tyre_life=25, race_laps=resolved_laps
            )
            rates[compound] = curve.degradation_rate_s_per_lap
        notes.append(
            "degradation from the Phase 2 pooled model, which under-predicts the "
            "measured rate by ~32% -- see docs/phase2-tire-degradation.md"
        )

    # A negative measured rate means track evolution outran tire wear in that
    # sample. Physically a tire does not improve, and a negative rate would make
    # the optimiser want infinitely long stints, so clamp at zero.
    for compound, rate in rates.items():
        if rate < 0:
            notes.append(
                f"{compound}: measured rate {rate:+.4f} s/lap is negative "
                "(track evolution); clamped to 0"
            )
            rates[compound] = 0.0

    # --- stint limits ---
    limits_table = compute_stint_limits(laps)
    limits: dict[str, int] = {}
    for compound in DRY_COMPOUNDS:
        match = limits_table[
            (limits_table.circuit == circuit) & (limits_table.compound == compound)
        ]
        if match.empty or int(match.iloc[0]["n_stints"]) < MIN_STINTS_FOR_RATE:
            limits[compound] = min(resolved_laps, MAX_STINT_LAPS_CEILING)
        else:
            limits[compound] = int(match.iloc[0]["max_stint_laps"])

    return StrategyInputs(
        circuit=circuit,
        race_laps=resolved_laps,
        base_pace_ms=base_pace_ms,
        pit_loss_ms=pit_loss_ms,
        pit_loss_sd_ms=pit_sd_ms,
        degradation_s_per_lap=rates,
        max_stint_laps=limits,
        sc_probability=sc_probability,
        median_sc_laps=median_sc,
        source=source,
        n_stops_observed=n_stops,
        n_races_observed=n_races,
        notes=notes,
    )


def available_circuits(engine: Engine) -> list[str]:
    return sorted(
        pd.read_sql(
            text("SELECT DISTINCT circuit FROM features.pit_loss ORDER BY circuit"), engine
        )["circuit"].tolist()
    )
