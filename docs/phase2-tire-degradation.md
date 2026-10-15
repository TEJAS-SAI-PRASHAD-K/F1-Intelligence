# Phase 2 — Tire Degradation

## What this phase is for

Predict how much slower a lap gets as a tire ages. It sounds like a straightforward regression,
and the reason it isn't is the most useful statistics lesson in the project.

Phase 1 ended with a puzzle: raw per-stint slopes were mostly **negative**, implying tires get
faster with age. This phase resolves it, and in doing so runs into three separate problems that
all look like modeling problems and are not.

**The lesson of this phase:** a model cannot separate two things that always move together, and
the metric you optimise determines what you get.

## What was built

| Component | File | Does |
|---|---|---|
| Fuel model | `app/ml/fuel.py` | Physics-based fuel-load correction |
| Modeling view | `alembic/versions/e455366cd113_*.py` | `features.stint_laps` — the clean dataset |
| Dataset | `app/ml/tire/dataset.py` | Fuel correction, target, features, splitting |
| Stage A | `app/ml/tire/stage_a.py` | Per-stint OLS + `(circuit, compound)` lookup |
| Stages B/C | `app/ml/tire/train.py` | Ridge, XGBoost, HistGBR + validation |
| Category pinning | `app/ml/categories.py` | Unseen categories → NaN, not a crash |
| Registry | `app/ml/registry.py` | XGBoost→JSON / sklearn→joblib, `ml.model_versions` |
| Prediction | `app/ml/tire/predict.py` | Degradation curves |
| API | `app/api/routes/tire.py` | 3 endpoints |
| Tests | `tests/test_fuel.py`, `test_tire_dataset.py`, `test_registry.py`, `test_tire_api.py` | 62 tests |
| Notebook | `notebooks/02_tire_degradation.ipynb` | The whole argument, with charts |

Data: **6 seasons (2019–2024)**, 103 races, 104k clean dry green-flag race laps.

---

## Problem 1 — Perfect collinearity

Lap time is a sum of effects:

```
lap_time = base_pace
         + degradation    x tyre_life        <- what we want
         - fuel_effect    x laps_completed   <- car gets lighter, so faster
         + track_evolution
         + noise
```

The obvious fix is to give a model both `tyre_life` and `lap_number` and let it separate them.
**Within a single stint that is mathematically impossible.** Measured on real stints:

```
LEC stint 1  n=15  corr(tyre_life, lap_number) = 1.000000
LEC stint 2  n=38  corr(tyre_life, lap_number) = 1.000000
VER stint 1  n=22  corr(tyre_life, lap_number) = 1.000000
```

Exactly 1.000000, because a stint advances both by one every lap. The design matrix is singular,
so the two coefficients are **unidentifiable** — not poorly estimated, unidentifiable. No sample
size helps.

There are exactly two ways out:

| | approach | used by |
|---|---|---|
| 1 | **Impose** the fuel coefficient from physics and subtract it | Stage A |
| 2 | **Pool** across stints, where the collinearity breaks | Stages B, C |

Pooling works because the same tire age occurs at very different fuel loads: at 2024 Monza a
`tyre_life` of 5 appears at lap 4 **and** at lap 46, with a standard deviation of 13.9 laps.

### The fuel model, and evidence it is honest

`app/ml/fuel.py` imposes 110 kg of race fuel at 0.03 s/kg, derived per circuit from race length:

```
fuel_effect(53 laps) = (110/53) * 0.03 = 0.0623 s/lap   # Monza
fuel_effect(78 laps) = (110/78) * 0.03 = 0.0423 s/lap   # Monaco
```

An imposed assumption deserves scrutiny, so here is an independent check. Applying the correction
to 2,700+ measured stint slopes:

| | median slope | % negative | median r² |
|---|---|---|---|
| Raw lap times | +0.0040 s/lap | 47.1% | 0.254 |
| Fuel-corrected | **+0.0578 s/lap** | 18.6% | **0.439** |

- The **shift** is +0.0537 s/lap.
- The **physics-predicted** effect at a typical 60-lap race is +0.0550 s/lap.
- They agree to within 2%, and these are two completely separate calculations.

The r² nearly **doubling** is the stronger evidence: the correction didn't merely translate the
slopes, it made lap time genuinely more linear in tire age. That is what you'd expect if it
removed a real competing effect rather than adding noise.

The share of physically impossible negative slopes falls from 47% to 19%. It does not reach zero,
and should not — a dry track rubbering in really does get faster, so some stints legitimately
improve.

---

## Problem 2 — Phase 1's filters were still insufficient

Phase 1 found that `is_accurate` **and** a strict green-flag filter together still admit a
148-second restart lap. Phase 2 hit the same class of problem one level down.

The view filters to dry compounds (SOFT/MEDIUM/HARD) to control track evolution. Still not
enough: a driver on slicks when rain arrives sets a lap **19–24 seconds** off their stint median
that is accurate, strictly green, *and* on a dry compound.

```
2023 Zandvoort   SAR  stint 1  lap  4  +24.2s
2023 Zandvoort   PIA  stint 1  lap  5  +21.9s
2022 Silverstone STR  stint 2  lap  3  +19.7s
2023 Monaco      PIA  stint 1  lap 53  +19.3s
```

All races with changeable conditions.

**How the threshold was chosen** — from the data, not by taste. Real degradation has a total
swing of 1.18s at the median stint, 3.09s at p95, 5.45s at p99. The target is a deviation from
the stint *median*, so it spans about half that: ±2.7s even for a p99 stint. The cap is therefore
set at **4s** (`dataset.MAX_ABS_TARGET_MS`): above any real degradation, below anything that is
clearly a different phenomenon.

Cost: 324 rows of 54,000 (**0.60%**).

**Why 0.6% matters.** Least squares minimises *squared* error, so a lap 24s off trend carries
roughly 80× the leverage of a typical 2.7s one. One contaminated lap can dominate a circuit's
entire fitted curve.

**Two-pass median.** The cap is applied, then the stint median is **recomputed** on the
survivors. The median is robust, so dropping outliers barely moves it — but "barely" is not "not
at all", and the baseline the model regresses against should be computed from the rows the model
actually sees.

### Filtering responsibility is split deliberately

| layer | enforces |
|---|---|
| `features.stint_laps` | **structural** validity: green, accurate, dry, enough laps, no restart laps |
| `dataset.add_target` | **pace** validity, in target units, unit-tested and tunable from a notebook |

---

## Target design

Not raw lap time — that is dominated by circuit and car, so the model would spend its capacity
learning that Monaco is slower than Monza. Instead each lap's deviation from its **own stint's**
fuel-corrected median:

```
target = corrected_lap - median(corrected_lap over that stint)
```

This removes, in one step, everything constant inside a stint: driver skill, car performance,
circuit, ambient conditions, even the compound's absolute grip.

**The obvious objection — doesn't that remove the degradation too?** No, and it is worth proving.
Build a synthetic stint with exactly 60 ms/lap of degradation under a real fuel effect:

| | slope |
|---|---|
| raw times | +12.4 ms/lap (understated by the fuel gain) |
| corrected times | **+60.0 ms/lap** (recovered exactly) |
| the target | **+60.0 ms/lap** (level removed, slope preserved) |

The level is a nuisance parameter; the slope is the quantity of interest. Pinned by
`tests/test_tire_dataset.py::TestTarget::test_target_preserves_the_slope_it_removes_the_level`.

---

## Problem 3 — A random split flatters the model by 20%

Consecutive laps in one stint are near-duplicates: same car, same driver, same track, tire age
differing by one. A **random** row split puts lap 12 in train and lap 13 in test, and the model is
scored on rows it has effectively memorised.

Measured on this dataset:

| split | MAE |
|---|---|
| baseline (do nothing) | 0.5276s |
| **random 5-fold** | **0.3842s** ← flattering, and invalid |
| `GroupKFold` by session | 0.4789s ← honest |

The random split makes the model look **~20% better than it is**. Reported as a generalisation
estimate it is simply false: it measures interpolation *within* stints the model has already seen.

### Two splits, two jobs

| split | grouped by | used for |
|---|---|---|
| `GroupKFold` | `session_id` | model selection and tuning |
| time split | `season_year` | **the number we report** |

Train 2019–2023 (82,939 laps), test 2024 (21,036 laps). The time split answers the question that
actually matters: *would this model have worked on a season it never saw?*

### The same leak, inside scikit-learn

`HistGradientBoostingRegressor(early_stopping=True, validation_fraction=0.15)` carves out its
validation set **at random**, not by group. On this data the internal score keeps improving, early
stopping never fires, and the model runs its full budget into overfitting:

| | holdout MAE | vs baseline | r² |
|---|---|---|---|
| leaky internal early stopping | 0.5254s | +0.4% | **−0.013** |
| disabled, `max_iter` tuned by GroupKFold | **0.4856s** | +8.0% | 0.143 |

sklearn's API offers no way to group that split, so the only correct option is
`early_stopping=False` with the budget chosen externally. **The control model is what exposed
this** — it is not detectable from one model in isolation.

---

## Results

| model | holdout MAE | r² | vs baseline |
|---|---|---|---|
| Baseline ("this lap == its stint median") | 0.5276s | — | — |
| Stage B Ridge | 0.4913s | 0.129 | +6.9% |
| Stage C XGBoost (primary) | 0.4885s | 0.126 | +7.4% |
| Stage C HistGBR (control) | **0.4856s** | **0.143** | **+8.0%** |

**All three are within 1.2% of each other.** HistGBR edges XGBoost by 0.0029s against a CV
standard deviation of 0.027s — that is noise, not a win. XGBoost was not re-tuned until it won;
that is precisely the over-tuning the control exists to catch.

That a plain Ridge sits within 1.2% of the best booster is itself a finding: **once fuel-corrected,
degradation is close to linear in tire age.** A genuine fact about F1, not a failure.

### Regularisation mattered far more than model choice

| config | CV MAE | holdout | train | gap |
|---|---|---|---|---|
| 600 trees, depth 6 (first attempt) | 0.5373s | 0.5410s | 0.3750s | **0.166s** |
| 120 trees, depth 3, λ=20 | **0.4779s** | **0.4846s** | 0.4570s | 0.028s |

The first config — an entirely ordinary starting point — scored **worse than Ridge and barely
better than doing nothing.** XGBoost's shipped defaults are not conservative (`reg_lambda=1`,
`subsample=1`, `colsample_bytree=1`), so with ~100k rows of mostly-noise target, regularisation is
not a refinement. It is the whole game.

### An r² of 0.13 is the honest number

Only ~13% of the variance in the target is explained, and that is expected. The target is
deviation from a stint's own median, and most of that is irreducible noise: traffic, a missed
apex, a gust, lifting for a blue flag. Tire age explains a real but minority share. A model
claiming to explain most of it would be overfitting.

---

## The finding that matters most for Phase 3

The MAE looks fine. **The curve magnitudes do not.**

Comparing the pooled model's predicted rate against Stage A's independently *measured* rate:

| circuit | compound | measured | predicted | n stints |
|---|---|---|---|---|
| Shanghai | MEDIUM | 0.1392 | 0.0594 | 39 |
| Sakhir | SOFT | 0.1377 | 0.0575 | 116 |
| Spa | MEDIUM | 0.1111 | 0.0526 | 101 |
| Suzuka | HARD | 0.1048 | 0.0589 | 56 |
| Las Vegas | MEDIUM | 0.0938 | 0.0171 | 23 |

- The model predicts a median **68% of the measured rate**
- Ordering correlation is good: **0.750**
- But the spread is **compressed ~4×**: 0.187 s/lap measured vs 0.045 s/lap predicted

Within a circuit it barely separates compounds at all — Sakhir SOFT/MEDIUM/HARD all predict
~0.057 s/lap, where Stage A measures 0.138 / 0.102 / 0.095.

### Why — and why it is not a bug

The regularisation that *minimised per-lap MAE* also **flattened the curve**:

| config | holdout MAE | % of measured rate | ordering corr |
|---|---|---|---|
| depth 3, λ=20 (shipped) | **0.4846s** best | 61% | **0.753** best |
| depth 6, λ=5 | 0.5170s | 81% | 0.707 |
| depth 8, λ=1 | 0.5814s worst | **132%** | 0.658 |

**The best per-lap MAE gives the worst magnitude fidelity.** The objectives genuinely conflict,
and structurally so: most lap-to-lap variation inside a stint is noise, so MAE is minimised by
shrinking predictions toward zero. A flat curve is *rewarded* by the metric.

This matters because **Phase 3 integrates this curve over a whole stint.** A model
under-predicting degradation by ~40% would tell the simulator tires last far longer than they do,
biasing every strategy toward fewer stops.

**The transferable lesson:** the metric you optimise determines what you get. MAE on individual
laps was the natural choice and is the wrong objective for this downstream use. Nothing in the
training output would have revealed it — only checking the model against an independent
measurement did. This is exactly why the plan insisted on building Stage A first: without an
independent measurement there is nothing to notice the problem against.

### Residual bias by tire age

```
 tire age      n   mean signed residual    MAE
   [0, 5)   2378          -0.176s        0.612s
  [5, 10)   4615          -0.122s        0.478s
 [10, 15)   4197          -0.051s        0.408s
 [15, 20)   3453          +0.097s        0.434s
 [20, 25)   2523          +0.175s        0.473s
 [25, 30)   1631          +0.173s        0.531s
[40, 100)    748          -0.349s        0.827s
```

Systematic, not random: the model over-predicts loss on fresh tires, under-predicts mid-life, and
over-predicts badly on very old tires. MAE alone would hide this because it discards the sign.

---

## Two gotchas handled, both of which fail silently

**1. XGBoost categorical models must be saved as JSON.** The docs state categorical information
is lost otherwise. `joblib.dump` appears to work — no exception, a file is written, it loads back
— and then mispredicts. `app/ml/registry.py` dispatches on `ml.model_versions.backend`:
`save_model()` JSON for XGBoost, joblib for sklearn.

**2. An unseen category is a hard error, not a fallback.** Predict for a circuit added after
training, or a rookie driver, and XGBoost raises. `app/ml/categories.py` pins the vocabulary at
train time, persists it beside the artifact, and maps anything unseen to NaN — which XGBoost
handles natively. The response reports the count so a caller can warn rather than silently
receive a global-average curve.

---

## A bug worth recording: the non-monotonic curve

The first working API returned a curve that went **negative** at 5 laps of tire age — a tire
apparently getting faster, despite `monotone_constraints`.

The constraint was fine. The curve was wrong. `predict_curve` advanced `lap_number` alongside
`tyre_life`, and a monotonic constraint binds `tyre_life` **holding other features fixed**, so an
unconstrained `lap_number` moving at the same time pulled the prediction back down.

`lap_number` earns its place as a feature (3rd most important; dropping it costs 0.6% MAE)
because it absorbs residual fuel and track-evolution effects. So the fix was to the **curve
semantics**, not the feature set: pin `lap_number` at a reference lap (default: the stint
midpoint) and sweep only `tyre_life`. That yields the *partial* effect of tire age, which is what
"degradation curve" means and what Phase 3 needs — the simulator applies its own fuel model, so a
curve already containing fuel effects would double-count.

The cost, stated plainly: the ends of the curve sit slightly outside the training distribution,
since 25 laps of tire age never occurs at a mid-stint lap number.

---

## Known limitations

1. **Weather is session-level, not per-lap.** `raw.laps` stores only durations, never a
   session-time offset, so there is no key to join a lap to a weather sample. Track temperature
   moves 5–10 °C across a race, so this is a real loss of resolution. Fixing it means storing
   FastF1's `LapStartTime` and re-ingesting.
2. **2018 is excluded, deliberately.** It used `SOFT / SUPERSOFT / ULTRASOFT / HYPERSOFT`; 2019+
   uses `HARD / MEDIUM / SOFT`. In 2018 "SOFT" was the *hardest* of three; from 2019 it is the
   *softest*. The same label means the opposite thing, so including it would corrupt the compound
   feature. The Phase 1 CHECK constraint caught this by rejecting the ingest.
3. **Degradation magnitudes are under-predicted by ~40%** (see above). Use Stage A's measured
   table, or calibrate, before feeding Phase 3.
4. **`alpha=1.0` for Ridge is an untuned default.** Deliberate — Stage B exists as an honest
   reference, not a contender.

---

## Verification

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -m app.ml.tire.train
.venv/bin/python -m pytest tests/ -q
.venv/bin/uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/api/v1/tire/model-info | python -m json.tool
curl -s -X POST localhost:8000/api/v1/tire/degradation \
  -H 'content-type: application/json' \
  -d '{"circuit":"Sakhir","compound":"SOFT","max_tyre_life":25}'
```

Check: the curve starts at exactly 0, never decreases, `degradation_rate_s_per_lap` is positive,
and `unseen_categories` is empty for a known circuit.

Then look at the data directly, which is where the API hides things:

```sql
SELECT season_year, count(*) AS laps, count(DISTINCT session_id) AS races
FROM features.stint_laps GROUP BY 1 ORDER BY 1;
```

The notebook is part of the verification. If you cannot explain the collinearity argument from
its plots, the phase isn't finished.

---

## What Phase 3 inherits

- `features.stint_laps` — 104k clean dry green-flag race laps across 6 seasons
- A validated fuel model (`app/ml/fuel.py`) the simulator should reuse rather than reinvent
- Stage A's measured `(circuit, compound)` degradation table — **probably the better input for
  the simulator than the pooled model**, per the finding above
- `raw.pit_stops` with clamped durations, for estimating per-circuit pit loss
- A registry that will hold Phase 3's artifacts unchanged

And one explicit warning: **do not feed the raw pooled prediction into a simulator that
integrates it over 30 laps.** It is systematically ~40% flat. Calibrate it, or use the measured
table.
