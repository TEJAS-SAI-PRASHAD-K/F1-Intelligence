# Phase 3 — Pit Stop Strategy Simulator

## What this phase is for

Decide how many times to stop, and when. Unlike Phases 1 and 2 there is **no model training** —
this phase consumes Phase 2's degradation rates and turns them into a decision.

That shift, from prediction to decision, brings its own ideas: closed-form optimisation,
constrained search, Monte Carlo with common random numbers, and the point at which expected value
stops being the right objective.

**The lesson of this phase:** before you simulate, look for the closed form. One line of algebra
gave three structural results, a derivable threshold, a 2.4-million-candidate exhaustive search,
and a test oracle for the simulator.

## What was built

| Component | File | Does |
|---|---|---|
| Empirical views | `alembic/.../1a3149c26012_*.py` | `features.pit_loss`, `features.race_conditions` |
| Circuit merge | `alembic/.../819c4ea913e1_*.py` | Repairs duplicate circuit rows |
| Inputs | `app/ml/strategy/inputs.py` | Assembles the six numbers a circuit needs |
| Enumeration | `app/ml/strategy/enumerate_strategies.py` | Every legal strategy, as a generator |
| Simulator | `app/ml/strategy/simulator.py` | Closed form + lap-by-lap + ranking |
| Monte Carlo | `app/ml/strategy/monte_carlo.py` | Safety cars, pit variance, paired comparison |
| API | `app/api/routes/strategy.py` | 3 endpoints |
| Tests | `tests/test_strategy_{simulator,monte_carlo,api}.py` | 71 tests |
| Notebook | `notebooks/03_pit_strategy.ipynb` | The whole argument |

---

## The whole problem, in one line

A lap on a tire `age` laps old at race lap `n`:

```
lap_time = base_pace + degradation x (age-1) + fuel_penalty(n)
```

Sum the degradation term over a stint of length L and it is an arithmetic series, so the only
**strategy-dependent** part of a race time is:

```
cost(strategy) = SUM over stints of  slope(compound) * L*(L-1)/2   +   stops * pit_loss
```

Base pace and fuel are identical for every strategy at a given race distance and cancel.

**Three results follow immediately, before running anything:**

| # | Consequence | Why |
|---|---|---|
| 1 | Stint **order** cannot affect the deterministic cost | the formula has no ordering term |
| 2 | Long stints are punished **quadratically** | `L*(L-1)/2` is convex, so even splits win |
| 3 | An extra stop costs a **fixed** pit loss and buys a **quadratic** saving | that trade is the whole game |

Consequence 1 is not a quirk to work around. It is why `rank_strategies` deduplicates plans, and
why stint order is only compared in the Monte Carlo — where a safety car at a specific lap breaks
the symmetry.

### The threshold is derivable

For a 50-lap race at one degradation rate `s`:

- 1-stop splits 25/25 → `s * 2 * (25*24/2)` = `s * 600`
- 2-stop splits 17/17/16 → `s * (136+136+120)` = `s * 392`

So a second stop **saves** `208*s` seconds and **costs** one pit loss. With a 22s pit lane it pays
when `s > 22/208 = 0.1058 s/lap`.

Brute force agrees:

```
   slope    best      1-stop      2-stop      3-stop
  0.0200       1       +0.0s      +17.8s      +37.8s
  0.0800       1       +0.0s       +5.4s      +19.0s
  0.1005       1       +0.0s       +1.1s      +12.7s   <- just below
  0.1111       2       +1.1s       +0.0s      +10.4s   <- just above
  0.2000       2      +19.6s       +0.0s       +1.2s
  0.4000       3      +80.8s      +19.6s       +0.0s
```

The flip lands exactly where the algebra says. That is not a coincidence to note and move past —
it is how we know the search is correct rather than merely plausible.

---

## Two implementations, on purpose

`deterministic_cost` evaluates the closed form in O(stints). `simulate_lap_by_lap` walks every
lap in O(laps) and is the extensible one — it can take a safety car, per-lap noise, or traffic.

They agree to **2.8×10⁻⁹ ms**, and `tests/test_strategy_simulator.py` asserts it. Two independent
implementations of one quantity is the strongest test available short of real-world validation.

The closed form is also what makes exhaustive search practical. Monaco has 78 laps and near-zero
degradation, so stints up to 45 laps are legal — **2,461,326 legal strategies**, of which 2.43
million are 3-stop.

### A performance bug worth recording

The first version built a `Strategy` dataclass per candidate and took **17.3 seconds** at Monaco.
Essentially all of it was object construction for candidates about to be discarded.

Restructuring `iter_candidates` to yield plain `(compounds, lengths)` tuples, with objects
materialised only for the top few, cut that to **0.66 seconds** — 26× faster, same exhaustive
search. The lesson is ordinary but easy to forget: the combinatorics were never the problem.

---

## The empirical inputs

Six numbers fully determine a circuit's simulation, and every recommendation follows from them.
`GET /api/v1/strategy/inputs/{circuit}` exposes them for exactly that reason.

### Pit loss: the definition matters more than the measurement

`raw.pit_stops.duration_ms_est` is the time from the pit **entry** line to the pit **exit** line.
That is not what a stop costs, because it omits the braking before and the acceleration after —
it understates pit loss by 2–3s.

What a strategist needs:

```
pit_loss = (in_lap + out_lap) - 2 x the driver's normal green-flag pace
```

That captures deceleration, pit lane transit, the stationary stop and acceleration back to speed.
Validation: **median 23.3s across 2,715 green-flag stops**, inside the 18–25s range quoted for
modern F1.

Measured range: **17.5s (Hockenheim) to 35.0s (Monaco)**.

### The filter lesson, for the third time

| phase | filters that looked sufficient | what still got through |
|---|---|---|
| 1 | `is_accurate` + strictly green | a 148s standing-restart lap |
| 2 | + dry compounds only | 20s laps from a drying track |
| 3 | the pit-loss formula above | safety-car stops |

A stop made under the safety car has a slow in-lap **by definition**, so the subtraction blames
the safety car's cost on the pit stop:

| subset | n | median | p90 |
|---|---|---|---|
| all stops | 3,658 | 24.6s | **71.9s** |
| in-lap AND out-lap strictly green | 2,715 | 23.3s | 32.0s |

The median barely moves (24.6 → 23.3) while p90 halves. **Looking only at the median would have
hidden this completely** — which is also exactly why the median is the right statistic to *use*.

Three phases in, the pattern is clear: the assumption that fails is never the one you were
thinking about. What caught all three was the same habit — look at the distribution, not the
summary.

### Safety car rates

From 121 races: **68.6% have at least one SC/VSC lap**, mean 5.1 SC laps, 14.9% have a red flag.
Per circuit, Jeddah, São Paulo, Baku and Mexico City all sit at 100% — street and high-incident
circuits, which matches their reputation.

### Degradation comes from Stage A, not the Phase 2 model

Phase 2 established that the pooled model predicts a median of only ~68% of the rate Stage A
measures, compressing the spread across circuits ~4×. Fine for predicting one lap; ruinous here,
because this simulator **integrates** the curve over ~30 laps. A rate 40% too flat makes long
stints look cheap and biases every strategy toward fewer stops.

So `DegradationSource.STAGE_A` is the default. `DegradationSource.MODEL` is available for
comparison and the notebook shows what it does.

---

## A data bug fixed along the way

FastF1 reports `Location` inconsistently across seasons, producing two rows for one physical
track:

- **"Monte Carlo"** (2019, 2021) and **"Monaco"** (2022–2024)
- **"Singapore"** (2019) and **"Marina Bay"** (2023–2024)

That fragments every per-circuit statistic. Monaco — where strategy matters most, because
overtaking is close to impossible — had its 62 green-flag stops split into samples of 34 and 28,
giving **two different answers for one pit lane: 35.0s and 21.1s**. Merged, the honest figure from
all 62 stops is **22.4s**.

The fix is in two places: `transform.CIRCUIT_NAME_ALIASES` canonicalises new ingests, and a
migration repairs the stored rows. Hockenheim/Nürburgring, Monza/Imola/Mugello and
Austin/Miami/Las Vegas share a country but are genuinely different tracks, so only true duplicates
are merged.

`circuit` is a feature of the Phase 2 model, so it was retrained afterwards (XGBoost 0.4885s →
0.4877s holdout MAE — a marginal improvement).

---

## Results, checked against reality

The strongest available test: does it reproduce strategies that actually happened?

| circuit | laps | pit loss | mean wear | best strategy | real-world |
|---|---|---|---|---|---|
| Monaco | 78 | 22.4s | 0.010 s/lap | **1-stop** | always a 1-stop |
| Monza | 53 | 25.9s | 0.052 s/lap | **1-stop** | Leclerc won 2024 on a 1-stop |
| Marina Bay | 62 | 29.0s | 0.029 s/lap | **1-stop** | |
| Sakhir | 57 | 24.5s | 0.112 s/lap | **2-stop** | classically a 2-stop |

Sakhir's optimum is a near-even **18/19/20** split — exactly what the convexity of `L(L-1)/2`
predicts.

---

## Monte Carlo, and the technique that makes it work

Three sources of randomness: pit loss variance (~±2.5s, from the per-circuit interdecile range),
lap noise (sd ~0.6s), and safety cars.

### Common random numbers

Scenarios are drawn **once** and every strategy is evaluated against the **same** ones. This
matters far more than it sounds.

At Sakhir:

| quantity | value |
|---|---|
| sd of **one strategy's** race time | **131.1s** |
| sd of the **margin** between two strategies | **8.7s** |

A 15× reduction, from the same runs. The 131s is almost entirely the safety-car penalty that every
strategy pays identically; subtracting per scenario cancels it.

**This changes the conclusion, not just the precision.** Quote the single-strategy sd and the
strategies look indistinguishable. Quote the paired margin and the 2-stop wins **78%** of
scenarios. Same data, opposite reading. It is why `StrategyOut` reports `margin_sd_s` alongside
`mc_sd_s`, with a docstring saying which to read.

### The safety car matters through exactly one channel

Worth being precise about, because it is easy to over-model. When a safety car is deployed on lap
*k* for *d* laps, every car runs those laps slowly — so **every strategy pays the same penalty and
it cancels out of any comparison**.

The ranking changes through one channel only: a stop inside the window costs far less. That is the
"free pit stop", and isolating it requires holding the strategy fixed and moving only the safety
car:

```
strategy M20 / H37, stops on lap 20
  SC on laps 18-22 (covers the stop): 5588.21s
  SC on laps 45-49 (misses it)      : 5601.69s
  saving                            : 13.48s  = pit_loss x (1 - 0.45)
```

Both scenarios have a 5-lap safety car, so the lap-time penalty cancels and the whole difference
is the discounted stop.

**A test bug worth recording.** The first version of this test compared two *different*
strategies, S20/H30 against S10/H40. They differ in degradation by 7.05s as well as in safety-car
luck, so the observed 5.05s gap was 12.10s of pit saving minus 7.05s of degradation and matched
neither number. The code was right; the test confounded two variables.

### More stops = more lottery tickets

At Sakhir, SC luck rate rises 14% → 26% → 35% from 1 to 3 stops. Each extra stop is another chance
to land under a safety car, so the Monte Carlo gap is consistently **smaller** than the
deterministic gap. The lottery partly pays for the extra stop — about 0.3s at Sakhir, 1.7–2.3s at
Monza.

---

## Expected value is not the whole decision

Everything above ranks by median race time, which answers "which strategy is fastest on average".
That is **not always the question**.

At Sakhir, P(gain at least X seconds on the median-best strategy):

| strategy | +0s | +5s | +10s | +15s |
|---|---|---|---|---|
| 1-stop | 21.8% | 13.2% | 8.1% | **1.1%** |
| 2-stop | *reference (best median)* | | | |
| 3-stop | 21.9% | 3.8% | 0.1% | **0.0%** |

The 2-stop has the best median and wins most scenarios outright. But if you are **15 seconds
behind**, and second place pays the same whether you lose by 2s or 20s, the 2-stop has
essentially no path and the 1-stop retains one — because its upside depends on a safety car
falling its way.

Being behind changes the objective from "minimise expected time" to "maximise P(gain > deficit)".
Same simulation output, different question, different answer.

---

## How much do the assumptions matter?

`SC_PIT_LOSS_FACTOR = 0.45` is the least defensible number in the model — an approximation, not a
measurement. The honest response is not to defend it but to vary it:

| SC factor | 1-stop | 2-stop | 3-stop |
|---|---|---|---|
| 0.25 | +2.5s | **+0.0s** | +11.1s |
| 0.45 | +2.5s | **+0.0s** | +11.1s |
| 0.65 | +2.5s | **+0.0s** | +11.2s |
| 1.00 | +2.4s | **+0.0s** | +12.1s |

Stable across the whole range, so the recommendation does not rest on that constant. Had it
flipped, the right move would have been to report the answer as uncertain —
`sensitivity_to_sc_factor` exists to make that check routine rather than optional.

---

## Known limitations

These are stated on every API response (`strategy.LIMITATIONS`), because a user who does not know
them will over-trust the numbers.

1. **Single-car model — the biggest one.** No competitors, so no track position, traffic, undercut
   or overcut. Strategies are ranked by **race time, not finishing position**, and where
   overtaking is hard those differ. Monaco is the extreme case: the simulator says 1-stop is 19s
   faster than 2-stop, but in a real Monaco race track position can outweigh 19s entirely.
   Fixing this needs a multi-car simulation.
2. **Linear degradation.** Real tires can fall off a cliff; this under-penalises very long stints.
   Phase 2 found degradation is close to linear once fuel-corrected, which makes the approximation
   defensible but not exact.
3. **Safety cars affect the ranking only through pit timing.** A real safety car also bunches the
   field and changes overtaking opportunities, which a single-car model cannot represent.
4. **Dry races only.** Wet-compound stints are excluded from the degradation data.
5. **Stint limits come from what teams actually did**, so they encode past strategic choices, not
   just tire physics. A compound rarely run long at a circuit gets a low ceiling even if it could
   have gone further.

---

## Verification

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -m pytest tests/ -q          # 220 passed
.venv/bin/uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/api/v1/strategy/inputs/Sakhir | python -m json.tool

curl -s -X POST localhost:8000/api/v1/strategy/simulate \
  -H 'content-type: application/json' \
  -d '{"circuit":"Sakhir","max_stops":3,"n_scenarios":2000}'
```

Check: Sakhir picks a 2-stop, Monza and Monaco pick 1-stop, every returned strategy's stint
lengths sum to the race distance and use at least two compounds, and `margin_sd_s` is far smaller
than `mc_sd_s`.

The notebook is part of the verification. If you cannot explain from it why stint order does not
affect the deterministic cost, the phase isn't finished.

---

## What Phase 4 inherits

- `features.pit_loss` and `features.race_conditions` — per-circuit pit loss and safety-car rates,
  both useful as race-prediction features
- Canonical circuit names, so per-circuit aggregates are no longer fragmented
- A reusable Monte Carlo pattern: Phase 4 converts finishing-position predictions into per-position
  probabilities, which is the same shape of problem
- The paired-comparison discipline, which applies directly to comparing models on the same races

And a warning that carries forward: Phase 4's features will include things measured from past
races, so the as-of discipline from the plan matters more there than anywhere else.
