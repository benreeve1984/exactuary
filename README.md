# Exactuary

Exactuary is an experimental toolkit that makes actuarial cashflow models run faster. It also
keeps evidence that the faster version is still the same model.

It takes a model written in Python, in the style of [lifelib](https://lifelib.io) and modelx. It
rewrites the model as a small compiled function. This documentation calls that function "the fast
model", and the code calls it the "kernel". Exactuary then checks the fast model against "the
original model", meaning the lifelib model run unchanged. The checks come in several strengths.

Exactuary is a research prototype. We have applied it to 4 open-source teaching models. It is
not a validated production tool.

**[Read the illustrated explanation](https://benreeve1984.github.io/exactuary/)**. It is written
for readers who are not specialists. It explains how cashflow models can be made faster, what
it takes to achieve each speed-up, and how theorem provers help.

## Approach

You do not need to trust whoever writes the fast model. In the intended use, an AI coding agent
writes it. Instead, you trust a short set of checks and the original model
itself. The original model is kept and run as the reference. Compiler research calls this
arrangement "translation validation".

The fast model is one short function. It describes one period of one policy. A period is one
projection step, such as a month or a year. The function is written in a restricted language.
Because the language is restricted, the same text can be:

- compiled for a processor
- compiled for a graphics processor (GPU)
- analysed as mathematical formulas rather than numbers
- run in interval arithmetic, where each value is a range that is guaranteed to contain the exact
  answer

## Levels of acceleration

Each level changes the model further and needs stronger evidence. Level 0 is the model run one
policy at a time. `docs/levels.md` defines the levels and explains how to choose
between them.

| Level | What changes | Evidence |
|---|---|---|
| 1. Vectorised | All policies are processed together | Totals are reconciled |
| 2. Compiled | Each policy is projected in machine code | Every intermediate value is compared exactly with the original model |
| 3. All processor cores (parallel) | The compiled model runs on all processor cores | As level 2 |
| 4. Restructured | The projection stops once nothing is in force, and the arithmetic is rearranged | Proof that the results are equal in exact arithmetic, re-checked in Lean |
| 5. GPU, double precision | The same arithmetic runs on a graphics processor | Proof, and a guaranteed interval that contains the exact result |
| 6. GPU, single precision | Numbers are held to about 7 significant digits | Guaranteed interval, and comparison at the level at which results are used |

## Results on the example models

We converted 4 lifelib models:

- a universal-life savings model with stochastic returns (`CashValue_ME`)
- the same model with dynamic lapses (`CashValue_ME_EX2`)
- a term assurance model (`BasicTerm_ME`)
- an annual model of term, whole-life and endowment business (`fastlife`)

All figures come from one machine with 20 Arm cores and an NVIDIA GB10.

### Agreement

Every intermediate value of every model, on every test portfolio, equals the original model's
value. About 1.6 billion values were compared, and none differed.

### Speed

The table shows seconds per million rows. A row is one policy in one scenario. Times for the fast
model were measured on 10 million rows. The time for the original model is scaled up from a
smaller run.

| Version | Savings | Savings, dynamic lapse | Term life | Annual |
|---|---|---|---|---|
| Original model, as lifelib ships it (vectorised) | 302 | 32 | 14.1 | 3,235 |
| Compiled, one core | 7.9 | 2.63 | 1.67 | 1.06 |
| Compiled, one core, stopping early | 3.66 | 2.65 | 0.68 | 0.43 |
| As above, on all 20 cores | 0.31 | 0.24 | 0.055 | 0.043 |
| GPU, double precision | 0.15 | 0.11 | 0.035 | 0.038 |
| GPU, single precision | 0.013 | 0.0036 | 0.0030 | 0.0039 |

lifelib also has versions of these models that work one policy at a time (`CashValue_SE`,
`BasicTerm_SE` and `simplelife`). They take between 14,000 and 160,000 seconds per million rows.

### Accuracy when results are not exact

For every row of the test portfolios (69,200 rows), we computed a guaranteed interval. The
interval contains the exact result of the fast model on the prepared inputs. The table shows the
greatest distance from the exact result, relative to the size of the value.

| Model | Original model and GPU double precision | GPU single precision | GPU single precision, net cashflow |
|---|---|---|---|
| Savings | 1.4e-7 | 8.7e-3 | 1.9e-2 |
| Savings, dynamic lapse | 6.4e-6 | 9.2e-3 | 5.5e-4 |
| Term life | 4.8e-14 | 1.6e-6 | 4.8e-3 |
| Annual | 3.8e-9 | 2.2e-6 | 6.7e-6 |

### Strength of the tests

We planted errors in each fast model, one small edit at a time. Reconciling portfolio totals
detects 58% to 72% of these planted errors. Comparing every intermediate value detects 95% to
99%. For the remainder, the proofs either show that the changed fast model differs from the
original model, or show that the change makes no difference.

### Proof

The proofs use 2 tools:

- Z3, an automated theorem prover from Microsoft Research that either proves a statement or finds
  a counterexample
- Lean, a proof assistant that checks every step of a proof with a small, well-studied checking
  program

For each model, Z3 proves that every output in every period equals the original model's output in
exact arithmetic. Lean then re-checks the proof. The proof depends on a stated description of the
original model written as formulas.

### Conclusions

Most of the gain comes from compiling and using all cores. At those levels, results match the
original model exactly.

A graphics processor in double precision adds little on this machine, which has few
double-precision units.

Single precision is 10 or more times faster again. The cost in accuracy depends on the product.
For an individual policy, the error can reach 1 or 2 per cent.

## Limitations

The checks are designed to catch errors made in good faith during a conversion. They do not
defend against someone who deliberately tries to get a wrong model to pass. The proofs concern
exact arithmetic and a description of the original model as formulas. That description must
itself be reviewed. The examples are teaching models. See `docs/trusted-base.md` and
`docs/limitations.md`.

The checks themselves have been through adversarial testing. The `tests/` folder contains wrong
fast models, wrong descriptions of the original model and false statements that earlier versions
accepted. The checks must now reject or disprove each of them.

## Contents

| Path | Contents |
|---|---|
| `exactuary/` | The library: the kernel language (`lang.py`, `spec.py`), numeric backends, exact comparison, planted errors, proof by formulas, export to Lean, guaranteed intervals, acceptance gates |
| `examples/` | The 4 converted models, each with a README, its results and its run script |
| `skills/convert-model/` | The written procedure an AI coding agent follows to convert a model |
| `benchmarks/` | Timing and certification of all 4 models, with results |
| `tests/` | Tests of the checks themselves, including adversarial cases |
| `docs/` | Levels, method, what must be trusted, how to add a model, limitations |
| `site/` | The illustrated explanation published with GitHub Pages; `benchmarks/build_site_data.py` regenerates its figures |

## Running it

You need:

- Python 3.12, NumPy, Numba and z3-solver
- lifelib, modelx and pandas for the examples
- `numba-cuda` and an NVIDIA GPU for the GPU backend
- `mpmath` for guaranteed intervals of powers (the code calls these intervals "enclosures")
- Lean 4.36 for proof checking (set `LEAN` to its path)

To install and run:

    pip install -e ".[examples,gpu]"
    python tests/test_interval.py
    python tests/test_counterexamples.py

    cd examples/lifelib_basicterm
    python run.py oracle      # run the original and record every intermediate value
    python run.py exact       # compare the kernel with it; time both
    python run.py harness     # exhaustive table checks, coverage, seeded defects
    python run.py prove       # proof of equivalence, and of early exit
    python run.py lean        # re-check the proofs in Lean

    python benchmarks/timings.py     # all models, all levels
    python benchmarks/certify.py     # interval enclosures against every variant

The commands `exact`, `harness`, `prove` and `lean` each exit with a non-zero status if a required
check fails. To convert another model, see `docs/adding-a-model.md` and the skill.

## Licence

MIT. The example models belong to the lifelib project. Exactuary uses them through lifelib's
published package and does not redistribute them.
