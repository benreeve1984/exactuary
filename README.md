# Exactuary

Exactuary is an experimental toolkit for making actuarial cashflow models run faster while
keeping evidence that the fast model is the same model. It converts a model written in Python
(in the style of [lifelib](https://lifelib.io) and modelx) into a small compiled kernel, and
checks the kernel against the original at several levels of strength.

It is a research prototype, applied to four open-source teaching models. It is not a validated
production tool.

**[Read the illustrated explanation](https://benreeve1984.github.io/exactuary/)**: how cashflow
models can be accelerated, what each step gains and costs, and how theorem provers help, written
for readers who are not specialists.

## The idea

The fast model is written by a party that is not trusted to be right: in the intended use, an
AI coding agent. What is trusted is a short set of checks, and the original model itself, which
is kept and run as the reference. This arrangement is known as translation validation.

The fast model is one short function describing one period of one policy, written in a
restricted language. Because the language is restricted, the same text can be compiled for a
processor, compiled for a graphics processor, analysed symbolically, and evaluated in interval
arithmetic.

## Levels of acceleration

Each level changes the model further and needs stronger evidence. `docs/levels.md` defines them
and discusses how to choose.

| Level | What changes | Evidence |
|---|---|---|
| 1. Vectorised | All policies processed together | Reconciliation |
| 2. Compiled | Each policy projected in machine code | Every intermediate value compared exactly with the original |
| 3. Parallel | All processor cores | As above |
| 4. Restructured | Projection stopped once nothing remains; algebra rearranged | Proof of equivalence in exact arithmetic, re-checked in Lean |
| 5. GPU, double precision | Same arithmetic on a graphics processor | Proof, and an interval enclosing the exact result |
| 6. GPU, single precision | About 7 significant digits | Enclosure; comparison at the level results are used |

## Results on the example models

Four lifelib models were converted: a universal-life savings model with stochastic returns
(`CashValue_ME`), the same with dynamic lapses (`CashValue_ME_EX2`), a term assurance model
(`BasicTerm_ME`), and an annual-step model of term, whole-life and endowment business
(`fastlife`). All figures were produced on one machine (20 Arm cores, NVIDIA GB10).

**Agreement.** Every intermediate value of every model, on every test portfolio, equals the
original's: about 1.6 billion values compared, none differing.

**Speed.** Seconds per million rows (a row is one policy in one scenario). Kernel times were
measured on ten million rows; the original's is scaled from a smaller run.

| Step | Savings | Savings, dynamic lapse | Term life | Annual |
|---|---|---|---|---|
| Original, as lifelib ships it (vectorised) | 302 | 32 | 14.1 | 3,235 |
| Compiled, one core | 7.9 | 2.63 | 1.67 | 1.06 |
| and stopping early | 3.66 | 2.65 | 0.68 | 0.43 |
| and all 20 cores | 0.31 | 0.24 | 0.055 | 0.043 |
| GPU, double precision | 0.15 | 0.11 | 0.035 | 0.038 |
| GPU, single precision | 0.013 | 0.0036 | 0.0030 | 0.0039 |

The same models written one policy at a time (lifelib's `CashValue_SE`, `BasicTerm_SE` and
`simplelife`) take between 14,000 and 160,000 seconds per million rows.

**Accuracy off the exact path.** For every row of the test portfolios (69,200 rows) an interval
was computed that contains the exact result of the kernel on the prepared inputs. Greatest
distance from the exact result, relative to the size of the value:

| Model | Original and GPU double precision | GPU single precision | GPU single, net cashflow |
|---|---|---|---|
| Savings | 1.4e-7 | 8.7e-3 | 1.9e-2 |
| Savings, dynamic lapse | 6.4e-6 | 9.2e-3 | 5.5e-4 |
| Term life | 4.8e-14 | 1.6e-6 | 4.8e-3 |
| Annual | 3.8e-9 | 2.2e-6 | 6.7e-6 |

**Strength of the tests.** Single-edit defects were seeded into each kernel. Reconciling
portfolio totals detects 58% to 72% of them. Comparing every intermediate value detects 95% to
99%; the proofs refute the remainder or show them equivalent.

**Proof.** For each model, every output in every period is proved equal to the original's in
exact arithmetic by Z3 and re-checked by Lean, relative to a stated symbolic reading of the
original.

Three observations follow. Most of the gain comes from compiling and using all cores, with
results that match the original exactly. A graphics processor in double precision adds little on
this machine, which has few double-precision units. Single precision adds a further factor of ten
or more, at a cost in accuracy that depends on the product and can reach one or two per cent for
an individual policy.

## What this does and does not establish

The checks are designed to catch errors made in good faith during a conversion. They are not a
defence against a party deliberately seeking a pass for a wrong model. The proofs concern exact
arithmetic and a symbolic description of the original, which must itself be reviewed. The
examples are teaching models. See `docs/trusted-base.md` and `docs/limitations.md`.

The checks themselves have been tested adversarially: `tests/` contains wrong kernels, wrong
readings of the original and false statements that earlier versions accepted, each of which must
now be refused or refuted.

## Contents

| Path | Contents |
|---|---|
| `exactuary/` | The library: the kernel language (`lang.py`, `spec.py`), numeric backends, exact comparison, seeded defects, symbolic proof, Lean export, interval enclosures, acceptance gates |
| `examples/` | The four converted models, each with a README, its results and its run script |
| `skills/convert-model/` | The written procedure an AI coding agent follows to convert a model |
| `benchmarks/` | Timing and certification of all four models, with results |
| `tests/` | Tests of the checks themselves, including adversarial cases |
| `docs/` | Levels, method, what must be trusted, how to add a model, limitations |
| `site/` | The illustrated explanation published with GitHub Pages; `benchmarks/build_site_data.py` regenerates its figures |

## Running it

Requires Python 3.12, NumPy, Numba and z3-solver; lifelib, modelx and pandas for the examples;
`numba-cuda` and an NVIDIA GPU for the GPU backend; `mpmath` for enclosures of powers;
Lean 4.36 for proof checking (set `LEAN` to its path).

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

Each of `exact`, `harness`, `prove` and `lean` exits with a non-zero status if a required check
fails. To convert a further model, see `docs/adding-a-model.md` and the skill.

## Licence

MIT. The example models belong to the lifelib project and are used through its published
package; they are not redistributed here.
