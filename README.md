# Exactuary

Exactuary is an experimental toolkit for faster actuarial cashflow models. It also checks, and
where possible proves, that an optimised model gives the same results as the original.

A person or an AI coding agent rewrites the original model as a small function for one projection
period of one policy. Exactuary compiles that function to machine code and checks
its results against the unchanged original model. The original models come from
[lifelib](https://lifelib.io) and use modelx, a Python tool for expressing models as formulas.
The [glossary](#glossary) explains the terms these pages use.

Exactuary is a research prototype. The examples cover four open source lifelib models. It is not a
validated production tool.

[Read the illustrated explanation](https://benreeve1984.github.io/exactuary/). It describes how
faster execution works, what each level gains and costs, and how theorem provers help. It is for
readers without a software background.

## What is different

Some platforms now let insurers run analyst-readable code, such as Python or R, on GPUs. That
gives some speed. In our tests a GPU also brought rounding differences, extra checks, and a wait
for hardware.

AI coding agents can now rewrite a model as low-level code quickly. That opens another route.
Compile the model, run it on ordinary processor cores, and use mathematical proofs that the new
code equals the original to support model validation. Exactuary tests that route.

On the savings example, 64 rented processor cores project 100 million rows in 12.5 seconds, for
about one US cent. lifelib's version would take about 7.8 hours on a desktop computer. The
recorded values are identical on the test portfolios. See
[Results on other hardware](docs/hardware.md).

## The idea

A person or an AI coding agent can write the fast model. The checks assess that model against the
original model. Exactuary runs the unchanged original model as its reference. A reviewer must
separately check the formal description and assumptions, even if the same author supplied them.
Compiler research calls this arrangement translation validation.

The fast model uses a step function for one period of one policy. Execution code repeats the step
through the projection. The author writes the step function in a restricted language, so
Exactuary can use the same text in four ways:

- compile it for a processor
- compile it for a graphics processor (GPU)
- evaluate it with symbols instead of numbers, for proof
- evaluate it in interval arithmetic, to bound rounding error

## Levels of acceleration

Each level changes the model further and needs more evidence.
[Levels of acceleration](docs/levels.md) defines the levels and explains how to choose between
them.

| Level | What changes | Evidence |
|---|---|---|
| 0. One policy at a time | lifelib's separate model processes one policy at a time, where such a model exists | Baseline checks against the level 1 model |
| 1. Vectorised | The unchanged original model processes policies together. It is the comparison reference | Baseline checks against the separate level 0 model, where available |
| 2. Compiled | Machine code projects each policy on one core | Compare every recorded intermediate value. Present values agree when both models add them in the same order |
| 3. All processor cores (parallel) | The compiled model runs on all cores | As level 2 |
| 4. Restructured | Early exit stops a projection once nothing is in force. The arithmetic may also change order | Equality under stated assumptions in exact arithmetic, a Lean check, and rounding checks where rounding changes |
| 5. GPU, double precision | The same arithmetic runs on a GPU, with about 16 significant digits | As level 4, with rounding error bounds |
| 6. GPU, single precision | The number format keeps about 7 significant digits | Rounding error bounds, and accuracy checks on each reported output |

## Results on the example models

The examples convert four lifelib models:

- Savings: universal-life savings with stochastic returns (`CashValue_ME`)
- Savings with dynamic lapses (`CashValue_ME_EX2`)
- Term assurance (`BasicTerm_ME`)
- Traditional life, annual: term, whole-life, and endowment business in yearly periods (`fastlife`)

The timing machine has 20 Arm processor cores and an NVIDIA GB10.

### Agreement

The comparison covers every recorded value of the selected outputs on the test portfolios. It
compares about 1.6 billion values and finds no differences. The annual example excludes 20 cells
that fail in the original model. [Its README](examples/lifelib_fastlife/README.md) lists those
exclusions.

At levels 2 and 3, all recorded intermediate values agree on the test portfolios. Present values
also agree when both models add them in the same order. The timed versions use running totals,
which can change the last digits.

### Speed

The table gives seconds per million rows, from `benchmarks/results/timings.json`.

| Version | Savings | Savings with dynamic lapses | Term assurance | Traditional life, annual |
|---|---|---|---|---|
| Original model, vectorised, as lifelib supplies it | 279 | 32 | 13.1 | 3,179 |
| Compiled, one core | 7.73 | 2.64 | 1.65 | 1.10 |
| Compiled, one core, early exit | 3.60 | 2.64 | 0.67 | 0.44 |
| Compiled, 20 cores, early exit | 0.32 | 0.24 | 0.054 | 0.044 |
| GPU, double precision | 0.15 | 0.11 | 0.035 | 0.038 |
| GPU, single precision | 0.013 | 0.0036 | 0.0030 | 0.0039 |

The timing workloads have 1,141 monthly periods for savings, 121 monthly periods for dynamic
savings (configuration 5), 277 monthly periods for term assurance, and 103 yearly periods for the
annual model.

Fast-model times are the best of two single-core runs or five parallel or GPU runs after warm-up, on
about ten million rows. The original model runs once on a smaller workload, and the benchmark
scales its time up linearly. These are projection-stage comparisons at different portfolio sizes.
The dynamic original timing also includes simulation summaries. The fast-model timing excludes
them. The figures do not measure a complete reporting workflow.

Fast-model times exclude input preparation, compilation, device transfer, and verification. Input
preparation takes about 1.76 seconds for the 10-million-row savings run. The separate interval
check takes about 97.5 seconds for 40,000 savings rows. The saved results do not separately
measure file input and output.

lifelib's separate level 0 models (`CashValue_SE`, `BasicTerm_SE` and `simplelife`) take between
14,000 and 160,000 seconds per million rows.

### Rounding error bounds

The bounds cover 69,200 rows in the certification runs: 40,000 savings, 9,000 dynamic savings,
10,000 term assurance and 10,200 annual rows. The dynamic savings run uses configuration 5. A
separate check computes an interval for each row. The interval contains the exact fast-model
result on the prepared inputs, provided the interval program is correct. The check reports any row
without a bound.

Each figure is the largest upper bound for the selected accumulated outputs. Net cashflow has a
separate column. The check divides each bound by the greater of 1 and the absolute midpoint of
its interval. The figures are upper bounds and do not measure the actual errors. The notation
1.4e-7 means 1.4 parts in ten million. For example, a bound of 1.9e-2 on a net cashflow of 1,000
places the computed value within 19 of the exact fast-model result, in the same currency.

| Model | Fast-model values with original summation order | GPU double precision | GPU single precision | GPU single precision, net cashflow |
|---|---|---|---|---|
| Savings | 1.4e-7 | 1.4e-7 | 8.7e-3 | 1.9e-2 |
| Savings with dynamic lapses | 6.4e-6 | 6.4e-6 | 9.2e-3 | 5.5e-4 |
| Term assurance | 4.9e-14 | 4.8e-14 | 1.6e-6 | 4.8e-3 |
| Traditional life, annual | 3.8e-9 | 3.8e-9 | 2.2e-6 | 6.7e-6 |

The first column reconstructs the original summation from the fast model's recorded values.
Separate tests compare those values with lifelib.

### Strength of the tests

Exactuary plants errors in each fast model, one small edit at a time. Reconciling portfolio totals
detects 58% to 72% of them. Comparing every recorded intermediate value detects 95% to 99%.

Proof handles most of the rest. For the annual model, 14 planted errors survive comparison. Proof
finds a difference for three and establishes equality in exact arithmetic for 9. The symbolic check
rejects two because it cannot process their table indices.

### Proof

For each covered output, Z3 proves equality in exact arithmetic under the model's stated
assumptions. Exactuary exports the statements to Lean, which checks separate proofs. Review must
confirm that the assumptions describe the original model and that the statements establish
agreement over the full projection. Tests still check the programs and their rounded results.

### Results summary

Most of the gain comes from compiling and using all cores.

A GPU in double precision adds little on this machine, which has few double-precision units.

On this machine, GPU single precision is about 9.6 to 29 times as fast as GPU double precision,
depending on the model. Single precision can materially change individual results. On the
certification workloads, the GPU net cashflow bounds are about 1.95% for savings and 0.055% for
dynamic savings. Separate comparison runs record differences from the original of up to 14.2%.
These figures cover different workloads and measures.

## Scope of the evidence

The checks aim to detect errors made in good faith during a conversion. They do not defend against
someone who deliberately seeks a pass for a wrong model. The proofs concern exact arithmetic and a
formal description of the original model, and a reviewer must check that description. The
examples are open source lifelib models, not production models. See [Dependencies of the evidence](docs/trusted-base.md) and
[Limitations](docs/limitations.md).

Tests in `tests/` include deliberately wrong models, descriptions, and proof statements. The checks
reject the known failure cases. The formal description and interval program still need review.
The tests do not establish complete protection against conversion errors.

## Contents

| Path | Contents |
|---|---|
| `exactuary/` | The library: the step-function language (`lang.py`, `spec.py`), backends, exact comparison, planted errors, symbolic proof, export to Lean, guaranteed intervals, acceptance gates |
| `examples/` | The 4 converted models, each with a README, its results, and its run script |
| `skills/convert-model/` | The conversion procedure for a person or an AI coding agent |
| `benchmarks/` | Timing and rounding error bounds for all 4 models, with results |
| `tests/` | Tests of the checks themselves, including adversarial cases |
| `docs/` | Levels, method, dependencies of the evidence, adding a model, results on other hardware, limitations |
| `site/` | The illustrated explanation on GitHub Pages; `benchmarks/build_site_data.py` regenerates its figures |

## Running the examples

You need:

- Python 3.12, NumPy, Numba, and z3-solver
- lifelib, modelx, and pandas for the examples
- `numba-cuda` and an NVIDIA GPU for the GPU backend
- `mpmath` for guaranteed intervals of powers
- Lean 4.36 for proof checking (set `LEAN` to its path)

To install and run:

    pip install -e ".[examples,gpu]"
    python tests/test_interval.py
    python tests/test_counterexamples.py

    cd examples/lifelib_basicterm
    python run.py oracle      # run the original and record the selected intermediate values
    python run.py exact       # compare the kernel with it; time both
    python run.py harness     # exhaustive table checks, coverage, seeded defects
    python run.py prove       # proof of equivalence, and of early exit
    python run.py lean        # re-check the proofs in Lean

    cd ../..
    python benchmarks/timings.py     # all models, all levels
    python benchmarks/certify.py     # interval enclosures against every variant

Return to the repository root with `cd ../..` before you run the benchmark commands. First prepare
the original-model data for every selected example.

The `exact`, `harness`, `prove` and `lean` commands each exit with a non-zero status if a required
check fails. To convert another model, see [Adding a model](docs/adding-a-model.md) and the
[model conversion procedure](skills/convert-model/SKILL.md).

## Glossary

- Original model: the lifelib model, run unchanged. This is level 1, the comparison reference.
- Fast model: the rewritten version. It uses a step function for one period of one policy, and
  execution code repeats the step through the projection. The code calls the step function a
  "kernel" or "step". `KernelSpec` declares the fast model's inputs, outputs, and carried values.
- Row: one policy in one scenario.
- Period: one projection step, such as a month or a year.
- Vectorised: calculating values for many policies together.
- Compiled: translated into machine code, the instructions a processor runs directly.
- Processor core: a part of the processor that can execute a stream of instructions.
- Double precision: a number format with about 16 significant digits. Single precision has about 7.
- Backend: the code that runs the fast model on a particular device.
- Early exit: stopping a projection once nothing is in force.
- Planted error: a small change to the fast model. The tests check whether the comparison detects
  it. The code calls these "seeded defects".
- Branch coverage: a record of whether the tests exercise both outcomes of each choice.
- Formal description: the description of the original model as formulas, with its assumptions,
  table definitions and input mapping.
- Assumption: a statement that replaces part of the original model in the proof. The code calls
  these "axioms".
- Statement: something to prove. The code calls it a "proof obligation".
- Exact arithmetic: arithmetic without computer rounding.
- Z3: an automated theorem prover from Microsoft Research. It searches for a proof or an example
  that contradicts the statement. It may also report that it cannot decide. Any example needs
  review against the model's assumptions.
- Lean: a proof assistant. Its small, well-studied checking program checks every step of a proof.
- Guaranteed interval: a pair of numbers that contains the exact fast-model result on the prepared
  inputs, provided the interval program is correct. The code calls this an "enclosure".
- Acceptance gate: a record of whether a required check passed.
- Adversarial testing: tests that deliberately try to make the checks accept a wrong model.
- NaN: "not a number", the computer value for an undefined result.

## Licence

MIT. The example models belong to the lifelib project. Exactuary uses them through lifelib's
published package and does not redistribute them.
