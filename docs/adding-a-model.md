# Adding a model

The full procedure is in `skills/convert-model/SKILL.md`. It is written so that either an AI
coding agent or a person can follow it.

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls the fast model the "kernel", and its main function
the "step function".

In outline, you:

1. Read the original model and identify its inputs, tables, totals and how
   each month follows from the one before.
2. Record the original model's intermediate values on a sample portfolio and on a portfolio of
   edge cases.
3. Write the kernel as a step function in the restricted form, with a `KernelSpec`.
4. Get exact agreement on the processor backend.
5. Measure how strong the tests are, using exhaustive table checks, branch coverage and planted
   errors.
6. Write the description of the original model as formulas, and get the proof from Z3. Z3 is an
   automated theorem prover from Microsoft Research that either proves a statement or finds a
   counterexample.
7. Build the GPU and single-precision versions, and compute bounds on rounding error.
8. Export the statements to be proved (proof obligations) to Lean. Lean is a proof assistant that
   checks every step of a proof with a small, well-studied checking program.

`examples/lifelib_savings/` is the reference example. Each example folder contains:

- `model.py`, which holds everything specific to the model
- `run.py`, with one command per stage
- `results/`, the figures quoted in this repository
- `REPORT.md`

## The restricted form

The step function may use arithmetic, comparisons and the following operations.

| Operation | Meaning |
|---|---|
| `sel(c, a, b)` | `a` if `c` is true, otherwise `b` |
| `fmax`, `fmin`, `imax`, `imin` | The larger or smaller of 2 real values or 2 integers |
| `band(a, b)` | Logical "and" |
| `idiv(a, k)` | Integer division by a positive constant, rounding down |
| `tof(a)` | Converts an integer to a real value |
| `lk1(table, i)`, `lk2(table, i, j)` | Table lookup; an index outside the table gives 0 |
| `fpow`, `fexp`, `flog` | Power, exponential and logarithm, where the argument depends on carried state |

Some powers, exponentials and logarithms depend only on inputs or on time. Compute these once,
outside the step function, using the same expression as the original model. This is faster. It
is also needed for exact agreement, because different mathematical libraries may round these
functions differently.
