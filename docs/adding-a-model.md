# Adding a model

The procedure is given in full in `skills/convert-model/SKILL.md`, which is written to be
followed by an AI coding agent or by a person. In outline:

1. Read the original model and identify its inputs, tables, monthly recursion and aggregates.
2. Record the original's intermediate values on a sample portfolio and an edge-case portfolio.
3. Write the kernel as a step function in the restricted form, with a `KernelSpec`.
4. Obtain exact agreement on the processor backend.
5. Measure the tests: exhaustive table checks, branch coverage, seeded defects.
6. Write the symbolic description and obtain the proof.
7. Build the GPU and single-precision variants and compute rounding bounds.
8. Export the proof obligations to Lean.

`examples/lifelib_savings/` is the reference example. Each example directory contains
`model.py` (everything specific to the model), `run.py` (one command per stage), `results/`
(the figures quoted in this repository) and `REPORT.md`.

## The restricted form

The step function may use arithmetic, comparisons, and the following operations.

| Operation | Meaning |
|---|---|
| `sel(c, a, b)` | `a` if `c` holds, otherwise `b` |
| `fmax`, `fmin`, `imax`, `imin` | Larger or smaller of two real values or integers |
| `band(a, b)` | Logical conjunction |
| `idiv(a, k)` | Integer floor division by a positive constant |
| `tof(a)` | Conversion of an integer to a real value |
| `lk1(table, i)`, `lk2(table, i, j)` | Table lookup; an index outside the table gives zero |
| `fpow`, `fexp`, `flog` | Power, exponential and logarithm, where the argument depends on carried state |

Powers, exponentials and logarithms that depend only on inputs or on time are computed once,
outside the step function, by the same expression as in the original. This is both faster and
necessary for exact agreement, since different mathematical libraries may round these functions
differently.
