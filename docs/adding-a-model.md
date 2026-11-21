# Adding a model

The [model conversion procedure](../skills/convert-model/SKILL.md) gives the full procedure. A
person or an AI coding agent can follow it. The [glossary](../README.md#glossary) explains the
terms on this page.

In outline:

1. Read the original model. Identify its inputs, tables, totals and how each period follows from
   the one before.
2. Record the original model's intermediate values on a sample portfolio and on a portfolio of
   edge cases.
3. Write the step function in the restricted form, with a `KernelSpec`. The `KernelSpec` declares
   the fast model's inputs, outputs and carried values.
4. Get exact agreement on the processor backend.
5. Measure test performance. Check every table position, record both outcomes of each choice
   (branch coverage), and test with planted errors.
6. Write the formal description, and get the proof from Z3.
7. Export the proof statements to Lean, and check them.
8. Build the GPU and single-precision versions, and compute rounding error bounds.

[`examples/lifelib_savings/`](../examples/lifelib_savings/) is the worked conversion example. Each
example folder contains:

- `model.py`, with everything specific to the model
- `run.py`, with one command per stage
- `results/`, with the saved result figures
- `README.md`, with the model description, results and limitations

## The restricted form

The step function may use arithmetic, comparisons and these operations:

| Operation | Meaning |
|---|---|
| `sel(c, a, b)` | `a` if `c` is true, otherwise `b` |
| `fmax`, `fmin`, `imax`, `imin` | The larger or smaller of 2 real values or 2 integers |
| `band(a, b)` | Logical "and" |
| `idiv(a, k)` | Integer division by a positive constant. The result rounds down |
| `tof(a)` | Converts an integer to a real value |
| `lk1(table, i)`, `lk2(table, i, j)` | Table lookup. A position outside the table gives 0 |
| `fpow`, `fexp`, `flog` | Power, exponential and logarithm, where the argument depends on carried values |

Some powers, exponentials and logarithms depend only on inputs or on time. Compute these once,
outside the step function, with the same expression as the original model. This is faster. It is
also necessary for exact agreement, because different mathematical libraries may round these
functions differently.
