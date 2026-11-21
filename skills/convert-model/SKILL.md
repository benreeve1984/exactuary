---
name: convert-model
description: Convert an actuarial cashflow model (modelx / lifelib style Python) into a fast kernel with Exactuary and establish, level by level, that the fast model is the same model. Use when asked to accelerate, port, vectorise or GPU-enable a cashflow projection model and to evidence equivalence.
---

# Converting a model with Exactuary

## Your role

You write the fast version of the model. You are **not trusted**: nothing you produce is accepted
until the checks in the `exactuary` library accept it. Therefore:

- Never edit anything under `exactuary/` to make a check pass, and never weaken a comparison
  (no tolerances where the check is exact, no dropping of outputs, no reduced portfolios).
- Never alter the original model. It is the oracle. Change only its inputs (model points,
  scenarios), and only through its own data objects.
- If the library cannot express something the model needs, stop and report exactly what is
  missing. Do not work around it inside the checker.
- Report what happened, including failures and anything left unproved.

The worked example in `examples/lifelib_savings/` (`model.py`, `run.py`) is the template. Read both
files and `exactuary/spec.py` before starting. Create `examples/<your_model>/` with the same layout.

"Period" means one projection step: a month in a monthly model, a year in an annual one. The
library's messages say "month"; read that as "period".

## Step 1. Understand the original

Read every formula. Classify each cell as: a per-policy input; a function of time only; a table
lookup; a per-month value; or a whole-projection aggregate (sums over months, present values).
Identify the recursion: which month-t cells refer to month t-1 (these define the carried state) and
which refer to month t+1. Note every place where the code branches on `t == 0`.

Evaluate every cell of the original once before relying on it. Cells that raise an error in the
original are out of scope: list them in the report with the error, and do not repair them.

## Step 2. Capture the oracle

Write `load_original`, at least two portfolios, and `run_original` as in the template:

- a **sample** portfolio from the model's own sample data;
- an **edge** portfolio generated to reach unusual cases (already in force, future-dated,
  beyond term, zero and extreme amounts, ages at and beyond the ends of each table). Include
  cases where the original itself returns NaN or infinity; the kernel must reproduce those too.
  If the sample data is too uniform to exercise table indexing, say so and rely on the edge data.

You may also vary the model's assumption tables and scenarios through its own data objects, to
reach behaviour the shipped assumptions never trigger. State in the report which were varied.

Capture **every** per-month cell with `exactuary.oracle.capture`, not only the reported results.
Record the original's run time on a workload large enough to time reliably.

## Step 3. Write the kernel

Write one `step` function as source text, in the restricted style described in
`exactuary/spec.py`: straight-line code, helper operations only, no Python `if` or loops.
Describe it with a `KernelSpec`. Guidance:

- **State.** Carry the fewest values that month t needs from month t-1. Month 0 is handled with
  `sel(is_t0, initial_value, carried_value)`.
- **Same operations, same order.** To be bit-identical the kernel must perform the original's
  floating-point operations in the original's order. `a - b - c` is `(a - b) - c`. Python's
  `sum()` starts from 0. A boolean multiplied by a number becomes `sel(cond, ONE, ZERO) * x`
  (not `sel(cond, x, ZERO)`: false times NaN is NaN, and the original produces NaN).
  pandas `.where(cond, other=0)` becomes `sel(cond, x, ZERO)`.
- **Integers.** Where the original computes in integers (counts at period 0, durations), the
  kernel's reals agree only below 2^53. Assert the input range in `prep`.
- **Division** follows NumPy: dividing by zero gives infinity or NaN and does not raise.
- **Hoist, do not re-derive.** Powers, exponentials and logarithms that depend only on inputs
  or on time belong in `prep`, computed with the *same expression and the same NumPy call* as the
  original, and passed to the step as tables or series. Do not compute them inside the step.
- **State-dependent powers.** Where a power, exponential or logarithm depends on carried state
  (for example a lapse rate driven by the account value) it cannot be hoisted; use `fpow`, `fexp`
  or `flog`. These may differ from NumPy in the last bit. If the exact gate then fails only
  downstream of such a call, do not hide it: report which outputs are exact, the size of the
  difference in the others, and treat those outputs as "bounded" rather than "exact".
- **Table lookups** use `lk1` / `lk2`; indices outside the table give zero, matching
  `reindex(..., fill_value=0)`. If the original behaves differently off-table (NaN, an error,
  clamping), handle that explicitly and test it. Off-table behaviour lies outside the proof.
- **Per-policy values that the step cannot express** (rounding with `np.around`, lookups that
  return NaN off-table) may be computed in `prep` with the original's own expression, provided
  each is compared exactly with the original's cell on every portfolio and enumerated in Step 5.
- **Aggregates.** List in `accumulate` the outputs whose discounted sums are reported. If the
  original sums with a matrix product, a running sum will differ in the last bits; compare both
  ways, as the template does.
- **Multi-pass models.** If an input to the projection is itself derived from a projection
  (for example a premium priced from present values), run a kernel for the first pass, derive
  the quantity in `prep` with the original's own expression, then run the second pass.
- **Early exit** (`exit_when`) is optional and must be justified by a proved lemma (Step 6).
  The lemma is over real numbers. Where the original yields NaN or infinity after the exit point
  (for example 0 x infinity), early exit returns a finite value instead; test for this and report
  the affected rows.
- **Backward recursions** (a value at t defined from t+1) cannot be written in the step.
  Compute them from the kernel's trace with the original's expression and compare exactly; they
  are then tested, not proved. Say so.

## Step 4. Level "exact": bit-identical on CPU

Compile with `exactuary.backends.numba_cpu.build` and compare the full trace with the oracle
using `exactuary.compare.compare_trace` on both portfolios. The gate is **zero differing values**
in every output. When values differ, `compare_trace` reports the first (month, row); inspect
that policy, find the first output that differs, and fix the kernel. Typical causes: operation
order, integer versus real arithmetic, off-table lookups, month-0 handling.

Then time the kernel against the original on the same workload: one core and all cores, with and
without early exit.

## Step 5. Level "measured": how good is the checking

- Check every table lookup exhaustively, in and out of range, against the original's own lookup
  cells, using a synthetic portfolio that enumerates the index space.
- `exactuary.harness.coverage` on each portfolio; extend the edge portfolio until every branch
  outcome is exercised or you can explain why one is unreachable.
- `exactuary.harness.mutation_scores` with a `swaps` map of confusable variable names.
  Report the share of seeded defects each check catches and list the survivors.

## Step 6. Level "proved": all policies, all months

Write a `SymbolicSpec` (see `exactuary/symbolic.py` and the template's `symbolic_spec`):

- `setup` declares the symbolic model point and installs **axioms** for cells that cannot be
  executed symbolically (inputs, pandas lookups, random draws). Keep axioms to a minimum; each one
  is an assumption a reader must accept, and each must be covered by a test in Step 5.
- `carry` states, in terms of the original's cells at month t, what the kernel should carry out
  of month t. `invariants` states any fact about the original alone that the proof needs.
- `tables` states how each hoisted table is defined in terms of the original's quantities.
- `assumptions` states any fact about inputs or tables that the proof needs (for example that a
  divisor is never zero). Each is reported alongside the axioms and must be tested.
- One proof covers one configuration of the model's switches. Prove each configuration in use.

Run `exactuary.symbolic.prove`. The gate is every obligation proved, none refuted, none unknown.
A refutation comes with a counterexample policy: run it through Step 4 to see whether the kernel or
the spec is wrong. Then run `refute_mutants` on the seeded defects; any defect that is *not* refuted
points to a weak state relation or an over-strong axiom. If you use early exit, state and prove the
lemma that justifies it (`lemmas`).

Real literals are treated as exact rationals. Folding constants in floating point (writing
`0.9991666` for `1 - 0.01/12`) changes the real-number meaning and will be refuted.

## Step 7. Level "bounded": GPU and single precision

Build the GPU kernel with `exactuary.backends.cuda.build`, and float32 variants with
`exactuary.spec.to_f32_src`. For each variant report: time, worst per-policy difference and
portfolio-total difference from the original, and the rounding bound from
`exactuary.errbound.run_pv_bounds`. State plainly whether single precision is acceptable per
policy, at portfolio level only, or not at all.

## Step 8. Level "kernel-checked": Lean

Export the obligations with `exactuary.lean.export` and check them with `run_lean`, together with
`exactuary/lean/Glue.lean`. Report how many are checked and which are not.

## Step 9. Report

Write `examples/<your_model>/REPORT.md`: what the model is; results at each level with numbers;
the axioms; what is not covered; every place the library or this skill was insufficient or
unclear, and what you did about it.

## Rules the library enforces

The library refuses work that breaks them.

- **The step and the exit rule** must be in the kernel language (`exactuary/lang.py`): no
  annotations, defaults, attribute access or calls other than the helpers. They are parsed, never
  executed as written.
- **The original's formulas** are read by an interpreter that accepts a listed subset of Python
  (see the `Original` class in `exactuary/symbolic.py`). If it refuses a cell, replace that cell
  with an axiom, state it in the report, and test it by enumeration. Do not edit the original.
- **Constants are exact.** `1/12` means one twelfth. In a symbolic description write
  `z3.Q(1, 12)` or `Fraction(1, 12)`, not a Python float.
- **Table keys.** If a table's symbolic definition does not depend on one of its indices (for
  example a per-policy assumption described by one symbol), declare in `table_keys` which kernel
  input that index must equal, or `None` if the table really is constant along it. A `None`
  declaration is an assumption: list it and test it.
- **Gates.** Record every acceptance check with `exactuary.gates.record` and list, per command,
  the gates that must have run. A check that is only printed is not a check.
- **The symbolic description is reviewed work.** It is trusted by the proofs. Keep it small,
  and say in the report exactly what it assumes.

## Practical rules

- Shared machine: check free memory before any parallel run, keep mutation workers at 6 or
  fewer, and keep full traces to a few gigabytes. A trace is outputs x months x rows x 8 bytes.
- Run commands that may exceed two minutes in the background.
- Save results as JSON under `results/`; large data goes under `data/`, which is not committed.
