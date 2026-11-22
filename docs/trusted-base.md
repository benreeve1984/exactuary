# Dependencies of the evidence

Each check supports a claim. The claim holds only if the items in the third column are correct.
The [glossary](../README.md#glossary) explains the terms on this page.

| Check | Claim | Dependencies |
|---|---|---|
| Exact comparison | The fast model reproduces every recorded value of the original model on the tested rows | The original model as run; `exactuary/oracle.py` and `exactuary/compare.py`; each example's choice of values to record, periods, and input mapping. Using the result for other workloads also requires a representative portfolio |
| Planted errors | The comparison detects a recorded proportion of planted errors | `exactuary/mutate.py` and `exactuary/harness.py`. The count describes the planted errors. Applying it to real conversion errors requires evidence that those errors are representative. Planted errors are one family among many (see [Limitations](limitations.md)) |
| Proof (Z3) | For all policies and periods, in exact arithmetic, each covered output equals the original model's, as the symbolic interpreter reads the original model | Z3; `exactuary/symbolic.py`, including its reading of Python, NumPy, and pandas; the formal description, with its assumptions about parts of the original model (the code calls these "axioms"), its definitions of precomputed tables and their keys, its other stated assumptions, and its relation between carried values and the original model; input preparation |
| Proof checking (Lean) | Each exported statement is a theorem, and its proof uses no axiom beyond Lean's 3 standard ones (Lean's built-in logical assumptions) | Lean's checking program; the translation in `exactuary/lean.py`, which must keep the meaning of each Z3 statement; the program that checks each theorem for axioms. That program accepts only the line forms in the generated files, and it works from a fixed list instead of Lean's own record of declarations. Also everything in the row above except Z3, because the symbolic interpreter still produces the statements from the same description |
| Early exit | Early exit loses nothing, in exact arithmetic | As for proof. The backends run an exit rule, and the statements come from that rule |
| Guaranteed interval | For each row in the run, the exact fast-model result on the prepared inputs lies in a stated interval | `exactuary/interval.py` and `exactuary/lang.py`. Tests compare them with exact fraction arithmetic, and they have no formal correctness proof. Also the standard for computer arithmetic (IEEE-754) as NumPy applies it on the machine that computes the interval, and `mpmath` for powers, exponentials, and logarithms |

## Checks on the fast model

The checks assess the fast-model source, whoever wrote it. The parser (the code reader) accepts
only the allowed syntax in `exactuary/lang.py`, for both the step function and its exit rule.

Every library entry point for a fast model checks the whole specification first. These are the
CPU, GPU, and NumPy backends, the proof, the interval check, and the estimator. The parser rejects
text outside the allowed syntax. The interpreters evaluate accepted expressions, or regenerate
code for compilation. Regenerated code can use only the helper operations and, in the generated
loops, `range`.

This protects against accidents. No tests have tried to make it run other code, and you should not
rely on it against a hostile author. Its correctness depends on the parser and on about 150 lines
of checking code.

## Additional dependencies

These dependencies are easy to overlook:

- The backends. Exact comparison and the interval check test the compiled CPU and GPU code on the
  tested rows. Claims about untested rows assume that the compiled code follows the source. The
  proof concerns the source formulas, and tests assess the compiled program.
- The formal description. The same author can write both the formal description and the fast
  model. If the description is wrong in a way that matches a wrong fast model, the proof can be
  valid under its stated assumptions while those assumptions describe the original model
  incorrectly. Review the description as code in its own right. The library rejects some known
  forms of this error, such as a table definition that ignores a position, but it cannot detect
  them all.
- Input preparation. This code turns a model point file and assumption tables into the fast
  model's inputs. Tests compare prepared inputs with the original model's inputs. Proof coverage
  excludes input preparation.
- Assumptions. Each one replaces part of the original model with a statement taken on trust. Each
  model's README lists its assumptions.
- Assembly. Lean checks the general induction theorem. No program checks that each model's
  statements fit it, so a reviewer must check that connection.
