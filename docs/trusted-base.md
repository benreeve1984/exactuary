# What must be trusted

Each form of evidence supports a claim. The claim holds only if certain things are correct. This
document lists those things.

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls it the "kernel". A row is one policy in one
scenario. A period is one projection step, such as a month or a year.

The proofs use 2 tools:

- Z3, an automated theorem prover from Microsoft Research that either proves a statement or finds
  a counterexample
- Lean, a proof assistant that checks every step of a proof with a small, well-studied checking
  program

| Evidence | Claim | What must be trusted |
|---|---|---|
| Exact comparison | The kernel reproduces every recorded value of the original model on the test portfolios | The original model as run; `exactuary/oracle.py` and `exactuary/compare.py`; the example's choice of which cells to record, over which periods, and how it maps inputs; that the portfolios are representative |
| Measured tests | A stated proportion of planted errors is detected. Each planted error is a single small edit | `exactuary/mutate.py` and `exactuary/harness.py`; that planted errors resemble real errors (they are one family among many: see Limitations) |
| Proof (Z3) | For all policies and periods, in exact arithmetic, each output equals the original model's, as the formula interpreter reads the original model | Z3; `exactuary/symbolic.py`, including how it reads Python, NumPy and pandas; the model's description as formulas, that is, its assumptions about parts of the original model (the code calls these "axioms"), its definitions of precomputed tables and their keys, its other stated assumptions and how it relates the carried state to the original model; input preparation |
| Proof checking (Lean) | Each exported statement is a theorem whose proof depends on no axiom beyond Lean's 3 standard ones | Lean's kernel (its core checking program); the translation in `exactuary/lean.py`, which must keep the meaning of each Z3 statement; the wrapper's audit of each theorem's axioms, which accepts only the line forms in the generated files and is built from a list, not from Lean's own record of declarations; everything in the row above except Z3, because the formula interpreter still produces the statements from the same description |
| Early exit | Stopping early loses nothing, in exact arithmetic | As for proof. The backends run an exit rule, and the statements to be proved come from that rule |
| Guaranteed interval (the code calls this an "enclosure") | For each row run, the exact result of the kernel on the prepared inputs lies in a stated interval | `exactuary/interval.py` and `exactuary/lang.py`, which are tested against exact arithmetic with fractions but not formally proved correct; IEEE-754 double arithmetic in NumPy on the machine that computes the interval; `mpmath` for powers, exponentials and logarithms |

## What is not trusted

The kernel's source code is not trusted, and neither is whoever wrote it.

The kernel and its exit rule are limited to a small grammar (`exactuary/lang.py`). Several parts
of the library accept a kernel: the CPU, GPU and NumPy backends, the proof, the interval and the
estimator. Each of these checks the whole specification first. Text outside the grammar is
rejected. Accepted text is either worked through one piece at a time, or recompiled from its
parsed form. Recompiled text can use only the helper operations and, in the generated loops,
`range`.

This protects against accidents. It has not been tested against attempts to run other code, and
you should not rely on it against a hostile author. Whether the limits work rests on the parser
and on about 150 lines of checking code.

## What is trusted but easy to overlook

They are:

- The backends. Exact comparison and the guaranteed interval check the compiled CPU and GPU code
  on the rows tested. For rows not tested, it is assumed that the code is faithful to the
  kernel's text. The proof concerns the text, not the machine code.
- The description of the original model as formulas. The same person or agent writes both
  this description and the kernel. If the description is wrong in a way that matches a wrong kernel, the result
  is a valid proof of a false claim. You must review it as code in its own right. The library
  rejects some known forms of this error, such as a table definition that ignores an index, but
  it cannot detect them all.
- Input preparation. This is the code that turns a model point file and assumption tables into
  the kernel's inputs. It is tested, by comparing its outputs with the original model's, but not
  proved.
- Assumptions. Each one replaces a part of the original model with a statement taken on trust.
  Each example lists its assumptions in its report.
- Assembly. The general induction theorem is proved in Lean. No machine checks that each model's
  statements fit it.
