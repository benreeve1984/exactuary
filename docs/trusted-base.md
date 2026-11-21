# What must be trusted

For each form of evidence, the claim it supports holds provided the following are correct.

| Evidence | Claim | What must be trusted |
|---|---|---|
| Exact comparison | The kernel reproduces every recorded value of the original on the test portfolios | The original model as executed; `exactuary/oracle.py` and `exactuary/compare.py`; the example's choice of which cells to record, over which periods, and its mapping of inputs; that the portfolios are representative |
| Measured tests | A stated proportion of single-edit defects is detected | `exactuary/mutate.py` and `exactuary/harness.py`; that single-edit defects resemble real errors (they are one family among many: see Limitations) |
| Proof (Z3) | For all policies and periods, in exact arithmetic, each output equals the original's, as the original is read by the symbolic interpreter | Z3; `exactuary/symbolic.py`, including its reading of Python, NumPy and pandas; the model's symbolic description: its axioms, its definitions of precomputed tables and their keys, its stated assumptions, its carried-state relation; input preparation |
| Proof checking (Lean) | Each exported statement is a theorem whose proof depends on no axiom beyond Lean's three standard ones | Lean's kernel; the translation in `exactuary/lean.py` (which must preserve the meaning of each Z3 statement); the wrapper's audit of each theorem's axioms, which accepts only the line forms the generated files use and is built from a list, not from Lean's own record of declarations; and everything in the row above except Z3, because the statements are still produced by the symbolic interpreter from the same description |
| Early exit | Stopping loses nothing, in exact arithmetic | As for proof; the obligations are generated from the exit rule the backends execute |
| Enclosure | For each row run, the exact result of the kernel on the prepared inputs lies in a stated interval | `exactuary/interval.py` and `exactuary/lang.py` (tested against exact rational arithmetic, not formally verified); IEEE-754 double arithmetic in NumPy on the machine that computes the enclosure; `mpmath` for powers, exponentials and logarithms |

What is not trusted: the kernel's source and the party that wrote it. The kernel and its exit
rule are confined to a small grammar (`exactuary/lang.py`). The entry points of the library that
accept a kernel (the CPU, GPU and NumPy backends, the proof, the enclosure and the estimator)
validate the whole specification first; text outside the grammar is rejected, and accepted text is either
interpreted node by node or recompiled from its parsed form with access only to the helper
operations (and, in the generated loops, `range`). This is protection against accident. It has
not been tested by attempts to execute code, and should not be relied on against a hostile author. The
correctness of that confinement rests on the parser and on about 150 lines of checking code.

What is trusted but easily overlooked:

- **The backends.** The compiled CPU and GPU code is validated by exact comparison and by
  enclosure on the rows tested. For rows not tested, its faithfulness to the kernel's text is
  assumed. The proof concerns the text, not the machine code.
- **The symbolic description.** It is written by the same party as the kernel. A description
  that is wrong in a way that matches a wrong kernel yields a valid proof of a false claim. It
  must be reviewed as code in its own right; the library refuses some known forms of this error
  (a table definition that ignores an index) but cannot detect them all.
- **Input preparation.** The code that converts a model point file and assumption tables into
  the kernel's inputs is tested (its outputs are compared with the original's) but not proved.
- **Axioms and assumptions.** Each replaces a part of the original with a statement taken on
  trust. Those for each example are listed in its report.
- **Assembly.** The general induction theorem is proved in Lean; that each model's obligations
  instantiate it is not machine-checked.
