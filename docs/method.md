# Method

## Scope: errors, not adversaries

The checks described here are designed to catch mistakes made in good faith by whoever converts a
model, whether a person or an AI coding agent: a wrong table row, a reordered operation, a
missing branch, a formula misread. The tests in `tests/` record the defects of this kind that
the checks are known to catch. A party deliberately trying to obtain a pass for a wrong model, by
writing unusual code or by editing the symbolic description or the proof files, is not defended
against, and adversarial testing found ways to do so.
The symbolic description of a model, and anything a person edits by hand, are therefore to be
reviewed as work products. They are not untrusted input that the tools make safe.

## Principle

The accelerated model is written by an untrusted party: a person or, as intended here, an AI
coding agent. The accelerated model itself is not accepted on its author's authority: acceptance
is decided by a small set of checks whose code is short enough to be read. One part of the
author's work is, however, trusted by the proofs: the symbolic description that says how the
original is to be read (its axioms, its table definitions and keys, its mapping of inputs). A
wrong description can produce a valid proof of a false claim, so it must be reviewed in its own
right. Exact comparison does not depend on it. This arrangement is known in compiler research as translation
validation: the translator is not verified; each translation is.

## The kernel

The accelerated model, here called the kernel, is a single function describing one month of
one policy. It receives the policy's data, the month's inputs, and the small amount of state
carried from the previous month, and returns that month's values and the state for the next
month. It is written in a deliberately restricted form: straight-line arithmetic, with choices
expressed as `sel(condition, a, b)`, and no loops or branches.

The restriction permits the same text to be interpreted in several ways:

| Interpretation | Purpose |
|---|---|
| Compiled for a processor (Numba) | Fast execution; exact comparison with the original |
| Compiled for a graphics processor (CUDA) | Faster execution |
| Array arithmetic (NumPy) | Test measurement |
| Symbolic (Z3) | Proof of equivalence |
| Interval arithmetic with outward rounding | An interval containing the exact result |

Because every interpretation is derived from one text, there is no separate GPU model, proof
model or test model to write and keep in step. The interpreters themselves must still agree on
what the text means. To that end the text is confined to a small grammar (`exactuary/lang.py`)
and is never executed as written: the interval and symbolic interpreters evaluate the parsed
text node by node, and the numeric backends compile text regenerated from the parsed form.

## The original model as oracle

For testing, the original model is executed as written, with every intermediate cell recorded.

For proof, the aim is likewise not to transcribe it: its source is executed symbolically, the
same formulas being run on symbolic values instead of numbers, yielding an expression for each
cell in terms of the policy data. Two qualifications apply. First, the original's formulas are
not run by Python: they are parsed and evaluated node by node by an interpreter written to accept
a listed set of constructs and to refuse others. That list has been wrong before: testing
found constructs that were accepted and misread, and the specific inputs found were then
refused; related forms may still be accepted. The
interpreter should be regarded as tested, not as proved complete. Objects that the symbolic
description itself supplies (axioms, stand-ins for other modules) are trusted code and are
called as they are. Cells outside that subset (inputs, table lookups implemented with data-frame operations,
random draws, and in one example a reserve formula that was transcribed by hand) are replaced
by stated assumptions, called axioms, each of which is separately tested by enumeration.
Secondly, the interpreter treats every value as a single real number or integer: it does not
model missing values, infinities, index alignment or integer overflow. The proof is therefore a
proof about a reading of the original, and that reading is part of what must be trusted.

Where the symbolic definition of a precomputed table does not depend on one of its indices (as
when a per-policy assumption is described by a single symbol), the description must state which
kernel input that index has to equal. A lookup with any other index is then treated as an
unknown value. A declared key is always enforced. A definition that does not depend on an index
and declares nothing is refused, by a check that is a safeguard against oversight, not a proof
that the description is right: a description that states the wrong key, or wrongly declares a
table constant, is a wrong description, and the proof will follow it.

## The proof

For each output the prover is asked two questions.

1. In month 0, does the kernel's value equal the original's, for every possible policy?
2. In any later month, if the state carried by the kernel is what the original implies it
   should be, does the kernel's value equal the original's, and is the state it carries
   forward again correct?

If both hold for every output, equality in every month follows by induction. The induction
itself and the equality of present values as running sums are proved once, in general form, in
Lean (`exactuary/lean/Glue.lean`). That a particular model's obligations are instances of the
hypotheses of those general theorems is not itself machine-checked: it rests on the exporter
and on the correspondence between the description's "state after month t" and the theorem's
"state entering month t+1".

## Stopping early

Where a kernel stops a projection once a stated rule holds (for example, no policies remain in
force), the justification is generated from the rule that the compiled code actually executes.
For month 0 and for any later month, and for any state whatever, the prover must show that if
the rule holds after a month then in the next month every accumulated cashflow is zero and the
rule holds again. In addition the stopped projection is compared, value for value, with the
full projection on every test portfolio. The proof is over real numbers: where the original
itself produces a non-finite value after the stopping point, the two differ, and the affected
rows are reported.

## Rounding

A proof of equality over the real numbers does not establish equality of computed results,
because computed arithmetic rounds. Two cases arise. Where the kernel is written to perform the
original's operations in the original's order, computed results are expected to be identical,
and exact comparison shows that they are on the portfolios tested. Where operations are reordered, or a different device or precision is used,
the kernel is evaluated once more in interval arithmetic: every value is carried as a pair of
numbers known to lie either side of the exact result, each operation rounding the lower one down
and the upper one up. A comparison that the intervals cannot decide keeps both outcomes. The
result, for each policy run, is an interval that contains what exact arithmetic would give on
the same inputs, provided the enclosure program is correct. Every number in the step, including
integer arithmetic, the time index and divisions of constants, passes through that program, so
that nothing is rounded before it is tracked; inputs it cannot represent exactly are refused. Any computed result held in
single or double precision, from any device, can then be
placed against it: its error is at most its distance to the interval's far end. Nothing need be
assumed about how that device rounds.

This is a statement about each policy actually run, made after the run. It is not a bound
valid for every possible input, and it takes the prepared inputs (including any powers
precomputed into tables) as given. `benchmarks/certify.py` applies it to every example.
