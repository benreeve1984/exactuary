# Method

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls the fast model the "kernel", and so does this
document from here on.

## Scope: errors, not deliberate attacks

The checks are designed to catch mistakes made in good faith by whoever converts a model. That
may be a person or an AI coding agent. Typical mistakes are a wrong table row, an operation in
the wrong order, a missing branch or a misread formula. The tests in `tests/` record mistakes of
this kind, and the checks are known to catch them.

The checks do not defend against someone who deliberately tries to get a wrong model to pass, for
example by writing unusual code or by editing the description of the model or the proof files.
Adversarial testing found ways to do this.

So you should review the description of a model, and any hand edits, as work products. Do not
treat them as untrusted input and assume the tools make them safe.

## Principle

A person or, as intended here, an AI coding agent writes the fast model, and you do not need to
trust them. It is not accepted on its author's word. A small set of checks, whose code is short
enough to read, decides whether to accept it.

The proofs do trust one part of the author's work. This is the description of the original model
as formulas, which says how the original model is to be read. It includes the model's
assumptions (the code calls these "axioms"), its table definitions and keys, and how it maps
inputs. A wrong description can produce a valid proof of a false claim. So you must review the
description in its own right. Exact comparison does not depend on it.

Compiler research calls this arrangement "translation validation". The translator is not proved
correct. Each translation is checked instead.

## The kernel

The kernel is a single function that describes one month of one policy. It receives:

- the policy's data
- the month's inputs
- the small amount of state carried from the previous month

It returns that month's values and the state for the next month.

The kernel is written in a deliberately restricted form. It is straight-line arithmetic, with no
loops or branches. Choices are written as `sel(condition, a, b)`.

Because of this restriction, the same text can be run in several ways:

| Interpretation | Purpose |
|---|---|
| Compiled for a processor (Numba) | Fast running; exact comparison with the original model |
| Compiled for a graphics processor (CUDA) | Faster running |
| Array arithmetic (NumPy) | Measuring how strong the tests are |
| As formulas (Z3) | Proof that the results are equal |
| Interval arithmetic with outward rounding | A guaranteed interval that contains the exact result |

Z3 is an automated theorem prover from Microsoft Research that either proves a statement or finds
a counterexample.

Every interpretation comes from one text. So there is no separate GPU model, proof model or test
model to write and keep in step. The interpreters must still agree on what the text means. To
help with this, the text is limited to a small grammar (`exactuary/lang.py`) and is never run as
written. The interval and formula interpreters work through the parsed text one piece at a time.
The numeric backends compile text that is regenerated from the parsed form.

## The original model as the reference

For testing, the original model is run as written, and every intermediate cell is recorded.

For proof, the original model is again not copied by hand. Instead, its source code is run on
symbols rather than numbers. This gives a formula for each cell in terms of the policy
data. There are 2 qualifications.

First, Python does not run the original model's formulas. An interpreter reads them and works
through them one piece at a time. The interpreter is written to accept a listed set of Python
constructs and to reject others. That list has been wrong before. Testing found constructs that
were accepted and misread. The specific inputs found are now rejected, but related forms may
still be accepted. You should treat the interpreter as tested, not as proved complete.

The description itself supplies some objects, such as assumptions and stand-ins for other
modules. These are trusted code, and they run as written.

Some cells are outside the accepted set. These are inputs, table lookups written with data-frame
operations, random draws and, in one example, a reserve formula that a person copied by hand. A stated
assumption replaces each of these. We test each assumption separately, case by case.

Secondly, the interpreter treats every value as a single real number or integer. It does not
model missing values, infinities, index alignment or integer overflow. So the proof is a proof
about one reading of the original model, and that reading is part of what must be trusted.

## Tables that ignore an index

Sometimes the description defines a precomputed table in a way that does not depend on one of its
indices. This happens, for example, when a single symbol stands for a per-policy assumption.
In that case, the description must state which kernel input that index has to equal. A lookup
with any other index is then treated as an unknown value. A declared key is always enforced.

If a definition does not depend on an index and declares nothing, the library rejects it. This
check is a safeguard against oversight. It is not a proof that the description is right. A
description that states the wrong key, or wrongly says a table is constant, is a wrong
description, and the proof will follow it.

## The proof

The proof is split into proof obligations. A proof obligation is a statement to be proved. For
each output, Z3 is asked to prove 2 statements.

1. In month 0, the kernel's value equals the original model's, for every possible policy.
2. In any later month, suppose the kernel's carried state matches the original model's. Then
   the kernel's value equals the original model's, and the kernel again carries forward the
   correct state.

If both statements hold for every output, then the results are equal in every month, by
induction.

Lean is a proof assistant that checks every step of a proof with a small, well-studied checking
program. Two general results are proved once in Lean, in `exactuary/lean/Glue.lean`:

- the induction itself
- that present values calculated as running totals are equal

No machine checks that a particular model's statements fit the conditions of those general
results. That rests on the exporter. It also rests on matching the description's "state after
month t" with the theorem's "state entering month t+1".

## Stopping early

A kernel may stop a projection once a stated rule holds, for example when no policies remain in
force. The compiled code runs this rule, and the statements that justify stopping early come from
that same rule.

For month 0 and for any later month, and for any state at all, Z3 must prove the following. If
the rule holds after a month, then in the next month every accumulated cashflow is 0 and the rule
holds again.

In addition, the stopped projection is compared, value for value, with the full projection on
every test portfolio.

The proof uses real numbers. Sometimes the original model itself produces a value that is not a
finite number after the stopping point. In that case the 2 projections differ, and the affected
rows are reported. A row is one policy in one scenario.

## Rounding

A proof of equality over real numbers does not show that computed results are equal. This is
because computer arithmetic rounds. There are 2 cases.

In the first case, the kernel performs the original model's operations in the original model's
order. The computed results are then expected to be identical. Exact comparison shows that they
are, on the portfolios tested.

In the second case, operations are reordered, or a different device or precision is used. The
kernel is then run once more in interval arithmetic. Every value is carried as a pair of numbers
known to lie either side of the exact result. Each operation rounds the lower number down and the
upper number up. Where the intervals cannot decide a comparison, both outcomes are kept.

The result, for each policy run, is a guaranteed interval. The code calls this an "enclosure". It
contains what exact arithmetic would give on the same inputs, provided the program that computes
it is correct.

Every number in the step passes through that program, including integer arithmetic, the time
index and divisions of constants. So nothing is rounded before it is tracked. If the program cannot
represent an input exactly, it rejects that input.

You can then place any computed result in single or double precision, from any device, against
the interval. Its error is at most its distance to the far
end of the interval. You do not need to assume anything about how that device rounds.

This is a statement about each policy actually run, made after the run. It is not a bound that
holds for every possible input. It also takes the prepared inputs as given, including any powers
precomputed into tables. `benchmarks/certify.py` applies it to every example.
