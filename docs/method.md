# Method

The [glossary](../README.md#glossary) explains the terms on this page.

This page follows the order of the illustrated explanation: why proofs matter, then testing, then
proof, then rounding.

## Scope

The checks aim to detect mistakes during model conversion by a person or an AI coding agent.
Typical mistakes are a wrong table row, an operation in the wrong order, a missing branch, or a
misread formula. The tests in `tests/` record mistakes of this kind, and the checks detect them.

The checks do not defend against someone who deliberately seeks a pass for a wrong model, for
example by writing unusual code or by editing the formal description or the proof files.
Adversarial testing shows that such an author can obtain a pass.

Review the formal description and any manual changes as work products. The tools do not make them
safe.

## Tests and proofs

A faster model is a new program. Before anyone relies on it, they need evidence that it gives the
same results as the original. Exactuary provides three kinds.

1. Test. Run the original model and the fast model on test portfolios, and compare every recorded
   value in every period.
2. Prove. Show that the fast model's formulas equal the original model's formulas for every
   policy, in exact arithmetic and under stated assumptions.
3. Bound. Work out a guaranteed limit on how far rounding can move each computed result.

A test covers only the policies tested. It always leaves the question of whether the tests were
enough, and a model validation team has to settle that question by judgement. A proof covers every
policy, so the question does not arise for the calculations the proof covers.

Exactuary therefore relies on proof wherever an optimisation changes the arithmetic. Compilation
and parallel cores leave the arithmetic unchanged, and comparison is sufficient evidence on the
policies tested. Early exit, a GPU, and reduced precision change the arithmetic, and the evidence
then comes from the proof and the rounding bounds.

A test of whether 3 × (4 + 5) equals 3 × 4 + 3 × 5 settles that one case. A proof that
a × (b + c) equals a × b + a × c settles every case. The proofs here apply the same idea to
models: they work with symbols such as "the policy's age" in place of numbers.

## Principle

A person or an AI coding agent can write the fast model. The checks assess that model against the
original model. A small set of checks decides whether to accept it, and their code is short enough
to read.

The proofs depend on one part of the author's work: the formal description. It says how to read
the original model. It includes the assumptions (the code calls them "axioms"), the table
definitions and keys, and the mapping of inputs. A reviewer must check the formal description and
assumptions, even if the same author supplied them. A proof can be valid under its stated
assumptions even when those assumptions describe the original model incorrectly. Exact comparison
does not depend on the formal description.

Compiler research calls this arrangement translation validation. Each converted model receives its
own checks.

## The step function

The fast model uses a step function for one period of one policy. Execution code repeats the step
through the projection. The step function receives:

- the policy's data
- the period's inputs
- the small set of values carried from the previous period

It returns that period's values and the values to carry to the next period.

Write the step function in a restricted form: arithmetic without loops or branches. Use
`sel(condition, a, b)` for choices.

Different programs interpret the same text:

| Interpretation | Purpose |
|---|---|
| Compiled for a processor (Numba) | Fast execution; exact comparison with the original model |
| Compiled for a graphics processor (CUDA) | Faster execution |
| Array arithmetic (NumPy) | Measuring test performance |
| Symbols (Z3) | Proof of equality in exact arithmetic |
| Interval arithmetic with outward rounding | A guaranteed interval around the exact result |

All interpretations come from one text, so there is no separate GPU model, proof model, or test
model to maintain. The interpreters must still agree on the meaning of the text. The parser (the
code reader) accepts only the allowed syntax in `exactuary/lang.py`. The interval and symbolic
interpreters evaluate the parsed expressions one at a time. The backends regenerate code from the
parsed form and compile that code.

## Testing

Run the unchanged original model and record the selected intermediate values: policies in force,
account values, and each cashflow, for every policy and every period. The fast model must
reproduce each value exactly. The test portfolios include each model's sample policies and a set
of unusual cases, such as policies already matured and ages at the ends of the mortality table.

Compare every value, not only totals. Planted errors show why. `exactuary/mutate.py` makes single
edits to the step function, such as a plus in place of a minus or the wrong row of a table. The
harness then counts how many of these errors each check catches. In the examples, a comparison of
portfolio totals or of each policy's present values catches 58% to 73%. A comparison of every
recorded value catches 95% to 99%. A difference in a single value also identifies the calculation
at fault.

A comparison test has three limits.

1. It says nothing about a policy outside the test portfolios.
2. Early exit and rearranged sums add arithmetic that the original model does not perform. Results
   can differ in the last digits, so an exact match is no longer the test.
3. A GPU can combine operations, and reduced precision keeps fewer digits. Results differ by
   design.

Proof addresses the first limit. Proof and rounding bounds together address the second and third.

## Proof tools

Mathematicians use two kinds of program for proofs.

A theorem prover searches for a proof without help. Exactuary uses Z3, from Microsoft Research.
Given a statement such as "for every policy, these two formulas are equal", Z3 finds a proof,
returns a counterexample, or reports that it cannot decide. Z3 is a large program, and its answer
depends on that program being correct.

A proof checker accepts a statement only when every step of its proof follows from the rules of
logic. Exactuary uses Lean. The part of Lean that checks proofs is small, and many people have
studied it. Mathematicians use Lean to check research proofs. In November 2026 the Lean and
Mathlib communities [confirmed](https://leanprover-community.github.io/blog/posts/riemann-hypothesis/)
that Lean had checked a 17-million-line proof of the Riemann hypothesis.

Exactuary uses the same tools on much smaller statements. Z3 finds each proof. Exactuary then
exports each main statement to Lean, which checks its own proof of it. The evidence then no longer
depends on Z3 being correct.

## How the proof reads the original model

The proof tool reads supported formulas from the original source. An interpreter (a
program that evaluates formulas) evaluates them with symbols. This gives a formula for each value
in terms of the policy data. A separate description supplies assumptions for unsupported parts.
The annual example includes a handwritten reserve formula. Each assumption needs its own tests
and review. There are two qualifications.

First, Python does not run the original formulas. The interpreter accepts a listed set of Python
constructs and rejects others. Tests found incorrect readings, and the interpreter now rejects
those cases. Related forms may still pass. Tests cover the listed supported constructs, and the
interpreter has no proof of completeness.

The formal description supplies trusted code, such as assumptions and stand-ins for other source
files. That code runs as written.

Unsupported parts include inputs, table lookups that use operations on tables of labelled data,
random draws and, in the annual example, the handwritten reserve formula. Assumptions replace each
of these. Tests check each assumption case by case.

Second, the interpreter treats every value as a single real number or integer. It does not model
missing values, infinities, index alignment (matching table rows by their labels) or integer
overflow (a whole number exceeding its storage range). So the proof concerns one reading of the
original model, and the evidence depends on that reading.

## Tables that ignore a position

Sometimes the formal description defines a precomputed table in a way that does not depend on one
of its row or column positions (indices). This happens, for example, when a single symbol
describes a per-policy assumption. The description must then state which fast-model input that
position must equal. This is the key. The proof treats a lookup at any other position as an
unknown value. The checks always enforce a declared key.

The library rejects a definition that ignores a position and declares no key. This safeguard
catches oversights. It does not prove the description right. If the description states the wrong
key, or wrongly treats a table as constant, the proof follows the wrong description.

## The statements to prove

Split the proof into statements (proof obligations). For each output, Z3 checks two statements.

1. In period 0, the fast model's value equals the original model's value for every possible
   policy.
2. In any later period, suppose the fast model's carried values match the original model's. Then
   the fast model's value equals the original model's, and the fast model again carries forward
   the correct values.

If both hold for every output, induction (a proof that agreement carries from one period to the
next) gives equality in every period.

Lean proves two general results once, in `exactuary/lean/Glue.lean`:

- the induction itself
- the equality of present values as running totals

No program checks that a particular model's statements fit the conditions of those general
results. A reviewer must check the exporter's connection between each model and the general
results. This includes the match between the description's "state after period t" and the
theorem's "state entering period t+1".

## Early exit

A fast model may stop a projection once a stated rule holds, for example when no policies remain
in force. The compiled code runs this rule, and the statements for early exit come from that same
rule.

For period 0, for any later period, and for any carried values, Z3 must prove the following. If the
rule holds after a period, then in the next period every accumulated cashflow is zero and the rule
holds again.

Then compare the stopped and full projections, value for value, on every test portfolio.

The proof establishes zero future accumulated cashflows in exact arithmetic once the exit rule
holds. Computer results can still differ if the original model later produces an undefined or
infinite value. The examples report the affected rows.

## Rounding

Proof of equality in exact arithmetic does not establish equality of computed results, because
computer arithmetic rounds. There are two cases.

If the fast model performs the original model's operations in the original order, expect agreement.
Exact comparison confirms it on the portfolios tested.

If the implementation changes operation order, device, or precision, run the fast model through the
interval program. The program stores each value as two endpoints, one on either side of the exact
result. Each operation rounds the lower endpoint down and the upper endpoint up. Where the
intervals cannot decide a comparison, the program keeps both outcomes.

For each row, the result is a guaranteed interval. It contains the exact-arithmetic result on the
same inputs, provided the interval program is correct. Every number in the step passes through
that program, including integer arithmetic, the time index, and divisions of constants. So nothing
rounds before the program tracks it. If the program cannot represent an input exactly, it rejects
that input.

You can then place any computed result in single or double precision, from any device, against the
interval. Its distance to the far endpoint gives an upper bound on its error. You do not need to
assume anything about the device's rounding.

The resulting bound applies to the rows in that run. It does not hold for every possible input. It
takes the prepared inputs as given, including any powers precomputed into tables.
`benchmarks/certify.py` applies it to every example.
