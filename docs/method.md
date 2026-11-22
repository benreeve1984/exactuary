# Method

The [glossary](../README.md#glossary) explains the terms on this page.

## Scope

The checks aim to detect mistakes during model conversion by a person or an AI coding agent.
Typical mistakes are a wrong table row, an operation in the wrong order, a missing branch or a
misread formula. The tests in `tests/` record mistakes of this kind, and the checks detect them.

The checks do not defend against someone who deliberately seeks a pass for a wrong model, for
example by writing unusual code or by editing the formal description or the proof files.
Adversarial testing shows that such an author can obtain a pass.

Review the formal description and any manual changes as work products. The tools do not make them
safe.

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

All interpretations come from one text, so there is no separate GPU model, proof model or test
model to maintain. The interpreters must still agree on the meaning of the text. The parser (the
code reader) accepts only the allowed syntax in `exactuary/lang.py`. The interval and symbolic
interpreters evaluate the parsed expressions one at a time. The backends regenerate code from the
parsed form and compile that code.

## The original model as reference

For testing, run the unchanged original model and record the selected intermediate values.

For proof, the proof tool reads supported formulas from the original source. An interpreter (a
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

## The proof

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

For period 0, for any later period and for any carried values, Z3 must prove the following. If the
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

If the implementation changes operation order, device or precision, run the fast model through the
interval program. The program stores each value as two endpoints, one on either side of the exact
result. Each operation rounds the lower endpoint down and the upper endpoint up. Where the
intervals cannot decide a comparison, the program keeps both outcomes.

For each row, the result is a guaranteed interval. It contains the exact-arithmetic result on the
same inputs, provided the interval program is correct. Every number in the step passes through
that program, including integer arithmetic, the time index and divisions of constants. So nothing
rounds before the program tracks it. If the program cannot represent an input exactly, it rejects
that input.

You can then place any computed result in single or double precision, from any device, against the
interval. Its distance to the far endpoint gives an upper bound on its error. You do not need to
assume anything about the device's rounding.

The resulting bound applies to the rows in that run. It does not hold for every possible input. It
takes the prepared inputs as given, including any powers precomputed into tables.
`benchmarks/certify.py` applies it to every example.
