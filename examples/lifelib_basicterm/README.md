# Term assurance: `BasicTerm_ME`

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls it the "kernel".

## Model

This is lifelib's monthly level term assurance model. It has 44 formulas. Premiums come from a
premium table. The model also computes the net premium from its own present values. The example
uses this to show a calculation in 2 passes: first price, then project.

## Fast model

The fast model carries one value from month to month: the policies in force. Mortality rates,
lapse rates and the premium are prepared once, with the original model's own expressions. They
are then compared exactly with the original model's cells.

## Results

A planted error is a single small edit to the fast model, made to test whether the checks catch
it. Z3 is an automated theorem prover from Microsoft Research that either proves a statement or
finds a counterexample. Lean is a proof assistant that checks every step of a proof with a small,
well-studied checking program.

| Check | Result |
|---|---|
| Exact comparison | 20 outputs, 78,568,000 values on 3 portfolios; none differs. This includes 506 policies for which the original model itself produces NaN (not a number) |
| Exhaustive tables | 17,061 made-up policies for mortality and lapse, and 19,926 for the premium table (19,206 of them off the table); identical |
| Planted errors | 105 planted errors that give programs that run. Totals catch 74 (70%). Every intermediate value catches 104 (99%). The one error not caught is proved to make no difference |
| Proof (Z3) | 44 of 44 statements proved for each version of the fast model; 12 statements for early exit |
| Lean | 56 of 56 theorems for each version. Lean rejects a fast model that is deliberately wrong |
| Assumptions (the code calls these "axioms") | The model point, the mortality table lookup, and the premium (a table lookup rounded to cents) |
| Pricing in 2 passes | For lifelib's pricing grid, the fast model's premium rates equal the original model's in 120 of 120 cases |

## Baseline one policy at a time

lifelib's `BasicTerm_SE` projects one policy at a time. It takes 14.6 ms per policy, about 640
times as long as the vectorised original model.
