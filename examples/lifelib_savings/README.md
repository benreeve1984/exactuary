# Savings: `CashValue_ME`

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls it the "kernel".

## Model

This is lifelib's monthly universal-life savings model. Premiums are credited to an account
value. The account value earns stochastic investment returns, and fees and a cost of insurance
are charged to it. Death, surrender and maturity benefits are paid from it, with guaranteed
minimum death and maturity benefits. The model has 79 formulas.

The test portfolios are lifelib's sample model points crossed with generated scenarios. There is
also an edge-case portfolio. It holds policies that are:

- already in force
- not yet started
- past their term
- at either end of the mortality table
- for zero or extreme amounts

## Fast model

The fast model is one function of about 60 lines. It carries 2 numbers per policy from one month
to the next: the account value and the policies in force.

A second, rearranged version groups the arithmetic differently. Its results are not identical to
the bit. It shows what proof allows.

## Results

A row is one policy in one scenario. A planted error is a single small edit to the fast model,
made to test whether the checks catch it. Z3 is an automated theorem prover from Microsoft
Research that either proves a statement or finds a counterexample. Lean is a proof assistant
that checks every step of a proof with a small, well-studied checking program.

| Check | Result |
|---|---|
| Exact comparison | 54 outputs, 741,960,000 values on 2 portfolios; none differs. Present values added up in the original model's order are identical. As a running total, they differ by at most 1 part in 10^11 |
| Exhaustive tables | 16,036 made-up policies cover every table position, in and out of range; identical |
| Planted errors | 195 planted errors that give programs that run. Portfolio totals catch 127 (65%). Comparing every intermediate value on both portfolios catches 192 (98.5%) |
| Proof (Z3) | 114 of 114 statements proved, for both versions of the fast model; 20 statements prove that early exit is safe; Z3 shows that all 195 planted errors change the results |
| Lean | 134 of 134 theorems for each version, using only Lean's standard axioms (its built-in basic rules) |
| Assumptions (the code calls these "axioms") | The model point, the last age of the mortality table, the mortality and surrender-charge table lookups, and the scenario return |

## Baseline one policy at a time

lifelib's `CashValue_SE` projects one policy at a time. It agrees with the vectorised original
model to 4 parts in 10^13 and takes 156 ms per policy. The timing runs use 40,000 rows. At that
size, this is about 500 times as long as the vectorised original model.
