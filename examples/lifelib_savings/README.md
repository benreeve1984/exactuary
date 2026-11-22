# Savings: `CashValue_ME`

The [glossary](../../README.md#glossary) explains the terms on this page.

## Model

This is lifelib's monthly universal-life savings model. The model credits premiums to an account
and deducts fees and insurance charges. The account earns stochastic investment returns. It funds
death, surrender and maturity benefits, subject to guaranteed minimum death and maturity benefits.
The model has 79 formulas.

The test portfolios combine lifelib's sample model points with generated scenarios. An edge-case
portfolio adds policies:

- already in force
- not yet started
- past their term
- at either end of the mortality table
- with zero or extreme amounts

## Fast model

The step function has about 60 lines. It carries two values per policy from one month to the next:
the account value and the policies in force.

A second, rearranged version groups the arithmetic differently. Its stored values are not always
identical to the original model's. It shows what proof allows.

## Results

| Check | Result |
|---|---|
| Exact comparison | 54 outputs, 741,960,000 recorded values on 2 portfolios; none differs. Present values agree when the fast model adds them in the original order. As a running total, they differ by at most 1 part in 10^11 |
| Exhaustive tables | 16,036 constructed test policies cover every table position, in and out of range; identical |
| Planted errors | 195 runnable planted errors. Portfolio totals detect 127 (65%). Comparing every recorded intermediate value on both portfolios detects 192 (98.5%) |
| Proof (Z3) | Z3 proves 114 of 114 statements for both versions of the fast model, and 20 statements for early exit. Z3 finds a difference for all 195 runnable planted errors |
| Lean | 134 of 134 theorems for each version. The proofs depend only on Lean's standard axioms (its built-in logical assumptions) |
| Assumptions | Policy data, the last age of the mortality table, the mortality and surrender-charge table lookups, and the scenario return |

## Level 0 baseline

lifelib's `CashValue_SE` projects one policy at a time. Its results agree with the vectorised
original model to four parts in 10^13. The scalar run takes about 156 milliseconds per policy on 300
policies. On that same portfolio it takes about 21 times as long as the vectorised run.
