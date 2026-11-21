# Traditional life, annual: `fastlife`

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls it the "kernel".

## Model

This is lifelib's annual model of term, whole-life and endowment business. It is spread over 4
modules: projection, policy, assumptions and economic. It covers:

- net-premium reserves from commutation columns
- surrender charges and cash values
- commissions and expenses
- present values defined backwards from the end of the projection

## Fast model

The fast model carries 2 values from year to year: the policies in force and the accumulated
cashflow. Commutation columns and rates for each policy are prepared once. They are then compared
exactly with the original model's cells.

## Results

A planted error is a single small edit to the fast model, made to test whether the checks catch
it. Z3 is an automated theorem prover from Microsoft Research that either proves a statement or
finds a counterexample. Lean is a proof assistant that checks every step of a proof with a small,
well-studied checking program.

| Check | Result |
|---|---|
| Exact comparison | 77 outputs, 16,008,300 values on 2 portfolios; none differs. The 4 present-value cells match all 838,800 values. They are computed backwards from the fast model's trace, using the original model's expression |
| Exhaustive tables | 5,544 commutation values, and 5,174,064 lookups over 4,872 made-up policies; identical |
| Planted errors | 264 planted errors that give programs that run. Totals catch 189 (72%). Every intermediate value catches 250 (95%). Some of the 14 errors not caught edit terms that are 0 for every product in the model, for example an annuity over a waiting period of zero length. No portfolio of real products can reveal them |
| Proof (Z3) | 158 of 158 statements proved for each version of the fast model; 8 statements for early exit; 253 of 262 planted errors shown to change the results, and 9 shown to make no difference |
| Lean | 166 of 166 theorems for each version |
| Assumptions (the code calls these "axioms") | Model point and assumption lookups; the economic scenario; the reserve rate. The original model computes the reserve rate row by row, with Python branches. The description of the model as formulas describes it with a formula written by hand. The proof of everything that depends on the reserve rests on that formula |

## Other notes

20 cells of the original model raise errors when run with lifelib 0.17.3 and pandas 3.1. These
are 10 valuation-basis reserve cells and 10 present-value cells. They are left out of the
comparison.

Present values defined backwards are tested exactly but not proved.

lifelib's `simplelife` projects one policy at a time. It agrees exactly and takes 20 ms per
policy.
