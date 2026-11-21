# Traditional life, annual: `fastlife`

The [glossary](../../README.md#glossary) explains the terms on this page.

## Model

This is lifelib's annual model of term, whole-life and endowment business. It spans 4 source
files: projection, policy, assumptions and economic. It covers:

- net-premium reserves from commutation columns
- surrender charges and cash values
- commissions and expenses
- present values defined backwards from the end of the projection

## Fast model

The fast model carries 2 values between years: the policies in force and the accumulated
cashflow. Input preparation computes commutation columns and rates for each policy once. Tests
compare them with the original model's values.

## Results

| Check | Result |
|---|---|
| Exact comparison | 77 outputs, 16,008,300 recorded values on 2 portfolios; none differs. A separate calculation works backwards through the fast model's recorded values with the original expression. Its 4 present-value outputs match all 838,800 values |
| Exhaustive tables | 5,544 commutation values, and 5,174,064 lookups over 4,872 constructed test policies; identical |
| Planted errors | 264 runnable planted errors. Totals detect 189 (72%). Every recorded intermediate value detects 250 (95%). Some of the 14 errors that survive comparison change terms that are 0 for every product in the model, for example an annuity over a waiting period of zero length. No portfolio of real products can reveal them |
| Proof outcomes for planted errors | 253 differ; 9 preserve exact-arithmetic results; 2 additional runnable errors fail symbolic processing. Of the 14 errors that survive comparison, Z3 finds a difference for 3 and establishes equality in exact arithmetic for 9. The symbolic check rejects 2 because it cannot process their table indices |
| Proof (Z3) | Z3 proves 158 of 158 statements for each version of the fast model, and 8 statements for early exit |
| Lean | 166 of 166 theorems for each version |
| Assumptions | Policy data and assumption lookups; the economic scenario; the reserve rate. The original model computes the reserve rate row by row with Python branches. The formal description replaces it with a handwritten formula. The proof of everything downstream of the reserve rate depends on that formula |

## Other notes

20 cells (formula results) of the original model raise errors with lifelib 0.17.3 and pandas 3.1.
These are 10 valuation-basis reserve cells and 10 present-value cells. The comparison excludes
them.

Exact comparison covers the present values defined backwards, and proof coverage excludes them.

lifelib's `simplelife` projects one policy at a time. Its recorded values agree on the baseline
portfolios. The scalar timing run takes about 20.5 milliseconds per policy.
