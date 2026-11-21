# Term assurance: `BasicTerm_ME`

The [glossary](../../README.md#glossary) explains the terms on this page.

## Model

This is lifelib's monthly level term assurance model. It has 44 formulas. Premiums come from a
premium table. The model also computes the net premium from its own present values. The example
uses this to show a calculation in 2 passes: first price, then project.

## Fast model

The fast model carries one value between months: the policies in force. Input preparation computes
mortality rates, lapse rates and the premium once, with the original model's own expressions.
Tests compare them with the original formula values.

## Results

| Check | Result |
|---|---|
| Exact comparison | 20 outputs, 78,568,000 recorded values on 3 portfolios; none differs. In 506 of the policies, the original model itself produces NaN |
| Exhaustive tables | 17,061 constructed test policies for mortality and lapse, and 19,926 for the premium table (19,206 of them outside the table); identical |
| Planted errors | 105 runnable planted errors. Totals detect 74 (70%). Every recorded intermediate value detects 104 (99%). Z3 proves that the remaining change preserves the result in exact arithmetic |
| Proof (Z3) | Z3 proves 44 of 44 statements for each version of the fast model, and 12 statements for early exit |
| Lean | 56 of 56 theorems for each version. Lean rejects a deliberately wrong fast model |
| Assumptions | Policy data, the mortality table lookup, and the premium (a table lookup rounded to cents) |
| Pricing in 2 passes | The 120 derived premium rates match the original model's pricing formulas. They differ from the shipped rate table. The second projection pass uses the same derived table in both models |

## Level 0 baseline

lifelib's `BasicTerm_SE` projects one policy at a time. The separate baseline run takes about 14.6
milliseconds per policy. That is about 640 times the extrapolated vectorised time in that
baseline report.
