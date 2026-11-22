# Savings with dynamic lapses: `CashValue_ME_EX2`

The [glossary](../../README.md#glossary) explains the terms on this page.

## Model

This is the savings model with switches for fees, mortality, and lapses. Its lapse rate depends on
the value of the guarantee, measured as the ratio of sum assured to account value. lifelib
defines five configurations (`sim_id` 1 to 5), and the example includes all five. The model has 92
formulas.

## Fast model

There are two versions of the fast model:

- a static version for configurations 1 to 4
- a dynamic version for configuration 5

In configuration 5, the lapse rate depends on the account value carried between months. So the
dynamic version computes a power inside the projection instead of in advance.

## Results

| Check | Result |
|---|---|
| Exact comparison | 59 outputs, 19 runs over 5 portfolios and 5 configurations, 790,582,300 recorded values; none differs. This includes the dynamic version. Across 2,622,100 tested arguments, the processor power results match NumPy. GPU results differ by 1 unit in the last place (the gap between adjacent stored numbers) for about 0.2% of those arguments |
| Exhaustive tables | 16,036 constructed test policies in each configuration; identical. The tests also change assumption tables through the model's data interface, to reach behaviour that never occurs with the shipped tables |
| Planted errors | Static version: 214 runnable planted errors. Totals detect 124 (58%), and every recorded intermediate value detects 211 (98.6%). Dynamic version: 218, 139 (64%), and 215 (98.6%) |
| Proof (Z3) | For each of the 5 configurations, Z3 proves 124 main statements and 52 early-exit goals. Of 432 runnable planted errors, Z3 finds a difference for 428 and establishes equality in exact arithmetic for 4 |
| Lean | For each configuration, Lean checks 158 exported statements: 124 main statements and 34 early-exit statements. Z3's 52 goals are these 34 statements plus 18 extra lemmas for this model. Lean does not check the extra lemmas |
| Assumptions | Policy data, switches, table lookups, and scenario returns. One proof covers one configuration |

## Values that are not finite

The original model itself produces a value that is not finite when the account value in the lapse
formula is 0. The fast model reproduces that value exactly. Early exit then gives a different
result. In the stress test portfolio, early exit changes 20 rows. `results/exact.json` records
the count.
