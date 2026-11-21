# Savings with dynamic lapses: `CashValue_ME_EX2`

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls it the "kernel".

## Model

This is the savings model with switches for fees, mortality and lapses. Its lapse rate depends on
how valuable the guarantee is, measured as the ratio of sum assured to account value. lifelib
defines 5 configurations (`sim_id` 1 to 5), and all 5 are converted. The model has 92 formulas.

## Fast model

There are 2 fast models:

- a static one for configurations 1 to 4
- a dynamic one for configuration 5

In configuration 5, the lapse rate depends on the account value carried from month to month. So
the dynamic fast model must compute a power inside the projection, rather than in advance.

## Results

A planted error is a single small edit to the fast model, made to test whether the checks catch
it. Z3 is an automated theorem prover from Microsoft Research that either proves a statement or
finds a counterexample. Lean is a proof assistant that checks every step of a proof with a small,
well-studied checking program.

| Check | Result |
|---|---|
| Exact comparison | 59 outputs, 19 runs over 5 portfolios and 5 configurations, 790,582,300 values; none differs. This includes the dynamic fast model. Its power function agrees with NumPy's on the processor. On the GPU it differs in the last bit for 0.2% of arguments |
| Exhaustive tables | 16,036 made-up policies in each configuration; identical. The assumption tables were also changed through the model's own data objects, to reach behaviour that never occurs with the shipped tables |
| Planted errors | Static fast model: 214 planted errors that give programs that run. Totals catch 124 (58%), and every intermediate value catches 211 (98.6%). Dynamic fast model: 218, 139 (64%) and 215 (98.6%) |
| Proof (Z3) | 124 of 124 statements proved in each configuration; 52 statements for early exit in each; 428 of 432 planted errors shown to change the results, and 4 shown to make no difference |
| Lean | 158 of 158 theorems in each configuration |
| Assumptions (the code calls these "axioms") | Model point, switches, table lookups, scenario returns. One proof covers one configuration |

## Values that are not finite numbers

Sometimes the original model itself produces a value that is not a finite number. This happens
when the account value in the lapse formula is 0. The fast model reproduces that value exactly.
Stopping early is then not equivalent, and the affected rows are reported. A row is one policy
in one scenario.
