# Savings: `CashValue_ME`

**Model.** lifelib's monthly universal-life savings model: premiums are credited to an account
value, which earns stochastic investment returns and is charged fees and a cost of insurance;
death, surrender and maturity benefits are paid from it, with guaranteed minimum death and
maturity benefits. 79 formulas. Portfolios are lifelib's sample model points crossed with
generated scenarios, plus an edge portfolio of policies already in force, not yet started,
past term, at the ends of the mortality table, and with zero or extreme amounts.

**Kernel.** One function of about 60 lines carrying two numbers per policy between months (the
account value and the policies in force). A second, rearranged version groups the algebra
differently; it is not bit-identical and is used to show what proof permits.

| Level | Result |
|---|---|
| Exact comparison | 54 outputs, 741,960,000 values on two portfolios; none differs. Present values summed in the original's order are identical; as a running sum they differ by at most 1 part in 10^11 |
| Exhaustive tables | 16,036 synthetic policies covering every table position in and out of range; identical |
| Seeded defects | 195 runnable single-edit defects. Portfolio totals catch 127 (65%); every intermediate value, both portfolios, catches 192 (98.5%) |
| Proof (Z3) | 114 of 114 obligations, for both kernels; 20 obligations justifying early exit; all 195 runnable defects refuted |
| Lean | 134 of 134 theorems for each kernel, using only Lean's standard axioms |
| Assumed (axioms) | The model point, the last age of the mortality table, the mortality and surrender-charge table lookups, and the scenario return |

Unvectorised baseline: lifelib's `CashValue_SE`, which projects one policy at a time, agrees
with the vectorised original to 4 parts in 10^13 and takes 156 ms per policy; at the 40,000-row
size used for timing, that is about 500 times the vectorised original.
