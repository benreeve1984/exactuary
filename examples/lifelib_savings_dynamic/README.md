# Savings with dynamic lapses: `CashValue_ME_EX2`

**Model.** The savings model with switches for fees, mortality and lapses, and a lapse rate that
depends on how valuable the guarantee is (the ratio of sum assured to account value). lifelib
defines five configurations (`sim_id` 1 to 5); all five are converted. 92 formulas.

**Kernel.** Two kernels: a static one for configurations 1 to 4, and a dynamic one for
configuration 5, in which the lapse rate depends on the carried account value and so requires a
power to be computed inside the projection rather than precomputed.

| Level | Result |
|---|---|
| Exact comparison | 59 outputs, 19 runs over five portfolios and five configurations, 790,582,300 values; none differs. This includes the dynamic kernel, whose power function agrees with NumPy's on the processor; on the GPU it differs in the last bit for 0.2% of arguments |
| Exhaustive tables | 16,036 synthetic policies in each configuration; identical. Assumption tables were also perturbed through the model's own data objects to reach behaviour the shipped tables never trigger |
| Seeded defects | Static kernel: 214 runnable defects, totals catch 124 (58%), every intermediate catches 211 (98.6%). Dynamic kernel: 218, 139 (64%) and 215 (98.6%) |
| Proof (Z3) | 124 of 124 obligations in each configuration; 52 early-exit obligations each; 428 of 432 runnable defects refuted, 4 proved equivalent |
| Lean | 158 of 158 theorems in each configuration |
| Assumed (axioms) | Model point, switches, table lookups, scenario returns. One proof covers one configuration |

Where the original itself produces a non-finite value (an account value of zero in the lapse
formula), the kernel reproduces it exactly; stopping early is then not equivalent and the rows
affected are reported.
