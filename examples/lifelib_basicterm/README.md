# Term assurance: `BasicTerm_ME`

**Model.** lifelib's monthly level term assurance model. 44 formulas. Premiums come from a
premium table; the model also computes the net premium from its own present values, which the
example uses to show a two-pass calculation (price, then project).

**Kernel.** One value carried between months (policies in force). Mortality and lapse rates and
the premium are prepared once, with the original's own expressions, and compared exactly with
the original's cells.

| Level | Result |
|---|---|
| Exact comparison | 20 outputs, 78,568,000 values on three portfolios, including 506 policies for which the original itself produces NaN; none differs |
| Exhaustive tables | 17,061 synthetic policies for mortality and lapse, 19,926 for the premium table (19,206 of them off the table); identical |
| Seeded defects | 105 runnable defects. Totals catch 74 (70%); every intermediate value catches 104 (99%); the survivor is proved equivalent |
| Proof (Z3) | 44 of 44 obligations for each kernel; 12 early-exit obligations |
| Lean | 56 of 56 theorems for each kernel. A deliberately wrong kernel is rejected by Lean |
| Assumed (axioms) | The model point, the mortality table lookup, and the premium (a table lookup rounded to cents) |
| Two-pass pricing | Premium rates derived by the kernel for lifelib's pricing grid equal the original's 120 of 120 |

Unvectorised baseline: lifelib's `BasicTerm_SE` takes 14.6 ms per policy, about 640 times the
vectorised original.
