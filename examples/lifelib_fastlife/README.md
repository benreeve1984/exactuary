# Traditional life, annual: `fastlife`

**Model.** lifelib's annual model of term, whole-life and endowment business, spread over four
modules (projection, policy, assumptions, economic): net-premium reserves from commutation
columns, surrender charges and cash values, commissions and expenses, and present values
defined backwards from the end of the projection.

**Kernel.** Two values carried between years (policies in force and accumulated cashflow).
Commutation columns and per-policy rates are prepared once and compared exactly with the
original's cells.

| Level | Result |
|---|---|
| Exact comparison | 77 outputs, 16,008,300 values on two portfolios; none differs. The four present-value cells, computed backwards from the kernel's trace with the original's expression, match all 838,800 values |
| Exhaustive tables | 5,544 commutation values and 5,174,064 lookups over 4,872 synthetic policies; identical |
| Seeded defects | 264 runnable defects. Totals catch 189 (72%); every intermediate value catches 250 (95%). The 14 survivors include edits to terms that are zero for every product the model defines (for example an annuity over a waiting period of zero length), which no portfolio of real products can reveal |
| Proof (Z3) | 158 of 158 obligations for each kernel; 8 early-exit obligations; 253 of 262 runnable defects refuted, 9 proved equivalent |
| Lean | 166 of 166 theorems for each kernel |
| Assumed (axioms) | Model point and assumption lookups; the economic scenario; the reserve rate, which the original computes row by row with Python branches and which is described by a hand-written formula in the symbolic description. The proof of everything downstream of the reserve is relative to that formula |

Twenty cells of the original raise errors when evaluated with lifelib 0.17.3 and pandas 3.1 (ten
valuation-basis reserve cells and ten present-value cells); they are outside the comparison.
Backward-defined present values are tested exactly but not proved. Unvectorised baseline:
lifelib's `simplelife` agrees exactly and takes 20 ms per policy.
