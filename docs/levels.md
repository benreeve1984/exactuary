# Levels of acceleration

You can make a cashflow model faster in a series of levels. Each level moves the model further
from its validated form, so each level needs stronger checks that it still matches the original. This page defines the
levels and the evidence for each, and sets out how to choose between them. The
[glossary](../README.md#glossary) explains the terms.

## Summary

| Level | What changes | Agreement with the original model | Evidence | Main cost |
|---|---|---|---|---|
| 0. One policy at a time | lifelib's separate model processes one policy at a time in interpreted Python, where such a model exists | Separate checks compare it with level 1 | Baseline checks against the level 1 model | Run time |
| 1. Vectorised | The unchanged original model processes policies together, with operations on arrays of policy values. It is the comparison reference | Reference | Baseline checks against the separate level 0 model, where available | Memory grows with policies times periods |
| 2. Compiled | Machine code projects each policy through all its periods, on one processor core | Recorded values agree on the test portfolios. Present values agree when both models add them in the same order | Exact comparison of every recorded intermediate value, and a measured detection rate for planted errors | The author rewrites the model in a restricted form |
| 3. All processor cores (parallel) | The compiled model runs on all processor cores | As level 2 | As level 2 | None beyond level 2 |
| 4. Restructured | Remove or reorder work: early exit, and rearranged arithmetic | Equal in exact arithmetic. Rounded results may differ in the last digits | Proof of equality in exact arithmetic under stated assumptions; comparison of the stopped and full projections; rounding checks where rounding changes | A formal description of the model. The proof concerns exact arithmetic, and separate checks assess rounding |
| 5. GPU, double precision | The same arithmetic runs on a graphics processor (GPU), with about 16 significant digits | Differs in the last digits | Proof as level 4, plus rounding error bounds | Hardware cost and availability. The gain depends strongly on the device |
| 6. GPU, single precision | The number format keeps about 7 significant digits | Individual values may differ materially | Rounding error bounds, and accuracy checks on each reported output | Accuracy. Each reported output needs an agreed error limit |

Levels 2 and 3 are separate because level 2 changes the model. Level 3 changes only the number of
processors.

## The checks

There are five complementary checks. Each addresses a different source of error.

1. Comparison of recorded values. Run the original model on a test portfolio and record each
   selected intermediate value in each period. The fast model must reproduce each value exactly.
   This shows agreement for the policies tested. The comparison treats equal numeric values and
   pairs of undefined values as matching. It separately prints differences in the stored binary
   representation, including signed zeros and undefined values. The saved JSON reports do not
   retain that count.
2. Tests with planted errors. Plant an error, run the comparison, and record whether it detects the
   error. The recorded detection rate measures test performance.
3. Proof in exact arithmetic. An interpreter evaluates the original model's formulas with symbols.
   Z3 then checks whether any policy, in any period, could give a different value in the fast
   model. For each covered output, this establishes equality in exact arithmetic under the
   model's stated assumptions (see [Dependencies of the evidence](trusted-base.md)).
4. Separate proof checking. The exporter translates the statements for Lean, which checks separate
   proofs. This removes reliance on Z3's verdict for those statements. The translation covers
   integers, real numbers, and true-or-false values, and rejects anything else. The translation,
   assumptions and connection to the full projection still need review.
5. Rounding error bounds. A separate check computes an interval for each row in the certification
   run. The interval contains the exact fast-model result on the prepared inputs, provided the
   interval program is correct. The check reports any row without a bound.

A running total of present values can differ in the last digits from the original model's method,
which multiplies cashflows by discount factors and adds them. Across the saved exact-comparison
runs, the largest reported relative difference from a running total is about 1.27 parts in ten
billion, for configuration 3 of savings with dynamic lapses. Matching the original summation
order removes this difference on those tests.

Exact comparison tests the programs that run, on real inputs, against the original model. Proof
extends the result to all inputs within the stated assumptions, for a formal reading of the
selected formulas. It excludes some calculations, non-finite values, and machine rounding.

Some errors leave the exact-arithmetic meaning unchanged but damage the computed result. One
example is adding and then subtracting a very large number. Such an error passes every proof. You
also need a numerical check to detect changes in rounding.

Proofs concern exact arithmetic. Z3 uses real numbers here, and Lean uses fractions. Both represent
powers, exponentials, and logarithms as named functions with unspecified behaviour. So the proof
does not establish how a computer calculates them.

At levels 2 and 3, exact comparison shows agreement on the policies tested. It does not show that
rounding agrees on other policies.

At levels 4 to 6, wherever rounding changes, the interval check applies. The interval program
rounds each lower endpoint down and each upper endpoint up. You can place any computed result
against the interval. Tests compare the program with exact fraction arithmetic and with
adversarial cases. The interval program has no formal correctness proof.

## Choosing a level

To choose a level, ask these questions.

1. Does the current run time meet the task's requirements? If so, you do not need to change
   anything.
2. Must every reported figure match the existing model exactly? If so, levels 2 and 3 are the
   limit. They need no special hardware. Present values match only when the fast model adds them
   in the original order.
3. Which outputs do you report? For each reported output, set an acceptable error and check it on
   the intended workload. Apply separate checks to totals, percentiles, and ratios. Aggregation
   alone does not guarantee an acceptable error, especially for tails, ratios, or totals near 0.
   Double precision can serve individual outputs if you check them.
4. Is compatible hardware available? GPU use needs compatible hardware, which may be costly or
   shared. It also needs bounds on rounding.
5. What will the reviewer accept? Reviewers can follow exact comparison easily. A proof asks the
   reviewer to accept a formal description of the model and a short list of stated assumptions.
   Separate proof checking reduces reliance on Z3 but adds nothing to speed.

Start with level 3 when its test results and run time meet the task's requirements. Move to level
4 where a proof is available. Consider levels 5 and 6 for workloads with very large numbers of
scenarios, after you check device support and accuracy on the intended outputs.
