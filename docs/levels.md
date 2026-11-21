# Levels of acceleration

You can make a cashflow model faster in a series of levels. Each level moves the model further
from its original, validated form. So each level needs stronger evidence that
the results have not changed.

This document defines the levels and says what evidence each one needs. It also sets out what to
consider when you choose between them.

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls the fast model the "kernel".

## Summary

| Level | What changes | Are results identical? | Evidence needed | Main cost |
|---|---|---|---|---|
| 0. One policy at a time | Nothing. Each policy runs through interpreted Python, one at a time. | This is the reference | None | Run time |
| 1. Vectorised | All policies are processed together, using operations on whole arrays. | Usually identical | Reconciliation of results | Memory grows with policies x months |
| 2. Compiled | Each policy is projected through all its months in machine code, on one processor core. | Every monthly value is identical. Present values are identical when added up in the original model's order | Exact comparison of every intermediate value, and a measure of how strong the tests are | The model is rewritten in a restricted form |
| 3. All processor cores (parallel) | The compiled model runs on all processor cores. | As level 2 | As level 2 | None beyond level 2 |
| 4. Restructured | Work is removed or reordered. The projection stops once nothing is in force, and the arithmetic is rearranged. | Equal in exact arithmetic. May differ in the last digits | Mathematical proof that the results are equal, and a comparison of the stopped projection with the full one | Needs a formal description of the model. Proofs concern real numbers, not rounding |
| 5. GPU, double precision | The same arithmetic runs on a graphics processor (GPU). Numbers are held to about 16 significant digits. | Differs in the last digits | Proof as level 4, plus a guaranteed interval that contains the exact result | Hardware cost and availability. The gain depends strongly on the device |
| 6. GPU, single precision | Numbers are held to about 7 significant digits rather than 16. | No. Individual values may differ materially | Guaranteed interval that contains the exact result, and comparison with the reference at the level at which results are used | Accuracy. Suitable only where you need aggregate results |

Levels 2 and 3 are separate because level 2 changes the model. Level 3 only changes how many
processors run it.

## The evidence available

There are 4 forms of evidence. They are listed here from weakest to strongest.

1. Exact comparison. The original model is run on a test portfolio, and every intermediate value
   in every month is recorded. The fast model must reproduce each value exactly. This shows the
   results are equal for the policies tested. "Exactly" means equal as numbers. The comparison
   also counts values that are equal as numbers but differ in their bits (the sign of a zero). In
   every example, that count is 0. Present values added up as a running total differ from the
   original model's matrix product in the last digits, by about 1 part in 10^11. When added up in
   the original model's order, they are identical.
2. Measured tests. Errors are planted in the fast model, one at a time. The comparison
   is then run, and its detection rate is recorded. This measures the quality of the test portfolio and of the
   comparison, rather than assuming it.
3. Proof. The formulas of the original model are run on symbols instead of numbers. Z3 is then
   asked whether any policy, in any month, could give a different value in the fast model. Z3 is
   an automated theorem prover from Microsoft Research that either proves a statement or finds a
   counterexample. This shows the results are equal for all policies, in exact arithmetic. The
   result depends on a description of the original model as formulas (see "What must be
   trusted").
4. Independent proof checking. The statements are translated for Lean. Lean is a proof assistant
   that checks every step of a proof with a small, well-studied checking program. The translation
   covers integers, real numbers and true-or-false values, and rejects anything else. With this
   check, you no longer need to trust Z3's verdict. You do still need to trust the statements
   themselves, and their translation.

These forms of evidence do not replace each other in turn. Exact comparison is the foundation. It
tests the programs that actually run, on actual numbers, against the actual original model.

Proof extends the conclusion from the policies tested to all policies. But it is a proof about a
description of the original model, and it concerns real numbers. Some errors leave the
real-number meaning unchanged but damage the computed result. One example is adding and then
subtracting a very large number. Such an error passes every proof. Only exact comparison catches
it.

Proofs are statements about exact arithmetic. Z3 works with real numbers and Lean with
rational numbers (fractions). Both treat powers, exponentials and logarithms as unknown functions,
not as the real ones. Computers round, so rounding is dealt with separately.

At levels 2 and 3, exact comparison shows that the computed values agree with the original
model's on the policies tested. It does not show that rounding agrees on policies not tested.

At levels 4 to 6, a guaranteed interval is computed for each policy run. The code calls this an
"enclosure". The interval is computed with outward rounding, which means the lower end is always
rounded down and the upper end up. It contains the exact result, provided the program that
computes it is correct. You can then place any computed result against it. That program has been
tested against exact arithmetic with fractions and by adversarial testing. It has not been
formally proved correct.

## Choosing a level

To choose the right level, ask these questions.

1. Is the current run time a constraint? If not, you do not need to change anything.
2. Must every reported figure match the existing model exactly? If so, levels 2 and 3 are the
   limit. For most purposes they are enough. They need no special hardware and no argument about
   rounding.
3. Is the model used for results for individual policies, or only in aggregate? Single precision
   (level 6) is defensible only for aggregates, such as a mean or a percentile over many
   scenarios. Even then, the measured aggregate error must be small compared with the statistical
   error of the simulation.
4. Will suitable hardware be available when the model must run? Levels 5 and 6 depend on a device
   that may be costly or shared.
5. What will the reviewer accept? Exact comparison is easy to understand. A proof needs the
   reviewer to accept a formal description of the model and a short list of stated assumptions.
   Independent proof checking reduces what must be trusted, but adds nothing to speed.

As a default, use level 3. Move to level 4 where a proof can be obtained. Consider levels 5 and 6
only for work with very large numbers of scenarios, where you report totals only.
