# Levels of acceleration

A cashflow model can be made faster in a sequence of steps. Each step changes the model
further from the form in which it was written and validated, and therefore requires stronger
evidence that the results are unchanged. This document defines the levels, the evidence each
requires, and the considerations in choosing between them.

## Summary

| Level | What changes | Are results identical? | Evidence required | Principal cost |
|---|---|---|---|---|
| 0. Original | Nothing. One policy at a time, interpreted Python. | Reference | None | Run time |
| 1. Vectorised | All policies are processed together with array operations. | Normally identical | Reconciliation of results | Memory grows with policies x months |
| 2. Compiled | Each policy is projected through all its months in machine code, on one processor core. | Every monthly value identical; present values identical when summed in the original's order | Exact comparison of every intermediate value; measured strength of the tests | The model is rewritten in a restricted form |
| 3. Parallel | The compiled model runs on all processor cores. | As level 2 | As level 2 | None beyond level 2 |
| 4. Restructured | Work is removed or reordered: stopping a projection once nothing remains in force; rearranging algebra. | Equal in exact arithmetic; may differ in the last digits | Mathematical proof of equivalence, and a comparison of the stopped projection with the full one | Requires a formal description of the model; proofs concern real numbers, not rounding |
| 5. GPU, double precision | The same arithmetic is executed on a graphics processor. | Differs in the last digits | Proof as level 4, plus an enclosure of the exact result | Hardware cost and availability; gain depends strongly on the device |
| 6. GPU, single precision | Numbers are held to about 7 significant digits rather than 16. | No: individual values may differ materially | Enclosure of the exact result; comparison with the reference at the level at which results are used | Accuracy; suitable only where aggregate results are required |

Levels 2 and 3 are distinguished because the first is a change to the model and the second is
only a change to how many processors execute it.

## The evidence available

Four forms of evidence are provided, in increasing strength.

1. **Exact comparison.** The original model is run on a test portfolio and every intermediate
   value in every month is recorded. The accelerated model must reproduce each value exactly.
   This establishes equality for the policies tested. "Exactly" means equal as numbers; the
   comparison also counts values that are equal as numbers but differ in their bits (the sign
   of a zero), and in every example that count is nil. Present values summed as a running total
   differ from the original's matrix product in the last digits (about one part in 10^11); summed
   in the original's order they are identical.
2. **Measured tests.** Deliberate defects are introduced into the accelerated model, one at a
   time, and the proportion detected by the comparison is recorded. This measures the quality of
   the test portfolio and of the comparison, rather than assuming it.
3. **Proof.** The formulas of the original model are executed symbolically and an automated
   theorem prover (Z3) is asked whether any policy, in any month, could produce a different value
   in the accelerated model. This establishes equality for all policies, in exact arithmetic,
   relative to a symbolic reading of the original (see "What must be trusted").
4. **Independent proof checking.** The statements are translated for a proof assistant (Lean),
   whose small trusted kernel checks each proof. The translation covers integers, real numbers
   and truth values and refuses anything else. The solver's verdict then no longer needs to be
   trusted. The statements themselves, and their translation, still do.

The four are not a ladder in which each replaces the one before. Exact comparison is the
foundation: it tests the programs that actually run, on actual numbers, against the actual
original. Proof extends the conclusion from the policies tested to all policies, but it is a
proof about a description of the original, and it concerns real numbers. A defect that leaves
the real-number meaning unchanged but damages the computed result (for example adding and then
subtracting a very large number) passes every proof and is caught only by exact comparison.

Proofs are statements about exact arithmetic: Z3 works over the real numbers and Lean over the
rationals, with powers, exponentials and logarithms treated as unknown functions rather than as
the real ones. Computers round. Rounding is addressed separately. At levels 2 and 3, exact
comparison shows that on the policies tested the computed values agree with the original's; it
does not show that rounding agrees on policies not tested. At levels 4 to 6, an enclosure is
computed for each policy run: an interval, obtained with outward rounding, that contains the
exact result provided the enclosure program is correct, and against which any computed result
can be placed. That program has been tested against exact rational arithmetic and by adversarial cases; it is
not formally verified.

## Choosing a level

The following questions determine the appropriate level.

1. **Is the present run time a constraint?** If not, no change is warranted.
2. **Must every reported figure match the existing model exactly?** If so, levels 2 and 3 are
   the limit, and for most purposes are sufficient: they require no special hardware and no
   argument about rounding.
3. **Is the model used for results at the level of the individual policy or only in
   aggregate?** Single precision (level 6) is defensible only for aggregates such as a mean or a
   percentile over many scenarios, and only where the measured aggregate error is small relative
   to the statistical error of the simulation.
4. **Is suitable hardware available at the time the model must run?** Levels 5 and 6 introduce
   a dependency on a device that may be costly or shared.
5. **What will the reviewer accept?** Exact comparison is readily understood. A proof requires
   the reviewer to accept a formal description of the model and a short list of stated
   assumptions. Independent proof checking reduces what must be trusted but adds nothing to
   speed.

As a default: adopt level 3; proceed to level 4 where a proof can be obtained; consider levels
5 and 6 only for workloads dominated by very large numbers of scenarios and reported in
aggregate.
