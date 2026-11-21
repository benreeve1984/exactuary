# Limitations

1. **Scope of the examples.** The examples are open-source teaching models from the lifelib
   project. They are smaller and more regular than production models. Features common in
   production, including management actions, nested stochastic projections and interactions
   between policies, are not represented unless an example's report states otherwise.
2. **Proofs concern exact arithmetic.** Equality is proved over the real numbers. Agreement of
   computed results is established separately, by exact comparison or by a rounding bound.
3. **Axioms.** Parts of each original model are assumed rather than executed symbolically.
   An incorrect axiom would invalidate the corresponding proof. Each axiom is tested but the
   tests are finite.
4. **Input preparation is tested, not proved.**
5. **Seeded defects are artificial.** They are single edits to the kernel. Real translation
   errors may be of other kinds.
6. **Enclosures are per row and after the fact.** An enclosure states where the exact result
   of the kernel lies for a row that was run. It is not a bound for inputs that were not run,
   it takes the prepared inputs as given, and it is produced by a short program that is tested
   but not formally verified: a defect in that program could yield a wrong interval marked as
   certified, as adversarial testing demonstrated in an earlier version. A row is reported as
   not certified where the program finds its exact value undefined or cannot enclose it, and a
   step is refused where its inputs or integer arithmetic cannot be represented exactly. The examples' `precision` commands also print an older
   first-order estimate of rounding error (`exactuary/errbound.py`); it is not a bound and is
   retained only for comparison.
7. **Timings** were taken on a single machine and depend on the device, particularly for
   graphics processors, where double-precision performance varies widely between products.
8. **Lean coverage is incomplete** where stated in an example's report: some obligations proved
   by Z3 are not yet proved by Lean's automation.
9. **Forms of model the library does not yet express.** Each was met in the examples and handled
   outside the proof, as the example's report states:
   - values defined backwards in time (a present value at t defined from t+1) are tested, not proved;
   - quantities derived from whole-projection totals (a premium solved from present values) are
     compared exactly but have no proof obligation;
   - originals spread over several modules, or that branch in Python on a policy attribute, need
     manual assistance in the symbolic step, which adds to what must be trusted;
   - one discount series per kernel, common to all scenarios;
   - the GPU backend returns accumulated results only, and gives no control over fused
     multiply-add, so GPU results are bounded rather than bit-identical;
   - a step with more than about thirty parameters compiles slowly.
10. **Early exit and non-finite values.** The early-exit lemma is proved over real numbers. Where
    the original produces NaN or infinity after the exit point, the kernel with early exit returns
    a finite number. The examples report the rows affected.
11. **Lean checks each obligation, not their assembly.** The induction theorems are proved once
    in general form. That a given model's obligations are exactly the hypotheses of those
    theorems is not machine-checked. Lean works over the rational numbers, where Z3 works over
    the reals; every floating-point input is rational, so the rationals are the domain that
    matters for actual inputs, but the two are not the same theory.
12. **The proof is about a reading of the original.** The original's formulas are evaluated by
    an interpreter written to accept a listed subset of Python, NumPy and pandas and to refuse
    the rest. Adversarial testing found accepted constructs that were misread; the specific
    inputs found are now refused, and related forms may still be accepted. Code supplied by the model's symbolic description is
    trusted and runs as written. Within that subset it treats values
    as real numbers or integers; missing values, infinities, index alignment and integer
    overflow are outside every proof.
13. **Seeded defects are one family.** They are single edits to the step: operators, constants,
    comparisons, branches, table indices and confusable variables. They do not include errors
    in the symbolic description, in input preparation, or in which outputs are accumulated.
    Scores are reported over defects that are runnable programs; those that are not are counted
    separately.
14. **Exit status.** In each example's run script the `exact`, `prove`, `lean` and `harness`
    commands record their checks as gates (trace comparison, early exit, prepared inputs,
    present values, proofs, exit obligations, Lean checks, exhaustive table lookups), and exit
    with a non-zero status if a gate fails or if a gate the command requires was never
    recorded. The `precision`, `baseline` and timing commands report measurements and have no
    acceptance criteria. The list of required gates is written in each script and is itself something a
    reviewer should read: a check that is computed but never recorded would not be noticed.
    Deliberately failing negative controls are excluded.
15. **Not designed against a hostile author.** See "Scope" in `method.md`. The tools catch
    errors made in good faith. Adversarial testing repeatedly found new ways to obtain a pass for
    a wrong conversion, some of which an honest author could also stumble on (for example an
    unusual use of Python iterators in the original model). The instances found are addressed and
    are tests; there is no reason to think none remain.

