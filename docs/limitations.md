# Limitations

In this document, "the original model" means the lifelib model run unchanged. "The fast model"
means the rewritten version. The code calls it the "kernel" or "step". A row is one policy in
one scenario.

The proofs use Z3, an automated theorem prover from Microsoft Research that either proves a
statement or finds a counterexample. They are re-checked with Lean, a proof assistant that checks
every step of a proof with a small, well-studied checking program.

These are the known limitations.

1. Scope of the examples. The examples are open-source teaching models from the lifelib project.
   They are smaller and more regular than production models. Some features are common in
   production, such as management actions, nested stochastic projections and interactions
   between policies. These are not represented unless an example's report says otherwise.
2. Proofs concern exact arithmetic. Equality is proved over the real numbers. Agreement of the
   computed results is shown separately, by exact comparison or by a bound on rounding error.
3. Assumptions. Parts of each original model are assumed rather than run on symbols. The code
   calls these assumptions "axioms". A wrong assumption would make the related proof invalid.
   Each assumption is tested, but the tests are finite.
4. Input preparation is tested, not proved.
5. Planted errors are artificial. Each one is a single edit to the fast model. Real translation
   errors may be of other kinds.
6. Guaranteed intervals apply to each row, after the run. The code calls these intervals
   "enclosures". An interval states where the exact result of the fast model lies for a row that
   was run.
   - It is not a bound for inputs that were not run.
   - It takes the prepared inputs as given.
   - A short program produces it. That program is tested but not formally proved correct. An
     error in the program could give a wrong interval marked as certified. Adversarial testing
     showed this happening in an earlier version.

   A row is reported as not certified if the program finds its exact value undefined or cannot
   compute an interval for it. A step is rejected if its inputs or integer arithmetic cannot be
   represented exactly. The examples' `precision` commands also print an older first-order
   estimate of rounding error, a simple approximation (`exactuary/errbound.py`). It is not a bound, and is kept only for
   comparison.
7. Timings were taken on a single machine and depend on the device. This applies particularly to
   graphics processors, where double-precision performance varies widely between products.
8. Lean coverage is incomplete where an example's report says so. For some statements, Z3 has a
   proof but Lean's automation does not yet.
9. The library cannot yet express some forms of model. Each of these came up in the examples. Each
   was handled outside the proof, as the example's report states. They are:
   - values defined backwards in time (a present value at t defined from t+1), which are tested
     but not proved
   - quantities derived from totals over the whole projection (a premium solved from present
     values), which are compared exactly but have no statement to be proved
   - original models spread over several modules, or that branch in Python on a policy
     attribute, which need manual help in the formula step and so add to what must be trusted
   - only one discount series per kernel, common to all scenarios
   - the GPU backend, which returns accumulated results only and gives no control over fused
     multiply-add (a hardware operation that multiplies and adds with a single rounding), so GPU results are bounded rather than identical to the bit
   - a step with more than about 30 parameters, which compiles slowly
10. Early exit and values that are not finite numbers. The proof that early exit is safe uses
    real numbers. Sometimes the original model produces NaN (not a number) or infinity after the
    exit point. In that case the fast model with early exit returns a finite number. The
    examples report the rows affected.
11. Lean checks each statement, not how they fit together. The induction theorems are proved
    once in general form. No machine checks that a given model's statements are exactly the
    conditions of those theorems. Lean works with rational numbers (fractions), while Z3 works
    with real numbers. Every floating-point input is rational, so the rationals are what matter
    for actual inputs. But the 2 theories are not the same.
12. The proof is about one reading of the original model. An interpreter evaluates the original
    model's formulas. It is written to accept a listed subset of Python, NumPy and pandas and to
    reject the rest. Adversarial testing found accepted constructs that were misread. The
    specific inputs found are now rejected, but related forms may still be accepted. Code
    supplied by the model's description as formulas is trusted and runs as written. Within the
    accepted subset, the interpreter treats values as real numbers or integers. Missing values,
    infinities, index alignment and integer overflow are outside every proof.
13. Planted errors are one family. They are single edits to the step: operators, constants,
    comparisons, branches, table indices and variables that are easy to confuse. They do not
    include errors in the description as formulas, in input preparation, or in which outputs are
    accumulated. Scores are reported over planted errors that give programs that run. Those that
    do not run are counted separately.
14. Exit status. In each example's run script, the `exact`, `prove`, `lean` and `harness`
    commands record their checks as gates. The gates are trace comparison, early exit, prepared
    inputs, present values, proofs, exit statements, Lean checks and exhaustive table lookups.
    Each command exits with a non-zero status if a gate fails, or if a gate the command needs was
    never recorded. The `precision`, `baseline` and timing commands report measurements and have
    no acceptance criteria. Each script lists its required gates, and a reviewer should read that
    list. A check that is computed but never recorded would not be noticed. Negative controls,
    which are designed to fail, are excluded.
15. Not designed against a hostile author. See "Scope" in `method.md`. The tools catch errors
    made in good faith. Adversarial testing repeatedly found new ways to get a wrong conversion to
    pass. An honest author could also stumble on some of these, for example through an unusual
    use of Python iterators in the original model. The cases found have been dealt with and are
    now tests. There is no reason to think that none remain.
