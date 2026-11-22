# Limitations

The [glossary](../README.md#glossary) explains the terms on this page. The current limitations
are as follows.

1. Scope of the examples. The examples are open source lifelib models. They
   are smaller and more regular than production models. Production models often include
   management actions, nested stochastic projections and interactions between policies. The
   examples omit these features unless their README states otherwise.
2. Exact arithmetic. The proofs establish equality in exact arithmetic. Separate checks assess
   computed results, through exact comparison or rounding error bounds.
3. Assumptions. Assumptions replace unsupported parts of each original model. A proof can be valid
   under its stated assumptions even when those assumptions describe the original model
   incorrectly. Tests assess each assumption, but the tests are finite. Review must check that the
   assumptions apply.
4. Input preparation. Input preparation has comparison tests, and proof coverage excludes it.
5. Planted errors are artificial. Each one is a single edit to the fast model. Real translation
   errors may be of other kinds.
6. Rounding error bounds apply to each row in the run, after the run. An interval states where the
   exact fast-model result lies for a row in the run. It does not cover other inputs, and it takes
   the prepared inputs as given. A short program produces it. Tests assess that program, but it
   has no formal correctness proof. An error in the program could give a wrong interval marked as
   certified.

   The interval program reports a row as uncertified if it finds the exact value undefined or
   cannot compute an interval for it. The program rejects a step if it cannot represent the
   step's inputs or integer arithmetic exactly. The examples' `precision` commands also print an
   older first-order estimate of rounding error (`exactuary/errbound.py`). This is an
   approximation for comparison only.
7. Timings. The benchmark uses one machine. Results depend on the device, particularly for
   graphics processors, where double-precision performance varies widely between products.
8. Lean. Lean reports no failures for the exported statements in the saved runs. The dynamic
   savings example has additional Z3 goals outside that exported set. Review must still connect
   each model's statements to the general induction result.
9. Current restrictions. The current implementation has the following restrictions. Of the items
   in this list, only the first two kinds of calculation lack proofs:
   - values defined backwards in time (a present value at t defined from t+1), which have exact
     comparison tests only
   - quantities derived from totals over the whole projection (a premium solved from present
     values), which have exact comparison tests only

   The other restrictions concern input forms, devices and performance:
   - original models spread over several source files, or that branch in Python on a policy
     attribute, need manual help in the symbolic step, and this adds to the dependencies
   - each fast model has one discount series, common to all scenarios
   - the GPU backend returns accumulated results only and gives no control over fused
     multiply-add (a hardware operation that multiplies and adds with a single rounding), so GPU
     accuracy uses error bounds
   - a step function with more than about 30 function inputs compiles slowly
10. Early exit and values that are not finite. The proof establishes zero future accumulated
    cashflows in exact arithmetic once the exit rule holds. Where the original model produces NaN
    or infinity after the exit point, the fast model with early exit returns a finite number. The
    model READMEs report the affected rows. In the dynamic savings stress test, early exit changes
    20 rows.
11. Assembly of Lean results. Lean checks individual statements. Review must establish their
    connection to the full projection. Lean proves the induction theorems once in general form, and
    no program checks that a given model's statements match their conditions exactly. Lean uses
    fractions, while Z3 uses real numbers. Every finite stored input is a rational number. That
    fact alone does not establish that the real-number and rational-number statements have the
    same meaning. The translation and treatment of powers, exponentials and logarithms need review.
12. Reading of the original model. An interpreter evaluates the original model's formulas. It
    accepts a listed subset of Python, NumPy and pandas and rejects the rest. Tests found
    incorrect readings, and the interpreter now rejects those cases. Related forms may still pass.
    The formal description supplies trusted code, which runs as written. Within the supported
    subset, the interpreter treats values as real numbers or integers. Missing values, infinities,
    index alignment (matching table rows by their labels) and integer overflow (a whole number
    exceeding its storage range) are outside every proof.
13. Planted errors are one family. They are single edits to the step function: operators,
    constants, comparisons, branches, table positions and easily confused variables. They exclude
    errors in the formal description, in input preparation and in the choice of accumulated
    outputs. Scores count runnable errors, and the report counts other errors separately.
14. Exit status. In each example's run script, the `exact`, `prove`, `lean` and `harness` commands
    record their checks as acceptance gates. The gates cover trace comparison, early exit,
    prepared inputs, present values, proofs, exit statements, Lean checks and exhaustive table
    lookups. Each command exits with a non-zero status if a gate fails or if the command failed to
    record a required gate. The `precision`, `baseline` and timing commands report measurements
    and have no acceptance criteria. Each script lists its required gates, and a reviewer should
    read that list. The gate system cannot detect an unregistered check. The gate system excludes
    negative controls, which are deliberately wrong cases that should fail.
15. Deliberate attacks. See "Scope" in [Method](method.md). The checks aim to detect errors made in
    good faith. Some ways to get a wrong conversion to pass could also affect an honest author, for
    example an unusual use of Python iterators in the original model. Known failure cases have
    regression tests. Other conversion errors may remain.
