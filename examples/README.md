# Examples

These are four open source models from the [lifelib](https://lifelib.io) project
(version 0.17.3). Each example contains a converted fast model and results for the acceleration
checks. Install the original models through lifelib and run them unchanged for comparison. The
[glossary](../README.md#glossary) explains the terms on this page.

| Example | lifelib model | Product | Period | Scenarios |
|---|---|---|---|---|
| [Savings](lifelib_savings/) | `savings/CashValue_ME` | Universal-life savings with account value, fees and guarantees | Month | Stochastic investment returns |
| [Savings with dynamic lapses](lifelib_savings_dynamic/) | `savings/CashValue_ME_EX2` | As above, with lapses that depend on the value of the guarantee | Month | Stochastic, with 5 configurations of the model's switches |
| [Term assurance](lifelib_basicterm/) | `basiclife/BasicTerm_ME` | Level term assurance | Month | One deterministic basis |
| [Traditional life, annual](lifelib_fastlife/) | `fastlife` | Term, whole life and endowment, with net-premium reserves and cash values | Year | One deterministic basis |

Each folder contains:

- `model.py`, with the portfolios, input preparation, the step function and its formal description
- `run.py`, with the commands
- `results/`, with the outputs of the last run

The scripts write large intermediate datasets to `data/`. The repository excludes that directory.

Run these commands from an example folder:

    python run.py oracle      # run the original model; record the selected intermediate values
    python run.py exact       # compare the kernel with it, value for value; time both
    python run.py harness     # exhaustive table checks, branch coverage, seeded defects
    python run.py prove       # proof of equivalence and of early exit (Z3)
    python run.py lean        # re-check the proofs in Lean (set LEAN to the executable)
    python run.py precision   # GPU and single-precision accuracy (measured)

The `harness` command tests the checks with planted errors (the code calls these "seeded
defects"). The `prove` command uses Z3, and the `lean` command uses Lean.

The scripts in [`../benchmarks`](../benchmarks/) produce rounding error bounds and timings on a
common basis for all four models.
