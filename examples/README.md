# Examples

Four models from the open-source [lifelib](https://lifelib.io) project (version 0.17.3), each
converted to an Exactuary kernel and checked at every level. The original models are installed
with lifelib and are run unchanged as the reference.

| Folder | lifelib model | Product | Step | Scenarios |
|---|---|---|---|---|
| [`lifelib_savings`](lifelib_savings/) | `savings/CashValue_ME` | Universal-life savings with account value, fees and guarantees | Month | Stochastic investment returns |
| [`lifelib_savings_dynamic`](lifelib_savings_dynamic/) | `savings/CashValue_ME_EX2` | As above, with lapses that depend on the guarantee's value | Month | Stochastic, five configurations of switches |
| [`lifelib_basicterm`](lifelib_basicterm/) | `basiclife/BasicTerm_ME` | Level term assurance | Month | One deterministic basis |
| [`lifelib_fastlife`](lifelib_fastlife/) | `fastlife` | Term, whole life and endowment, with net-premium reserves and cash values | Year | One deterministic basis |

Each folder contains `model.py` (portfolios, input preparation, the kernel and its symbolic
description), `run.py` (the commands) and `results/` (the outputs of the last run). Large
intermediate data is written to `data/`, which is not kept in the repository.

    python run.py oracle      # run the original model; record every intermediate value
    python run.py exact       # compare the kernel with it, value for value; time both
    python run.py harness     # exhaustive table checks, branch coverage, seeded defects
    python run.py prove       # proof of equivalence and of early exit (Z3)
    python run.py lean        # re-check the proofs in Lean (set LEAN to the executable)
    python run.py precision   # GPU and single-precision accuracy (measured)

Accuracy with a guarantee, and timings on a common basis, are produced for all four models by
the scripts in [`../benchmarks`](../benchmarks/).
