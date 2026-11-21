# Examples

These are 4 models from the open-source [lifelib](https://lifelib.io) project (version 0.17.3).
Each one has been converted to a fast model and checked at every level. The code calls the fast
model the "kernel". The original models are installed with lifelib and run unchanged as the
reference.

| Folder | lifelib model | Product | Period | Scenarios |
|---|---|---|---|---|
| [`lifelib_savings`](lifelib_savings/) | `savings/CashValue_ME` | Universal-life savings with account value, fees and guarantees | Month | Stochastic investment returns |
| [`lifelib_savings_dynamic`](lifelib_savings_dynamic/) | `savings/CashValue_ME_EX2` | As above, with lapses that depend on the value of the guarantee | Month | Stochastic, with 5 settings of the model's switches |
| [`lifelib_basicterm`](lifelib_basicterm/) | `basiclife/BasicTerm_ME` | Level term assurance | Month | One deterministic basis |
| [`lifelib_fastlife`](lifelib_fastlife/) | `fastlife` | Term, whole life and endowment, with net-premium reserves and cash values | Year | One deterministic basis |

Each folder contains:

- `model.py`, with the portfolios, input preparation, the kernel and its description as formulas
- `run.py`, with the commands
- `results/`, with the outputs of the last run

Large intermediate data is written to `data/`, which is not kept in the repository.

The commands are:

    python run.py oracle      # run the original model; record every intermediate value
    python run.py exact       # compare the kernel with it, value for value; time both
    python run.py harness     # exhaustive table checks, branch coverage, seeded defects
    python run.py prove       # proof of equivalence and of early exit (Z3)
    python run.py lean        # re-check the proofs in Lean (set LEAN to the executable)
    python run.py precision   # GPU and single-precision accuracy (measured)

The `harness` command plants errors in the kernel to test the checks (the code calls these
"seeded defects"). `prove` uses Z3, an automated theorem prover from Microsoft Research that
either proves a statement or finds a counterexample. `lean` uses Lean, a proof assistant that
checks every step of a proof with a small, well-studied checking program.

The scripts in [`../benchmarks`](../benchmarks/) produce guaranteed bounds on accuracy, and
timings on a common basis, for all 4 models.
