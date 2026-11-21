"""Acceptance gates. Every check that a conversion must pass records its outcome here, and a run
script ends with `gates.finish()`, which exits with a non-zero status if any gate failed. A result
file that merely records a failure is not an acceptance signal; the exit status is."""
import sys
import numpy as np

_results = []

def record(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"  GATE {'PASS' if ok else 'FAIL'}: {name}{' - ' + detail if detail else ''}", flush=True)
    return bool(ok)

def results(): return list(_results)
def reset(): _results.clear()

def early_exit_agrees(run_pv, p, label="early exit"):
    """The accumulated results with early exit must equal those of the full projection, value for value.
    Rows where the full projection is itself not finite are reported, not compared: there the exit rule,
    which is proved over the real numbers, has nothing to say."""
    full, cut = np.asarray(run_pv(p, False)), np.asarray(run_pv(p, True))
    finite = np.isfinite(full).all(1)
    ok = bool((full[finite] == cut[finite]).all())
    return record(f"{label}: stopped projection equals full projection", ok,
                  f"{int(finite.sum()):,} rows compared; {int((~finite).sum()):,} rows not finite in the full projection")

def missing(required):
    """Required checks (given as the leading words of their names) that were never recorded."""
    return [r for r in required if not any(name.startswith(r) for name, _, _ in _results)]

def finish(required=()):
    """End a run: exit with a non-zero status if any recorded gate failed, or if a required gate was never
    recorded. `required` lists the leading words of the gates this command must have recorded."""
    failed = [r for r in _results if not r[1]]; absent = missing(required)
    print(f"gates: {len(_results) - len(failed)} passed, {len(failed)} failed, {len(absent)} required but not run", flush=True)
    for name, _, detail in failed: print(f"  FAILED: {name}{' - ' + detail if detail else ''}", flush=True)
    for name in absent: print(f"  NOT RUN: {name}", flush=True)
    if failed or absent: sys.exit(1)
