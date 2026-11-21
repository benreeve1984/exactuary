"""Measuring the checks themselves: branch-outcome coverage of a test portfolio, and the share
of seeded defects that checks of increasing strength detect."""
import ast, numpy as np
from .spec import make_step
from .compare import same
from .backends import numpy_vec
from .mutate import mutants

def coverage(spec, p):
    """For every choice point in the step (sel, min/max, table lookup) record which outcomes the
    portfolio exercises. Returns (labels, [set of outcomes seen])."""
    labels = []
    class V(ast.NodeVisitor):
        def visit_Call(self, n):
            self.generic_visit(n)
            if getattr(n.func, "id", "") in ("sel", "fmax", "fmin", "imin", "imax", "lk1", "lk2"):
                labels.append(ast.unparse(n))
    for st in ast.parse(spec.step_src).body[0].body: V().visit(st)
    cov = [set() for _ in labels]; ctr = [0]; h = numpy_vec.helpers(); A = np.asarray
    def rec(outs):
        cov[ctr[0]].update(k for k, v in outs.items() if v); ctr[0] += 1
    def sel(c, a, b): rec({"true": A(c).any(), "false": (~A(c)).any()}); return h["sel"](c, a, b)
    def mm(name):
        def f(a, b): rec({"first": A(a > b).any() if "max" in name else A(a < b).any(),
                          "second": A(a < b).any() if "max" in name else A(a > b).any()}); return h[name](a, b)
        return f
    def lk1(tab, i):
        ok = (i >= 0) & (i < tab.shape[0]); rec({"in-table": A(ok).any(), "off-table": (~A(ok)).any()}); return h["lk1"](tab, i)
    def lk2(tab, i, j):
        ok = (i >= 0) & (i < tab.shape[0]) & (j >= 0) & (j < tab.shape[1])
        rec({"in-table": A(ok).any(), "off-table": (~A(ok)).any()}); return h["lk2"](tab, i, j)
    step = make_step({**h, "sel": sel, "lk1": lk1, "lk2": lk2, **{n: mm(n) for n in ("fmax", "fmin", "imin", "imax")}}, spec.step_src)
    n = len(p[spec.row_args[0]]); state = [np.zeros(n) for _ in spec.state]
    with np.errstate(all="ignore"):
        for t in range(p["T"]):
            ctr[0] = 0
            r = step(t, t == 0, *numpy_vec.step_args(spec, p, t, state)); state = list(r[spec.n_out:])
    return labels, cov

def accumulated(spec, trace, p):
    return np.column_stack([trace[spec.ix(n)].T @ p[spec.discount] for n in spec.accumulate])

_G = {}
def _init(spec, datasets):
    _G["spec"], _G["data"] = spec, datasets
def _judge(src):
    """For one version of the step: which checks flag it? True = caught. Returns None if the version is not
    a runnable program (it is then no evidence about the checks, and is counted separately)."""
    spec, res = _G["spec"].with_src(src), {}
    for tag, d in _G["data"].items():
        p = d["p"]
        try: tr = numpy_vec.run_trace(spec, p)
        except Exception: return None
        pv, opv = accumulated(spec, tr, p), d["pv"]
        with np.errstate(all="ignore"):
            fin = np.isfinite(opv)                  # rows where the original itself gives NaN/inf are left out of totals
            tot, otot = np.where(fin, pv, 0.0).sum(0), np.where(fin, opv, 0.0).sum(0)
            res[f"{tag}_totals"] = not bool(np.all(same(tot, otot) | (np.abs(tot - otot) <= 1e-6 * np.abs(otot))))
            both = np.isfinite(pv) & fin            # a tolerance applies only where both values are finite
            res[f"{tag}_pv"] = not bool(np.all(same(pv, opv) | (both & (np.abs(pv - opv) <= 1e-9 * np.maximum(np.abs(opv), 1.0)))))
        res[f"{tag}_trace"] = not all(bool(same(tr[k], d["trace"][k]).all()) for k in range(spec.n_out))
    return res

def mutation_scores(spec, datasets, swaps=None, workers=4):
    """datasets: {tag: {"p": inputs, "trace": oracle array [output, month, row]}}; the first tag is the
    'sample' portfolio, any others are added for the strongest check.
    Returns dict with the runnable mutants, a summary table (check, caught, runnable mutants), the survivors of
    the strongest check, and the mutants that were not runnable programs (excluded from the scores).
    Memory: each worker holds a copy of every dataset plus one trace; size `workers` accordingly."""
    from multiprocessing import Pool
    for d in datasets.values(): d["pv"] = accumulated(spec, d["trace"], d["p"])
    _init(spec, datasets)
    base = _judge(spec.step_src)
    if base is None or any(base.values()): raise RuntimeError(f"the unmutated step is itself flagged: {base}")
    every = mutants(spec.step_src, swaps)
    with Pool(workers, initializer=_init, initargs=(spec, datasets)) as pool:
        judged = pool.map(_judge, [m[2] for m in every], chunksize=2)
    invalid = [(m[0], m[1]) for m, v in zip(every, judged) if v is None]
    ms = [m for m, v in zip(every, judged) if v is not None]; verdicts = [v for v in judged if v is not None]
    first = next(iter(datasets))
    checks = [("portfolio totals within 1e-6, sample portfolio", lambda v: v[f"{first}_totals"]),
              ("per-policy present values within 1e-9, sample portfolio", lambda v: v[f"{first}_pv"]),
              ("every intermediate exact, sample portfolio", lambda v: v[f"{first}_trace"]),
              ("every intermediate exact, all portfolios", lambda v: any(v[f"{t}_trace"] for t in datasets))]
    table = [(name, sum(bool(f(v)) for v in verdicts), len(ms)) for name, f in checks]
    survivors = [(m[0], m[1]) for m, v in zip(ms, verdicts) if not checks[-1][1](v)]
    return {"mutants": [(m[0], m[1]) for m in ms], "table": table, "survivors": survivors, "invalid": invalid}
