"""NumPy backend: all rows at once, a Python loop over months. Slow but dependency-free;
used for mutation testing, coverage and as the carrier for the rounding-error analysis."""
import numpy as np
from ..spec import make_step

def helpers(xp=np):
    def lk1(tab, i):
        ok = (i >= 0) & (i < tab.shape[0]); return xp.where(ok, tab[xp.where(ok, i, 0)], 0)
    def lk2(tab, i, j):
        ok = (i >= 0) & (i < tab.shape[0]) & (j >= 0) & (j < tab.shape[1])
        return xp.where(ok, tab[xp.where(ok, i, 0), xp.where(ok, j, 0)], 0)
    return dict(sel=lambda c, a, b: xp.where(c, a, b), fmax=xp.maximum, fmin=xp.minimum, imax=xp.maximum,
                imin=xp.minimum, band=lambda a, b: xp.logical_and(a, b), idiv=lambda a, k: a // k,
                tof=lambda a: a.astype(np.float64), fpow=xp.power, fexp=xp.exp, flog=xp.log, C=lambda x: x, ONE=1.0, ZERO=0.0, lk1=lk1, lk2=lk2)

def step_args(spec, p, t, state):
    return ([p[a] for a in spec.row_args] + [p[a][p["scen"], t] for a in spec.scen_series]
            + [p[a][t] for a in spec.time_series] + list(state) + [p[a] for a in spec.consts])

def run_trace(spec, p, names=None, extra_helpers=None):
    """Returns array [output, month, row] for the requested outputs (default: all). The whole specification is
    validated, including the exit rule, although the trace always runs every period."""
    from ..spec import check_spec
    check_spec(spec)
    step = make_step({**helpers(), **(extra_helpers or {})}, spec.step_src)
    n, T = len(p[spec.row_args[0]]), p["T"]
    keep = list(range(spec.n_out)) if names is None else [spec.ix(x) for x in names]
    out = np.zeros((len(keep), T, n))
    state = [np.zeros(n) for _ in spec.state]
    with np.errstate(all="ignore"):
        for t in range(T):
            r = step(t, t == 0, *step_args(spec, p, t, state))
            state = list(r[spec.n_out:])
            for j, k in enumerate(keep):
                out[j, t] = r[k]
    return out
