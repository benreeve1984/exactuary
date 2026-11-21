"""CPU backend: the step compiled with Numba; one policy-scenario per loop iteration,
all its months in sequence, rows in parallel across cores."""
import math
import numpy as np
from ..spec import make_step

def helpers(ftype="float64"):
    import numba as nb
    j = nb.njit(inline="always", error_model="numpy")
    cast = np.float32 if ftype == "float32" else np.float64
    zero = cast(0)
    @j
    def sel(c, a, b): return a if c else b
    @j
    def fmax(a, b):                 # as np.maximum: NaN in either argument gives NaN
        if a != a: return a
        if b != b: return b
        return a if a > b else b
    @j
    def fmin(a, b):
        if a != a: return a
        if b != b: return b
        return a if a < b else b
    @j
    def imax(a, b): return a if a > b else b
    @j
    def imin(a, b): return a if a < b else b
    @j
    def band(a, b): return (a != 0) and (b != 0)      # a truth value, as np.logical_and gives
    @j
    def idiv(a, k): return a // k
    @j
    def tof(a): return cast(a)
    @j
    def C(x): return cast(x)
    @j
    def fpow(x, y): return x ** y
    @j
    def fexp(x): return math.exp(x)
    @j
    def flog(x): return math.log(x)
    @j
    def lk1(tab, i):
        if i < 0 or i >= tab.shape[0]: return zero
        return tab[i]
    @j
    def lk2(tab, i, k):
        if i < 0 or i >= tab.shape[0] or k < 0 or k >= tab.shape[1]: return zero
        return tab[i, k]
    return dict(sel=sel, fmax=fmax, fmin=fmin, imax=imax, imin=imin, band=band, idiv=idiv, tof=tof, C=C, fpow=fpow, fexp=fexp, flog=flog,
                ONE=cast(1), ZERO=zero, lk1=lk1, lk2=lk2)

def signature(spec):
    return ", ".join([f"R_{a}" for a in spec.row_args] + ["scen"] + [f"S_{a}" for a in spec.scen_series]
                     + [f"T_{a}" for a in spec.time_series] + ["DISC", "T"] + [f"K_{a}" for a in spec.consts])

def call(spec):
    a = ([f"R_{x}[i]" for x in spec.row_args] + [f"S_{x}[sc, t]" for x in spec.scen_series]
         + [f"T_{x}[t]" for x in spec.time_series] + [f"st{k}" for k in range(len(spec.state))]
         + [f"K_{x}" for x in spec.consts])
    return f"step(t, t == 0, {', '.join(a)})"

def args_of(spec, p):
    return ([p[a] for a in spec.row_args] + [p["scen"]] + [p[a] for a in spec.scen_series]
            + [p[a] for a in spec.time_series] + [p[spec.discount], p["T"]] + [p[a] for a in spec.consts])

def loop_body(spec, indent):
    ns, na = len(spec.state), len(spec.accumulate)
    pad = " " * indent
    lines = [f"r = {call(spec)}"] + [f"st{k} = r[{spec.n_out + k}]" for k in range(ns)]
    return pad + ("\n" + pad).join(lines)

def build(spec, ftype="float64", parallel=True):
    """Returns (run_pv, run_trace).
    run_pv(p, early_exit=True) -> array [row, accumulated output]
    run_trace(p)               -> array [output, month, row]"""
    import numba as nb
    from ..spec import check_spec
    check_spec(spec); h = helpers(ftype)
    step = nb.njit(inline="always", error_model="numpy")(make_step(h, spec.step_src))
    ns, na = len(spec.state), len(spec.accumulate)
    init = "; ".join([f"st{k} = ZERO" for k in range(ns)] + [f"a{k} = ZERO" for k in range(na)])
    acc = "\n".join(f"            a{k} += r[{spec.ix(n)}] * dsc" for k, n in enumerate(spec.accumulate))
    ex = f"            if early_exit and ({spec.exit_expr()}):\n                break" if spec.exit_when else ""
    pv = f'''
def run_pv({signature(spec)}, early_exit):
    n = R_{spec.row_args[0]}.shape[0]
    out = np.zeros((n, {na}), dtype=DISC.dtype)
    for i in prange(n):
        {init}
        sc = scen[i]
        for t in range(T):
{loop_body(spec, 12)}
            dsc = DISC[t]
{acc}
{ex}
        {'; '.join(f'out[i, {k}] = a{k}' for k in range(na))}
    return out
'''
    tr = f'''
def run_trace({signature(spec)}):
    n = R_{spec.row_args[0]}.shape[0]
    out = np.zeros(({spec.n_out}, T, n), dtype=DISC.dtype)
    for i in prange(n):
        {'; '.join(f'st{k} = ZERO' for k in range(ns))}
        sc = scen[i]
        for t in range(T):
{loop_body(spec, 12)}
{chr(10).join(f"            out[{k}, t, i] = r[{k}]" for k in range(spec.n_out))}
    return out
'''
    g = {"__builtins__": {"range": range}, "np": np, "prange": nb.prange, "step": step, "ZERO": h["ZERO"]}
    exec(pv, g); exec(tr, g)
    jit = nb.njit(parallel=parallel, error_model="numpy")   # division by zero gives inf/NaN, as NumPy does
    f_pv, f_tr = jit(g["run_pv"]), jit(g["run_trace"])
    return (lambda p, early_exit=True: f_pv(*args_of(spec, p), early_exit)), (lambda p: f_tr(*args_of(spec, p)))
