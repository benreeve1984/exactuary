"""GPU backend: the same step compiled for CUDA; one policy-scenario per GPU thread."""
import math
import numpy as np
from ..spec import make_step
from .numba_cpu import signature, call, args_of, loop_body

def build(spec, ftype="float64"):
    """Returns prepare(p) -> (go, out): `go()` runs the projection on the device and waits;
    `out.copy_to_host()` gives the array [row, accumulated output]."""
    from numba import cuda
    from ..spec import check_spec
    check_spec(spec)
    d = cuda.jit(device=True, inline="always")
    cast = np.float32 if ftype == "float32" else np.float64
    zero = cast(0)
    @d
    def sel(c, a, b): return a if c else b
    @d
    def fmax(a, b):                 # as np.maximum: NaN in either argument gives NaN
        if a != a: return a
        if b != b: return b
        return a if a > b else b
    @d
    def fmin(a, b):
        if a != a: return a
        if b != b: return b
        return a if a < b else b
    @d
    def imax(a, b): return a if a > b else b
    @d
    def imin(a, b): return a if a < b else b
    @d
    def band(a, b): return (a != 0) and (b != 0)      # a truth value, as np.logical_and gives
    @d
    def idiv(a, k): return a // k
    @d
    def tof(a): return cast(a)
    @d
    def C(x): return cast(x)
    @d
    def fpow(x, y): return math.pow(x, y)
    @d
    def fexp(x): return math.exp(x)
    @d
    def flog(x): return math.log(x)
    @d
    def lk1(tab, i):
        if i < 0 or i >= tab.shape[0]: return zero
        return tab[i]
    @d
    def lk2(tab, i, k):
        if i < 0 or i >= tab.shape[0] or k < 0 or k >= tab.shape[1]: return zero
        return tab[i, k]
    h = dict(sel=sel, fmax=fmax, fmin=fmin, imax=imax, imin=imin, band=band, idiv=idiv, tof=tof, C=C, fpow=fpow, fexp=fexp, flog=flog,
             ONE=cast(1), ZERO=zero, lk1=lk1, lk2=lk2)
    step = d(make_step(h, spec.step_src))
    ns, na = len(spec.state), len(spec.accumulate)
    init = "; ".join([f"st{k} = ZERO" for k in range(ns)] + [f"a{k} = ZERO" for k in range(na)])
    acc = "\n".join(f"        a{k} += r[{spec.ix(n)}] * dsc" for k, n in enumerate(spec.accumulate))
    ex = f"        if {spec.exit_expr()}:\n            break" if spec.exit_when else ""
    code = f'''
def kern({signature(spec)}, out):
    i = cuda.grid(1)
    if i >= R_{spec.row_args[0]}.shape[0]:
        return
    {init}
    sc = scen[i]
    for t in range(T):
{loop_body(spec, 8)}
        dsc = DISC[t]
{acc}
{ex}
    {'; '.join(f'out[i, {k}] = a{k}' for k in range(na))}
'''
    g = {"__builtins__": {"range": range}, "cuda": cuda, "step": step, "ZERO": zero}; exec(code, g)
    kern = cuda.jit(g["kern"])
    def prepare(p):
        n = len(p[spec.row_args[0]])
        dev = [cuda.to_device(x) if isinstance(x, np.ndarray) else x for x in args_of(spec, p)]
        out = cuda.device_array((n, na), dtype=p[spec.discount].dtype)
        def go():
            kern[(n + 255) // 256, 256](*dev, out); cuda.synchronize()
        return go, out
    return prepare
