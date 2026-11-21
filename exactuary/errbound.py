"""A first-order ESTIMATE of rounding error, retained for comparison only. It is not a bound: use
`exactuary.interval` for that.

Running error analysis: evaluate the step in working precision while carrying, for every
value, a bound on its distance from the result of exact real arithmetic on the same inputs.
The bound is first-order rigorous for round-to-nearest arithmetic without fused multiply-add."""
import numpy as np
from .spec import make_step
from .backends import numpy_vec

class EV:
    u, ft = 2.0 ** -53, np.float64
    __array_priority__ = 1000
    def __init__(s, v, e=None): s.v = np.asarray(v, dtype=EV.ft); s.e = np.zeros(s.v.shape) if e is None else e
    @staticmethod
    def lift(x):
        if isinstance(x, EV): return x
        v = np.asarray(x, dtype=EV.ft)
        return EV(v, np.abs(np.asarray(x, dtype=np.float64) - v.astype(np.float64)))   # cost of storing in working precision
    def _r(s, v, e): return EV(v, e + EV.u * np.abs(v.astype(np.float64)))
    def __add__(s, o): o = EV.lift(o); return s._r(s.v + o.v, s.e + o.e)
    __radd__ = __add__
    def __sub__(s, o): o = EV.lift(o); return s._r(s.v - o.v, s.e + o.e)
    def __rsub__(s, o): return EV.lift(o) - s
    def __mul__(s, o):
        o = EV.lift(o); a, b = np.abs(s.v.astype(np.float64)), np.abs(o.v.astype(np.float64))
        return s._r(s.v * o.v, a * o.e + b * s.e + s.e * o.e)
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = EV.lift(o); q = s.v / o.v
        room = np.abs(o.v.astype(np.float64)) - o.e          # the divisor's distance from zero, at worst
        e = np.where(room > 0, (s.e + np.abs(q.astype(np.float64)) * o.e) / np.where(room > 0, room, 1.0), np.inf)
        e = np.where((s.e == 0) & (o.e == 0), 0.0, e)        # exact operands: only the rounding of the quotient
        return s._r(q, e)
    def __rtruediv__(s, o): return EV.lift(o) / s
    def __neg__(s): return EV(-s.v, s.e)
    # Comparisons use the working-precision values. Where the two sides are closer than their bounds,
    # exact arithmetic might have decided otherwise; such cases are counted in EV.uncertain.
    uncertain = 0
    def _cmp(s, o, f):
        o = EV.lift(o); EV.uncertain += int(np.sum(np.abs(s.v.astype(np.float64) - o.v.astype(np.float64)) < s.e + o.e))
        return f(s.v, o.v)
    def __lt__(s, o): return s._cmp(o, np.less)
    def __le__(s, o): return s._cmp(o, np.less_equal)
    def __gt__(s, o): return s._cmp(o, np.greater)
    def __ge__(s, o): return s._cmp(o, np.greater_equal)
    def __eq__(s, o): return s._cmp(o, np.equal)
    def __ne__(s, o): return s._cmp(o, np.not_equal)
    __hash__ = None

def helpers():
    h = numpy_vec.helpers(); L = EV.lift
    def sel(c, a, b): a, b = L(a), L(b); return EV(np.where(c, a.v, b.v), np.where(c, a.e, b.e))
    def fmax(a, b): a, b = L(a), L(b); return EV(np.maximum(a.v, b.v), np.maximum(a.e, b.e))
    def fmin(a, b): a, b = L(a), L(b); return EV(np.minimum(a.v, b.v), np.maximum(a.e, b.e))
    def f64(x): return np.abs(x.v.astype(np.float64))
    def fpow(x, y):          # first-order bound in both arguments; allows 1 ulp for the library routine
        x, y = L(x), L(y); v = np.power(x.v, y.v); av, yv = np.abs(v.astype(np.float64)), y.v.astype(np.float64)
        ex = np.where(x.e > 0, np.abs(yv) * f64(x) ** (yv - 1) * x.e, 0.0)
        ey = np.where(y.e > 0, av * np.abs(np.log(np.maximum(f64(x), 1e-300))) * y.e, 0.0)
        return EV(v, ex + ey + 2 * EV.u * av)
    def fexp(x): x = L(x); v = np.exp(x.v); return EV(v, np.abs(v.astype(np.float64)) * (np.expm1(x.e) + 2 * EV.u))
    def flog(x): x = L(x); v = np.log(x.v); return EV(v, x.e / np.maximum(f64(x) - x.e, 1e-300) + 2 * EV.u * np.abs(v.astype(np.float64)))
    return dict(h, sel=sel, fmax=fmax, fmin=fmin, fpow=fpow, fexp=fexp, flog=flog, tof=lambda a: L(a.astype(np.float64)), C=L, ONE=1.0, ZERO=0.0,
                lk1=lambda *a: L(h["lk1"](*a)), lk2=lambda *a: L(h["lk2"](*a)))

def run_pv_bounds(spec, p64, ftype="float64"):
    """p64: inputs prepared in float64. Returns (values, bounds), each [row, accumulated output]:
    the accumulated outputs computed in `ftype`, and for each a bound on |value - exact real result|."""
    EV.ft = np.float32 if ftype == "float32" else np.float64; EV.u = 2.0 ** (-24 if ftype == "float32" else -53); EV.uncertain = 0
    from .spec import check_spec
    from . import lang
    import operator as _op
    check_spec(spec)
    step = make_step(helpers(), spec.step_src); L = EV.lift; n = len(p64[spec.row_args[0]])
    class _Plain:                                   # the exit rule evaluated on working-precision values
        integer = staticmethod(int); real = staticmethod(float); neg = staticmethod(_op.neg)
        binop = staticmethod(lambda op, a, b: {"+": _op.add, "-": _op.sub, "*": _op.mul, "/": _op.truediv}[op](a, b))
        compare = staticmethod(lambda op, a, b: {"==": _op.eq, "!=": _op.ne, "<": _op.lt, "<=": _op.le, ">": _op.gt, ">=": _op.ge}[op](a, b))
        band = staticmethod(lambda a, b: np.logical_and(a, b))
    exit_tree = lang.parse_exit(spec.exit_when, spec.outputs, spec.row_args) if spec.exit_when else None
    fl = lambda a: L(a) if isinstance(a, np.ndarray) and a.dtype == np.float64 else a
    rows = [fl(p64[a]) for a in spec.row_args]; consts = [p64[a] for a in spec.consts]
    state = [L(np.zeros(n)) for _ in spec.state]; acc = [L(np.zeros(n)) for _ in spec.accumulate]
    alive = np.ones(n, bool)
    with np.errstate(all="ignore"):
        for t in range(p64["T"]):
            r = step(t, t == 0, *rows, *[L(p64[a][p64["scen"], t]) for a in spec.scen_series],
                     *[L(p64[a][t]) for a in spec.time_series], *state, *consts)
            state = list(r[spec.n_out:]); dsc = L(p64[spec.discount][t])
            for k, nm in enumerate(spec.accumulate):
                new = acc[k] + L(r[spec.ix(nm)]) * dsc
                acc[k] = EV(np.where(alive, new.v, acc[k].v), np.where(alive, new.e, acc[k].e))
            if exit_tree is not None:      # mirror the kernel's early exit
                env = {"OUT": {nm: (r[k].v if isinstance(r[k], EV) else r[k]) for k, nm in enumerate(spec.outputs)},
                       "ROW": {a: p64[a] for a in spec.row_args}, "t": t}
                alive = alive & ~np.asarray(lang.evaluate(exit_tree, env, _Plain), bool)
                if not alive.any(): break
    return np.column_stack([c.v for c in acc]).astype(np.float64), np.column_stack([c.e for c in acc])
