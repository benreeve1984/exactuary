"""Rigorous enclosures of a kernel's exact result, by interval arithmetic with outward rounding.

For each row actually run, `enclose` returns an interval [lo, hi] for every accumulated output such that
the value the kernel would produce in exact real arithmetic, on the float64 inputs given, lies in it.
Any computed result v (CPU or GPU, double or single precision, any summation order, with or without
fused multiply-add) then differs from that exact value by at most max(v - lo, hi - v): a bound obtained
without modelling the arithmetic of the machine that produced v.

How the enclosure is kept valid:
  * every + - * / is rounded outward (one step with `nextafter`, or none when an error-free
    transformation shows the result exact), which also covers underflow; overflow widens to infinity;
  * a comparison whose two sides overlap is undecided, and `sel` then returns the hull of both arms;
  * the step is evaluated node by node by this module (`exactuary/lang.py`), never by Python: every
    constant, every integer operation and every division passes through it, so nothing is rounded
    before it is tracked. Integers are exact and are refused if they could overflow; `/` is always real
    division of enclosures; a real literal such as 0.01 stands for the decimal number written;
  * powers, exponentials and logarithms use interval arithmetic in `mpmath`;
  * a row is reported as not certified (infinite interval) if its inputs contain NaN or infinity, if it
    divides by an interval containing zero, or if an integer-valued choice depends on an undecided
    comparison.

What it is not: a bound for all possible inputs (it is computed per row, after the fact); a statement
about input preparation (tables and series are taken as given, including any powers hoisted into them);
or a machine-checked proof (this module, NumPy's IEEE-754 double arithmetic and mpmath are trusted;
`tests/test_interval.py` checks the module against exact rational arithmetic).
"""
from fractions import Fraction
import numpy as np
from .spec import check_spec
from . import lang

INF = np.inf
def _dn(x): return np.nextafter(x, -INF)
def _up(x): return np.nextafter(x, INF)

class NotCertifiable(Exception):
    """The step uses something for which no enclosure can be given."""

class _Ctx:
    undefined = False       # rows whose exact value is not a real number, or could not be enclosed
    undecided = 0           # comparisons that the enclosures could not decide (both arms were kept)
    @classmethod
    def flag(cls, mask): cls.undefined = cls.undefined | np.asarray(mask)

class Tri:
    """A truth value known only up to its enclosure: definitely true, definitely false, or undecided."""
    def __init__(s, t, f): s.t, s.f = np.asarray(t), np.asarray(f)
    @staticmethod
    def lift(c): return c if isinstance(c, Tri) else Tri(np.asarray(c, bool), ~np.asarray(c, bool))
    def __and__(s, o): o = Tri.lift(o); return Tri(s.t & o.t, s.f | o.f)
    __rand__ = __and__
    def __invert__(s): return Tri(s.f, s.t)
    def __bool__(s): raise NotCertifiable("a Python branch on a computed value")

class IV:
    """An interval [lo, hi] of float64 arrays known to contain an exact real value."""
    __array_priority__ = 1000
    __hash__ = None
    def __init__(s, lo, hi, alias=()):
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        bad = np.isnan(lo) | np.isnan(hi)                       # inf - inf, 0 * inf at an end point: nothing known
        if bad.any(): lo, hi = np.where(bad, -INF, lo), np.where(bad, INF, hi)
        s.lo, s.hi = lo, hi
        s.alias = list(alias)       # [(other IV, mask)]: where mask holds, this is exactly the same real number as `other`
    @staticmethod
    def lift(x):
        if isinstance(x, IV): return x
        if isinstance(x, Tri): raise NotCertifiable("a truth value used as a number")
        a = np.asarray(x)
        if a.dtype.kind == "b": raise NotCertifiable("a truth value used as a number")
        if a.dtype.kind in "iu":
            if a.size and np.abs(a).max() > 2 ** 53: raise NotCertifiable("integer beyond 2^53")
            a = a.astype(np.float64)
        elif a.dtype != np.float64: raise NotCertifiable(f"inputs must be float64, not {a.dtype}")
        if not np.isfinite(a).all(): _Ctx.flag(~np.isfinite(a))
        return IV(a, a)
    def _same(s, o):
        """Mask where `s` and `o` are known to be the same real number (so that s - o is exactly zero)."""
        m = np.zeros(np.broadcast(s.lo, o.lo).shape, bool)
        for r1, m1 in s.alias + [(s, True)]:
            for r2, m2 in o.alias + [(o, True)]:
                if r1 is r2: m = m | (np.asarray(m1) & np.asarray(m2))
        return m
    def __add__(s, o):
        o = IV.lift(o); lo, hi = _add(s.lo, o.lo, False), _add(s.hi, o.hi, True)
        al = [(s, (o.lo == 0) & (o.hi == 0)), (o, (s.lo == 0) & (s.hi == 0))]
        return IV(lo, hi, al)
    __radd__ = __add__
    def __neg__(s): return IV(-s.hi, -s.lo)
    def __sub__(s, o):
        o = IV.lift(o); z = s._same(o)
        lo, hi = _add(s.lo, -o.hi, False), _add(s.hi, -o.lo, True)
        return IV(np.where(z, 0.0, lo), np.where(z, 0.0, hi), [(s, (o.lo == 0) & (o.hi == 0))])
    def __rsub__(s, o): return IV.lift(o) - s
    def __mul__(s, o):
        o = IV.lift(o)
        with np.errstate(all="ignore"):
            c = np.stack(np.broadcast_arrays(s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi))
        nan = np.isnan(c).any(0)                                 # 0 * inf at an end point
        lo, hi = _dn(np.nanmin(np.where(np.isnan(c), INF, c), 0)), _up(np.nanmax(np.where(np.isnan(c), -INF, c), 0))
        lo, hi = np.where(nan, -INF, lo), np.where(nan, INF, hi)
        z = ((s.lo == 0) & (s.hi == 0)) | ((o.lo == 0) & (o.hi == 0))      # exactly zero times any real number is zero
        one_s, one_o = (s.lo == 1) & (s.hi == 1), (o.lo == 1) & (o.hi == 1)
        lo, hi = np.where(one_s, o.lo, np.where(one_o, s.lo, lo)), np.where(one_s, o.hi, np.where(one_o, s.hi, hi))
        al = [(o, one_s), (s, one_o)] + [(r, m & one_s) for r, m in o.alias] + [(r, m & one_o) for r, m in s.alias]
        return IV(np.where(z, 0.0, lo), np.where(z, 0.0, hi), al)
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = IV.lift(o); ok = (o.lo > 0) | (o.hi < 0)
        if not np.all(ok): _Ctx.flag(~ok)                        # the divisor may be zero: the quotient is not defined
        with np.errstate(all="ignore"):
            c = np.stack(np.broadcast_arrays(s.lo / o.lo, s.lo / o.hi, s.hi / o.lo, s.hi / o.hi))
        lo, hi = _dn(np.min(c, 0)), _up(np.max(c, 0))
        z = (s.lo == 0) & (s.hi == 0) & ok
        return IV(np.where(ok, np.where(z, 0.0, lo), -INF), np.where(ok, np.where(z, 0.0, hi), INF))
    def __rtruediv__(s, o): return IV.lift(o) / s
    def _cmp(s, o, op):
        o = IV.lift(o)
        if op == "lt": t, f = s.hi < o.lo, s.lo >= o.hi
        elif op == "le": t, f = s.hi <= o.lo, s.lo > o.hi
        else: t, f = (s.lo == s.hi) & (o.lo == o.hi) & (s.lo == o.lo), (s.hi < o.lo) | (s.lo > o.hi)
        t = t | (s._same(o) if op in ("le", "eq") else False)
        _Ctx.undecided += int(np.sum(~(t | f)))
        return Tri(t, f)
    def __lt__(s, o): return s._cmp(o, "lt")
    def __le__(s, o): return s._cmp(o, "le")
    def __gt__(s, o): return IV.lift(o)._cmp(s, "lt")
    def __ge__(s, o): return IV.lift(o)._cmp(s, "le")
    def __eq__(s, o): return s._cmp(o, "eq")
    def __ne__(s, o): return ~s._cmp(o, "eq")

def _add(a, b, upper):
    """a + b rounded toward +inf (upper) or -inf; exact, with no widening, when the sum is representable."""
    with np.errstate(all="ignore"):
        s = a + b; bb = s - a; err = (a - (s - bb)) + (b - bb)   # TwoSum: s + err is the exact sum when all are finite
    exact = np.isfinite(s) & (err == 0)
    if upper: return np.where(exact | (err < 0), s, _up(s))
    return np.where(exact | (err > 0), s, _dn(s))

def _literal(text):
    """The decimal number written in the source, enclosed by the two floats either side of it."""
    f = float(text); exact, q = Fraction(f), Fraction(text)
    if exact == q: return IV(f, f)
    return IV(f, _up(f)) if exact < q else IV(_dn(f), f)

def _mp(fn, *ivs):
    """Apply an mpmath interval function elementwise and round its result outward to float64."""
    try: from mpmath import iv
    except ImportError: raise NotCertifiable("powers, exponentials and logarithms need the `mpmath` package")
    iv.prec = 113
    arrs = np.broadcast_arrays(*[x for v in ivs for x in (v.lo, v.hi)]); shape = arrs[0].shape
    flat = [a.ravel() for a in arrs]; lo, hi = np.empty(flat[0].size), np.empty(flat[0].size)
    for k in range(flat[0].size):
        args = [iv.mpf([float(flat[2 * j][k]), float(flat[2 * j + 1][k])]) for j in range(len(ivs))]
        try:
            r = fn(iv, *args); lo[k], hi[k] = _dn(float(r.a)), _up(float(r.b))
        except Exception: lo[k], hi[k] = -INF, INF
    return IV(lo.reshape(shape), hi.reshape(shape))

_I64 = np.iinfo(np.int64)
def _is_int(x):
    if isinstance(x, (bool, np.bool_)): return False
    return isinstance(x, (int, np.integer)) or (isinstance(x, np.ndarray) and x.dtype.kind in "iu")
def _is_bool(x): return isinstance(x, (bool, np.bool_, Tri)) or (isinstance(x, np.ndarray) and x.dtype.kind == "b")
def _mag(x):
    """Largest absolute value of an integer or integer array, as an unbounded Python integer."""
    if isinstance(x, np.ndarray):
        if x.size == 0: return 0
        return max(abs(int(x.min())), abs(int(x.max())))
    return abs(int(x))
def _int(x):
    """Normalise an integer input: Python int, or int64 array."""
    if isinstance(x, np.ndarray):
        if x.dtype.kind == "u" and x.size and int(x.max()) > _I64.max: raise NotCertifiable("integer beyond 64 bits")
        return x.astype(np.int64)
    return int(x)
def _real(x):
    """A real value as an enclosure. Integers convert exactly or are refused."""
    if isinstance(x, IV): return x
    if _is_int(x):
        if _mag(x) > 2 ** 53: raise NotCertifiable("integer beyond 2^53 used as a real number")
        return IV.lift(np.asarray(x, dtype=np.float64))
    raise NotCertifiable(f"{type(x).__name__} used as a real number")

class Domain:
    """Interval evaluation of the kernel language: integers exact, reals enclosed, truth values three-valued."""
    def integer(s, n): return int(n)
    def real(s, text): return _literal(text)
    def neg(s, a):
        if _is_int(a):
            if _mag(a) >= 2 ** 62: raise NotCertifiable("integer overflow")
            return -a
        return -_real(a)
    def binop(s, op, a, b):
        if op == "/": return _real(a) / _real(b)                 # always real division, never Python's
        if _is_int(a) and _is_int(b):
            bound = _mag(a) * _mag(b) if op == "*" else _mag(a) + _mag(b)
            if bound >= 2 ** 62: raise NotCertifiable("integer arithmetic could overflow 64 bits")
            return a + b if op == "+" else a - b if op == "-" else a * b
        a, b = _real(a), _real(b)
        return a + b if op == "+" else a - b if op == "-" else a * b
    def compare(s, op, a, b):
        if _is_bool(a) or _is_bool(b): raise NotCertifiable("comparison of a truth value")
        if _is_int(a) and _is_int(b):
            return {"==": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}[op](a, b)
        a, b = _real(a), _real(b)
        if op == "==": return a == b
        if op == "!=": return a != b
        if op == "<": return a < b
        if op == "<=": return a <= b
        return a > b if op == ">" else a >= b
    def band(s, a, b):
        if not (_is_bool(a) and _is_bool(b)): raise NotCertifiable("`band` of something that is not a truth value")
        return Tri.lift(a) & Tri.lift(b) if isinstance(a, Tri) or isinstance(b, Tri) else np.logical_and(a, b)
    def call(s, name, args):
        if name == "sel":
            c, a, b = args
            if not _is_bool(c): raise NotCertifiable("`sel` condition is not a truth value")
            if isinstance(c, (bool, np.bool_)): return a if c else b
            c = Tri.lift(c)
            if _is_bool(a) or _is_bool(b): raise NotCertifiable("`sel` between truth values")
            if _is_int(a) and _is_int(b):
                if not np.all(c.t | c.f): raise NotCertifiable("an integer choice depends on an undecided comparison")
                return np.where(c.t, a, b)
            a, b = _real(a), _real(b)
            lo = np.where(c.t, a.lo, np.where(c.f, b.lo, np.minimum(a.lo, b.lo)))
            hi = np.where(c.t, a.hi, np.where(c.f, b.hi, np.maximum(a.hi, b.hi)))
            al = [(a, c.t), (b, c.f)] + [(r, m & c.t) for r, m in a.alias] + [(r, m & c.f) for r, m in b.alias]
            return IV(lo, hi, al)
        if name in ("fmax", "fmin"):
            a, b = _real(args[0]), _real(args[1]); f = np.maximum if name == "fmax" else np.minimum
            return IV(f(a.lo, b.lo), f(a.hi, b.hi))
        if name in ("imax", "imin"):
            if not (_is_int(args[0]) and _is_int(args[1])): raise NotCertifiable(f"`{name}` of a value that is not an integer")
            return (np.maximum if name == "imax" else np.minimum)(args[0], args[1])
        if name == "idiv":
            a, k = args
            if not _is_int(a) or isinstance(k, np.ndarray) or not _is_int(k) or int(k) <= 0:
                raise NotCertifiable("`idiv` needs an integer and a positive integer constant")
            return a // int(k)
        if name == "band": return s.band(*args)
        if name == "tof":
            if not _is_int(args[0]): raise NotCertifiable("`tof` of a value that is not an integer")
            return _real(args[0])
        if name == "C": return args[0]
        if name in ("lk1", "lk2"):
            tab, idx = args[0], args[1:]
            if not (isinstance(tab, np.ndarray) and tab.ndim == len(idx) and tab.dtype == np.float64):
                raise NotCertifiable("a lookup needs a float64 table of the right rank")
            if not all(_is_int(i) for i in idx): raise NotCertifiable("a table index is not an integer")
            ok = np.ones(np.broadcast(*[np.asarray(i) for i in idx]).shape, bool)
            for d, i in enumerate(idx): ok = ok & (np.asarray(i) >= 0) & (np.asarray(i) < tab.shape[d])
            return IV.lift(np.where(ok, tab[tuple(np.where(ok, i, 0) for i in idx)], 0.0))
        if name == "fpow":
            x, y = _real(args[0]), _real(args[1]); pos = x.lo > 0
            if not np.all(pos): _Ctx.flag(~pos)
            return _mp(lambda iv, a, b: iv.exp(b * iv.log(a)), IV(np.where(pos, x.lo, 1.0), np.where(pos, x.hi, 1.0)), y)
        if name == "fexp": return _mp(lambda iv, a: iv.exp(a), _real(args[0]))
        if name == "flog":
            x = _real(args[0]); pos = x.lo > 0
            if not np.all(pos): _Ctx.flag(~pos)
            return _mp(lambda iv, a: iv.log(a), IV(np.where(pos, x.lo, 1.0), np.where(pos, x.hi, 1.0)))
        raise NotCertifiable(name)

def _input(x):
    """Classify one input: truth values and integers are exact; float64 becomes a point enclosure; anything else is refused."""
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.float64)): return IV.lift(np.float64(x))
    a = np.asarray(x)
    if a.dtype.kind == "b": return a
    if a.dtype.kind in "iu": return _int(a)
    if a.dtype == np.float64: return IV.lift(a)
    raise NotCertifiable(f"inputs must be float64, integer or boolean, not {a.dtype}")

def enclose(spec, p, rows=None):
    """Enclosures of the accumulated outputs for the rows of `p` (float64 inputs; `rows` selects a subset).
    Returns a dict:
        lo, hi      arrays [row, accumulated output]: the exact result lies in [lo, hi]
        certified   bool array [row]: False where no finite enclosure could be given (lo, hi are then infinite)
        undecided   number of comparisons the enclosures could not decide (both outcomes were kept)
    The projection is run for all periods; early exit plays no part (it is justified separately, by proof)."""
    check_spec(spec)
    step = lang.parse_step(spec.step_src); dom = Domain()
    idx = slice(None) if rows is None else np.asarray(rows)
    n = len(np.asarray(p[spec.row_args[0]])[idx])
    _Ctx.undefined, _Ctx.undecided = np.zeros(n, bool), 0
    row = [_input(np.asarray(p[a])[idx]) for a in spec.row_args]; scen = np.asarray(p["scen"])[idx]
    consts = []
    for a in spec.consts:
        c = p[a]
        if isinstance(c, np.ndarray) and c.ndim > 0:
            if c.dtype != np.float64: raise NotCertifiable(f"table {a} must be float64")
            consts.append(c)
        else: consts.append(_input(c.item() if isinstance(c, np.ndarray) else c))
    T = int(p["T"]); disc = np.asarray(p[spec.discount])
    if disc.dtype != np.float64: raise NotCertifiable("discount factors must be float64")
    zero = lambda: IV(np.zeros(n), np.zeros(n))
    state = [zero() for _ in spec.state]; acc = [zero() for _ in spec.accumulate]
    with np.errstate(all="ignore"):
        for t in range(T):
            r = lang.run(step, [t, t == 0] + row + [_input(np.asarray(p[a])[scen, t]) for a in spec.scen_series]
                         + [_input(np.asarray(p[a])[t]) for a in spec.time_series] + state + consts, dom)
            state = [_real(x) for x in r[spec.n_out:]]
            for k, nm in enumerate(spec.accumulate):
                acc[k] = acc[k] + _real(r[spec.ix(nm)]) * IV.lift(disc[t])
    lo, hi = np.column_stack([np.broadcast_to(a.lo, (n,)) for a in acc]), np.column_stack([np.broadcast_to(a.hi, (n,)) for a in acc])
    bad = np.broadcast_to(_Ctx.undefined, (n,)) | ~np.isfinite(lo).all(1) | ~np.isfinite(hi).all(1)
    lo, hi = np.where(bad[:, None], -INF, lo), np.where(bad[:, None], INF, hi)
    return {"lo": lo, "hi": hi, "certified": ~bad, "undecided": int(_Ctx.undecided)}

def distance(v, enc):
    """For computed values v [row, accumulated output] (a floating-point array): a bound on |v - exact result|
    per value (infinite where the row is not certified or v is not finite)."""
    v = np.asarray(v)
    if v.dtype not in (np.float32, np.float64): raise NotCertifiable("computed values must be float32 or float64 (which convert exactly)")
    v = v.astype(np.float64)                                     # float32 -> float64 is exact
    with np.errstate(all="ignore"):
        d = np.maximum(_up(np.abs(v - enc["lo"])), _up(np.abs(enc["hi"] - v)))
    return np.where(np.isfinite(v), d, INF)

def combine(enc, weights):
    """Enclosure of a fixed linear combination of the accumulated outputs (for example a net cashflow):
    weights is a list of numbers, one per accumulated output. Returns (lo, hi) arrays [row]."""
    lo, hi = np.asarray(enc["lo"]), np.asarray(enc["hi"])
    if lo.dtype != np.float64 or hi.dtype != np.float64: raise NotCertifiable("enclosures must be float64")
    tot = IV(np.zeros(len(lo)), np.zeros(len(lo)))
    for k, w in enumerate(weights):
        if isinstance(w, (bool, np.bool_)) or not isinstance(w, (int, float, np.integer, np.float64)): raise NotCertifiable("weights must be numbers")
        if w: tot = tot + IV(lo[:, k], hi[:, k]) * _input(w if not isinstance(w, (int, np.integer)) else np.float64(w) if abs(int(w)) <= 2 ** 53 else _real(int(w)))
    return tot.lo, tot.hi
