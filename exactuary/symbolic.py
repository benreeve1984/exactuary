"""Symbolic equivalence. The original model's formulas are executed symbolically, as written,
and compared with the kernel's step over the real numbers: once for month 0 and once for an
arbitrary month t >= 1 under the assumption that the kernel's carried state is correct. Together
with the induction in lean/Glue.lean this covers every policy and every month.

What must be trusted: Z3, this module, and the model's `SymbolicSpec` (its axioms in particular).

The interpreter models a deliberately small subset of Python, NumPy and pandas, listed here. Anything
outside it raises `Unsupported`; it is never guessed.
  * + - * / on integers and reals; ** as an uninterpreted function; unary minus.
  * // only for an integer divided by a positive integer constant (where floor, truncation and
    Euclidean division agree with Z3's).
  * comparisons; & | ~ on truth values; a truth value multiplied by a number.
  * np.maximum, np.minimum, np.exp, np.log (the last two uninterpreted).
  * Series.where(cond, other) with `other` given explicitly.
  * `t == 0` on the time argument; no other Python branch on a symbolic value or on time.
Values are single unlabelled real numbers or integers. Index alignment, missing values (NaN),
infinities, dtype promotion and integer overflow are not modelled: the proofs say nothing about them."""
import ast, time, operator, z3
from fractions import Fraction
from . import lang
from dataclasses import dataclass, field
from typing import Callable
from .spec import make_step
from .oracle import parse_cell

class Unsupported(NotImplementedError):
    """The original uses a construct whose meaning this interpreter does not model."""
class SpecError(ValueError):
    """The symbolic description of a model is not acceptable as written."""
_MISSING = object()
REAL, INT = z3.RealSort(), z3.IntSort()
POW = z3.Function("POW", REAL, REAL, REAL)     # x ** y, left uninterpreted
EXP = z3.Function("EXP", REAL, REAL)
LOG = z3.Function("LOG", REAL, REAL)

def lift(x):
    if isinstance(x, Sym): return x.e
    if isinstance(x, Time): return x.expr()
    if isinstance(x, bool): return z3.BoolVal(x)
    if isinstance(x, int): return z3.IntVal(x)
    if isinstance(x, Fraction): return z3.RealVal(x.numerator) if x.denominator == 1 else z3.Q(x.numerator, x.denominator)
    if isinstance(x, float): return z3.RealVal(repr(x))
    raise TypeError(f"cannot treat {type(x)} symbolically")
def real(e): return z3.ToReal(e) if z3.is_int(e) else e
def ite(c, a, b):
    if z3.is_int(a) != z3.is_int(b): a, b = real(a), real(b)
    return z3.If(c, a, b)
def _arith(a, b, f):
    a, b = lift(a), lift(b)
    if f == "mul":                              # a literal zero times anything is zero (over the real numbers)
        for z, o in ((a, b), (b, a)):
            if (z3.is_int_value(z) or z3.is_rational_value(z)) and z3.is_true(z3.simplify(z == 0)):
                return z3.RealVal(0) if (z3.is_real(z) or z3.is_real(o)) else z3.IntVal(0)
    if z3.is_bool(a) or z3.is_bool(b):          # True/False used as 1/0 in a product
        if f != "mul": raise TypeError("booleans may only be multiplied")
        other = b if z3.is_bool(a) else a
        if (z3.is_int_value(other) or z3.is_rational_value(other)) and z3.simplify(other == 0).eq(z3.BoolVal(True)):
            return other                          # (condition) * 0 is 0 whichever way the condition falls
        return ite(a, b, z3.IntVal(0)) if z3.is_bool(a) else ite(b, a, z3.IntVal(0))
    if z3.is_int(a) != z3.is_int(b): a, b = real(a), real(b)
    return a + b if f == "add" else a - b if f == "sub" else a * b

class Sym:
    """A symbolic value that behaves like the numbers, Series and arrays in a model's formulas."""
    __array_priority__ = 1000
    __hash__ = None
    def __init__(s, e): s.e = e
    def __add__(s, o): return Sym(_arith(s, o, "add"))
    def __radd__(s, o): return Sym(_arith(o, s, "add"))
    def __sub__(s, o): return Sym(_arith(s, o, "sub"))
    def __rsub__(s, o): return Sym(_arith(o, s, "sub"))
    def __mul__(s, o): return Sym(_arith(s, o, "mul"))
    def __rmul__(s, o): return Sym(_arith(o, s, "mul"))
    def __neg__(s): return Sym(-lift(s))
    def __truediv__(s, o): return Sym(real(lift(s)) / real(lift(o)))
    def __rtruediv__(s, o): return Sym(real(lift(o)) / real(lift(s)))
    def __floordiv__(s, o): return Sym(_floordiv(lift(s), o))
    def __pow__(s, o): return Sym(POW(real(lift(s)), real(lift(o))))
    def __rpow__(s, o): return Sym(POW(real(lift(o)), real(lift(s))))
    def __eq__(s, o): return Sym(lift(s) == lift(o))
    def __ne__(s, o): return Sym(lift(s) != lift(o))
    def __lt__(s, o): return Sym(_cmp(s, o, "lt"))
    def __gt__(s, o): return Sym(_cmp(s, o, "gt"))
    def __le__(s, o): return Sym(_cmp(s, o, "le"))
    def __ge__(s, o): return Sym(_cmp(s, o, "ge"))
    def __and__(s, o): return Sym(z3.And(lift(s), lift(o)))
    def __or__(s, o): return Sym(z3.Or(lift(s), lift(o)))
    __rand__, __ror__ = __and__, __or__
    def __invert__(s): return Sym(z3.Not(lift(s)))
    def __bool__(s): raise RuntimeError("a Python branch depends on a symbolic value; axiomatise that cell")
    def where(s, cond, other=_MISSING):         # pandas Series.where
        if other is _MISSING:
            raise Unsupported("Series.where without `other` gives missing values (NaN), which are not modelled")
        return Sym(ite(lift(cond), lift(s), lift(other)))
def _floordiv(a, o):
    if isinstance(o, bool) or not isinstance(o, int) or o <= 0:
        raise Unsupported("floor division is modelled only for a positive integer constant divisor")
    if not z3.is_int(a): raise Unsupported("floor division of a real value is not modelled")
    return a / z3.IntVal(o)
def _cmp(a, b, op):
    a, b = lift(a), lift(b)
    if z3.is_int(a) != z3.is_int(b): a, b = real(a), real(b)
    return {"lt": a < b, "gt": a > b, "le": a <= b, "ge": a >= b}[op]
def smax(a, b):
    a, b = lift(a), lift(b)
    if z3.is_int(a) != z3.is_int(b): a, b = real(a), real(b)
    return Sym(z3.If(a > b, a, b))
def smin(a, b):
    a, b = lift(a), lift(b)
    if z3.is_int(a) != z3.is_int(b): a, b = real(a), real(b)
    return Sym(z3.If(a < b, a, b))

TVAR = z3.Int("t")
class Time:
    """Month t+k, for a symbolic month t >= 1."""
    def __init__(s, k): s.k = k
    def expr(s): return TVAR + s.k
    def __add__(s, o): return Time(s.k + o)
    def __sub__(s, o): return Time(s.k - o)
    def __eq__(s, o):
        if o != 0 or s.k < 0: raise RuntimeError("unsupported comparison on symbolic time")
        return False
    def __hash__(s): return hash(("T", s.k))
    def __bool__(s): raise Unsupported("a Python branch depends on the time argument; only `t == 0` is modelled")
    def __floordiv__(s, o): return Sym(_floordiv(s.expr(), o))
    def __truediv__(s, o): return Sym(z3.ToReal(s.expr()) / o)
    def __repr__(s): return f"t{s.k:+d}"
def tval(t):
    """Month number as a symbolic integer, for a Time or a concrete int."""
    return Sym(t.expr()) if isinstance(t, Time) else Sym(z3.IntVal(t))

class Enum:
    """A categorical column: comparing it with a label gives the corresponding boolean symbol."""
    __hash__ = None
    def __init__(s, flags): s.flags = flags
    def __eq__(s, o): return s.flags[o]

def _exact(x): return isinstance(x, (int, Fraction)) and not isinstance(x, bool)
def _norm(x):
    """A Python float can enter only from a model's symbolic description; it stands for the decimal it prints as."""
    return Fraction(repr(x)) if isinstance(x, float) else x
_NORETURN = object()
_PLAIN = (int, Fraction, bool, str, type(None), list, tuple, dict, range)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.BitAnd: operator.and_, ast.BitOr: operator.or_}
_CMPS = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}

class Original:
    """Symbolic interpreter for the unmodified formulas of a modelx space (its `__init__.py`).

    The formulas are not executed by Python. Each is parsed and evaluated node by node, and only the
    constructs listed here are accepted; anything else raises `Unsupported`, and the cell must then be
    replaced by an axiom.
      statements   assignment to a name; if / elif / else on a concrete condition; return; a docstring;
                   `x.name = "label"` (naming a pandas Series), which is ignored.
                   (`raise` is refused if reached.)
      expressions  numbers (a real literal is the decimal written), strings, None, True/False; names of cells
                   and local names; + - * / ** and // (integer by positive integer constant); & |; unary minus;
                   comparisons, `is None`, `in` a concrete list; `x if c else y` on a concrete condition;
                   calls to cells, to sum/range/len/max/min/abs on concrete arguments, to np.maximum, np.minimum,
                   np.exp, np.log, and to Series.where(cond, other); subscripts of concrete lists and dictionaries;
                   list and tuple displays; a list comprehension over a concrete sequence; a generator expression only
                   as the direct argument of sum().
      definitions  plain positional parameters; defaults only if literal; no decorators; no name defined twice.
      attributes   only `np.maximum/minimum/exp/log`, `.where` on a value, and attributes of objects the symbolic
                   description supplies; never an attribute of a function, and never a name beginning with an underscore.
    Arithmetic on two concrete numbers is exact (rational), never floating point.
    Objects placed in `ns` by the model's symbolic description, and axioms, are trusted code: attribute access,
    subscripts and calls on them are performed natively."""
    def __init__(self, path, int_cells=()):
        self.memo, self.prev, self.int_cells = {}, {}, set(int_cells)
        self.src = open(path).read(); tree = ast.parse(self.src)
        class NP:
            maximum = staticmethod(smax); minimum = staticmethod(smin)
            exp = staticmethod(lambda x: Sym(EXP(real(lift(x))))); log = staticmethod(lambda x: Sym(LOG(real(lift(x)))))
        self._np = NP; self.ns = {"np": NP}
        self.defs = {}
        for n in tree.body:
            if type(n) is not ast.FunctionDef: continue
            if n.name in self.defs: raise Unsupported(f"`{n.name}` is defined twice")
            self.defs[n.name] = n
        self.raw = {}
        self.n_formulas, self.axiomatised = len(self.defs), []
        for name in self.defs: self._bind(name)
    def _bind(self, name):
        self.ns[name] = (lambda n: lambda *a: self(n, *a))(name)
    def axiom(self, name, fn):
        """Replace a cell by an assumed definition (inputs, table lookups, anything not executable symbolically)."""
        self.raw[name] = fn; self.axiomatised.append(name); self._bind(name)
    def __call__(self, name, *args):
        key = (name,) + tuple(("T", a.k) if isinstance(a, Time) else a for a in args)
        if key not in self.memo:
            if args and isinstance(args[0], Time) and args[0].k < 0:     # induction hypothesis: left opaque
                sort = INT if name in self.int_cells else REAL
                self.prev[key] = self.memo[key] = Sym(z3.Const("prev:" + repr(key), sort))
            elif name in self.raw: self.memo[key] = _norm(self.raw[name](*args))
            elif name in self.defs: self.memo[key] = self._apply(self.defs[name], list(args), {})
            else: raise Unsupported(f"`{name}` is not a cell of this space")
        return self.memo[key]

    # ---- evaluation
    def _apply(self, f, args, kwargs):
        a = f.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs: raise Unsupported(f"{f.name}: unsupported parameter form")
        if f.decorator_list: raise Unsupported(f"{f.name}: decorators are not modelled")
        for d in a.defaults:                        # only literal defaults: their value cannot depend on when they are bound
            if not (type(d) is ast.Constant and (d.value is None or isinstance(d.value, (bool, int, float, str)))):
                raise Unsupported(f"{f.name}: a default argument that is not a literal is not modelled")
        names = [x.arg for x in a.args]; env = {}
        defaults = dict(zip(names[len(names) - len(a.defaults):], a.defaults))
        if len(args) > len(names): raise Unsupported(f"{f.name}: too many arguments")
        for i, n in enumerate(names):
            if i < len(args): env[n] = args[i]
            elif n in kwargs: env[n] = kwargs[n]
            elif n in defaults: env[n] = self._eval(defaults[n], {})
            else: raise Unsupported(f"{f.name}: missing argument `{n}`")
        r = self._exec(f.body, env, f.name)
        return None if r is _NORETURN else r
    def _cond(self, c, where):
        if not isinstance(c, bool): raise Unsupported(f"{where}: a branch depends on a symbolic value; axiomatise that cell")
        return c
    def _exec(self, stmts, env, where):
        for st in stmts:
            if type(st) is ast.Expr and type(st.value) is ast.Constant and isinstance(st.value.value, str): continue
            if type(st) is ast.Pass: continue
            if (type(st) is ast.Assign and len(st.targets) == 1 and type(st.targets[0]) is ast.Attribute and st.targets[0].attr == "name"
                    and type(st.targets[0].value) is ast.Name and type(st.value) is ast.Constant and isinstance(st.value.value, str)):
                continue                                         # `result.name = "..."`: labels a pandas Series; no effect on values
            if type(st) is ast.Assign and len(st.targets) == 1 and type(st.targets[0]) is ast.Name:
                env[st.targets[0].id] = self._eval(st.value, env)
            elif type(st) is ast.If:
                r = self._exec(st.body if self._cond(self._eval(st.test, env), where) else st.orelse, env, where)
                if r is not _NORETURN: return r
            elif type(st) is ast.Return: return None if st.value is None else self._eval(st.value, env)
            elif type(st) is ast.Raise: raise Unsupported(f"{where}: the original raises an error on this path")
            else: raise Unsupported(f"{where}: statement {type(st).__name__} is not modelled")
        return _NORETURN
    def _binop(self, op, a, b):
        try:
            if op is ast.Pow:
                if _exact(a) and isinstance(b, int) and not isinstance(b, bool) and abs(b) <= 64 and (a != 0 or b >= 0): return Fraction(a) ** b
                return Sym(POW(real(lift(a)), real(lift(b))))
            if op is ast.Div:
                if _exact(a) and _exact(b):
                    if b == 0: raise Unsupported("division by zero")
                    return Fraction(a) / Fraction(b)
                return a / b
            if op is ast.FloorDiv:
                if _exact(a) and _exact(b):
                    if not (isinstance(a, int) and isinstance(b, int) and b > 0): raise Unsupported("floor division is modelled only for integers and a positive divisor")
                    return a // b
                return a // b
            if op in _OPS: return _norm(_OPS[op](a, b))
        except TypeError as e: raise Unsupported(f"operation not modelled: {e}")
        raise Unsupported(f"operator {op.__name__} is not modelled")
    def _attr(self, obj, name):
        if name.startswith("_"): raise Unsupported(f"attribute `.{name}` is not modelled")
        if obj is self._np:
            if name in ("maximum", "minimum", "exp", "log"): return getattr(obj, name)
            raise Unsupported(f"`np.{name}` is not modelled")
        if callable(obj) and not hasattr(type(obj), "__getattr__"):
            raise Unsupported(f"attribute `.{name}` of a function is not modelled")
        if isinstance(obj, Sym):
            if name == "where" or any(name in k.__dict__ for k in type(obj).__mro__ if k not in (Sym, object)): return getattr(obj, name)
            raise Unsupported(f"attribute `.{name}` of a value is not modelled")
        if isinstance(obj, _PLAIN + (Time, Enum, float)): raise Unsupported(f"attribute `.{name}` of a {type(obj).__name__} is not modelled")
        try: return getattr(obj, name)               # an object supplied by the symbolic description: trusted
        except AttributeError: raise Unsupported(f"`{name}` is not provided")
    def _eval(self, n, env):
        t = type(n)
        if t is not ast.GeneratorExp and "__consumed_once__" in env:
            env = {k: v for k, v in env.items() if k != "__consumed_once__"}
        if t is ast.Constant:
            v = n.value
            if isinstance(v, float): return Fraction(ast.get_source_segment(self.src, n) or repr(v))
            if v is None or isinstance(v, (bool, int, str)): return v
            raise Unsupported(f"constant {v!r}")
        if t is ast.Name:
            if n.id in env: return env[n.id]
            if n.id in self.ns: return self.ns[n.id]
            raise Unsupported(f"name `{n.id}` is not modelled")
        if t is ast.BinOp: return self._binop(type(n.op), self._eval(n.left, env), self._eval(n.right, env))
        if t is ast.UnaryOp:
            v = self._eval(n.operand, env)
            if type(n.op) is ast.USub and (_exact(v) or isinstance(v, Sym)): return -v
            if type(n.op) is ast.UAdd and (_exact(v) or isinstance(v, Sym)): return v
            if type(n.op) is ast.Not and isinstance(v, bool): return not v
            if type(n.op) is ast.Invert and isinstance(v, Sym): return ~v
            raise Unsupported(f"unary {type(n.op).__name__} on {type(v).__name__} is not modelled")
        if t is ast.BoolOp:
            vals = [self._cond(self._eval(x, env), "and/or") for x in n.values]
            return all(vals) if type(n.op) is ast.And else any(vals)
        if t is ast.Compare:
            left, out = self._eval(n.left, env), []
            for op, rn in zip(n.ops, n.comparators):
                right = self._eval(rn, env)
                if type(op) in (ast.Is, ast.IsNot):
                    if right is not None: raise Unsupported("`is` is modelled only against None")
                    out.append((left is None) == (type(op) is ast.Is))
                elif type(op) in (ast.In, ast.NotIn):
                    if not isinstance(right, (list, tuple, dict)) or not isinstance(left, (str, int)) or isinstance(left, bool): raise Unsupported("`in` needs a concrete value and container")
                    out.append((left in right) == (type(op) is ast.In))
                elif type(op) in _CMPS:
                    if isinstance(left, float) or isinstance(right, float): raise Unsupported("float in a comparison")
                    try: out.append(_CMPS[type(op)](left, right))
                    except TypeError as e: raise Unsupported(f"comparison not modelled: {e}")
                else: raise Unsupported(type(op).__name__)
                left = right
            if len(out) == 1: return out[0]
            return all(self._cond(x, "chained comparison") for x in out)
        if t is ast.IfExp:
            return self._eval(n.body if self._cond(self._eval(n.test, env), "conditional expression") else n.orelse, env)
        if t in (ast.List, ast.Tuple):
            vals = [self._eval(x, env) for x in n.elts]; return vals if t is ast.List else tuple(vals)
        if t is ast.GeneratorExp:
            if not env.get("__consumed_once__"):
                raise Unsupported("a generator expression is modelled only as the direct argument of sum()")
            env = {k: v for k, v in env.items() if k != "__consumed_once__"}   # the permission is for this node only
        if t in (ast.ListComp, ast.GeneratorExp):
            if len(n.generators) != 1 or type(n.generators[0].target) is not ast.Name or n.generators[0].is_async: raise Unsupported("comprehension form")
            g = n.generators[0]; seq = self._eval(g.iter, env)
            if not isinstance(seq, (list, tuple, range)): raise Unsupported("comprehension over a non-concrete sequence")
            out = []
            for x in seq:
                e2 = dict(env); e2[g.target.id] = x
                if all(self._cond(self._eval(c, e2), "comprehension filter") for c in g.ifs): out.append(self._eval(n.elt, e2))
            return out
        if t is ast.Subscript:
            obj = self._eval(n.value, env)
            if type(n.slice) is ast.Slice: raise Unsupported("slices are not modelled")
            key = self._eval(n.slice, env)
            if isinstance(obj, (list, tuple, dict)):
                if not isinstance(key, (str, int)) or isinstance(key, bool): raise Unsupported("subscript with a non-concrete key")
                try: return obj[key]
                except (KeyError, IndexError): raise Unsupported(f"subscript {key!r} does not exist")
            if isinstance(obj, (Sym, Time, Enum) + _PLAIN): raise Unsupported(f"subscript of a {type(obj).__name__} is not modelled")
            return _norm(obj[key])                   # an object supplied by the symbolic description: trusted
        if t is ast.Attribute: return self._attr(self._eval(n.value, env), n.attr)
        if t is ast.Call:
            if any(type(a) is ast.Starred for a in n.args) or any(k.arg is None for k in n.keywords): raise Unsupported("starred arguments")
            direct_sum = type(n.func) is ast.Name and n.func.id == "sum" and n.func.id not in env and n.func.id not in self.ns
            args = [self._eval(a, dict(env, __consumed_once__=True) if direct_sum and type(a) is ast.GeneratorExp else env) for a in n.args]
            kw = {k.arg: self._eval(k.value, env) for k in n.keywords}
            if type(n.func) is ast.Name:
                name = n.func.id
                if name in env: raise Unsupported(f"call of the local name `{name}`")
                if name in self.ns:
                    if kw and name in self.defs and name not in self.raw and self.ns[name].__closure__ is not None:
                        params = [x.arg for x in self.defs[name].args.args]      # keywords to positions, so the cell is memoised by position
                        extra = params[len(args):]
                        if all(k in extra for k in kw) and all(p in kw for p in extra[:len(kw)]): args, kw = args + [kw[p] for p in extra[:len(kw)]], {}
                    return _norm(self.ns[name](*args, **kw))
                if name == "sum" and len(args) in (1, 2) and not kw and isinstance(args[0], (list, tuple)):
                    tot = args[1] if len(args) == 2 else 0
                    for x in args[0]: tot = self._binop(ast.Add, tot, x)
                    return tot
                if name == "range" and not kw and all(isinstance(a, int) and not isinstance(a, bool) for a in args): return range(*args)
                if name == "len" and not kw and len(args) == 1 and isinstance(args[0], (list, tuple, dict, range)): return len(args[0])
                if name in ("max", "min", "abs") and not kw and args and all(_exact(a) for a in args): return {"max": max, "min": min, "abs": abs}[name](*args)
                raise Unsupported(f"call to `{name}` is not modelled")
            if type(n.func) is ast.Attribute:
                f = self._attr(self._eval(n.func.value, env), n.func.attr)
                return _norm(f(*args, **kw))
            raise Unsupported("call form")
        raise Unsupported(f"expression {t.__name__} is not modelled")

@dataclass
class SymbolicSpec:
    source_path: str                 # the original space's formulas
    setup: Callable                  # (Original) -> mp: installs axioms, returns the symbolic model point
    kernel_args: Callable            # (O, mp, t) -> {name: value} for row_args, scen_series and time_series
    carry: Callable                  # (O, mp, t) -> [value per state variable]: what the kernel should carry out of month t
    invariants: Callable = None      # (O, mp, t) -> [(label, lhs, rhs)]: facts about the original alone, assumed at t-1, proved at t
    tables: dict = field(default_factory=dict)    # const name -> (lambda *indices: symbolic value) for lk1/lk2
    consts: dict = field(default_factory=dict)    # const name -> symbolic value, for consts that are not tables
    int_cells: tuple = ()            # original cells whose values are integers
    output_cell: Callable = None     # output name -> (cell, *args); default parses "cell[ARG]"
    lemmas: Callable = None          # (symbolic step function, mp) -> [(label, assumptions, lhs, rhs)]: extra obligations
    table_keys: dict = field(default_factory=dict)   # const name -> {index position: kernel argument that index must equal,
                                     # or None to declare the table constant along that index}. Required for every index
                                     # that a table's symbolic definition does not use: without it, a lookup in the wrong
                                     # row would be indistinguishable from the right one.
    assumptions: Callable = None     # (O, mp) -> [(label, fact)]: stated facts about inputs or tables (for example
                                     # "this table has no zero entry"); assumed in every obligation and reported

def sym_helpers(sspec, ka=None):
    def lk(tab, *idx):
        fn = sspec.tables[tab]; val = lift(fn(*idx)); keys = sspec.table_keys.get(tab, {}); guard = []
        for pos, i in enumerate(idx):
            key = keys.get(pos, _MISSING)
            if key is not _MISSING and key is not None:         # a declared key is enforced, whatever the definition looks like
                if ka is None: raise SpecError(f"table {tab} is keyed by a kernel argument, which this context does not supply")
                guard.append(lift(i) == lift(ka[key])); continue
            if key is None: continue                            # declared constant along this index: a stated assumption
            # undeclared: the definition must genuinely depend on this index (compared after simplification)
            probe = list(idx); probe[pos] = Sym(z3.FreshInt("probe")) if isinstance(i, (Sym, Time)) else i + 1
            try: ignored = z3.simplify(lift(fn(*probe))).eq(z3.simplify(val))
            except Exception: ignored = False
            if ignored:
                raise SpecError(f"the symbolic definition of table {tab} does not depend on index {pos}; declare in "
                                f"`table_keys` which kernel argument that index must equal, or None if the table is constant along it")
        g = z3.simplify(z3.And(*guard)) if guard else z3.BoolVal(True)
        if z3.is_true(g): return Sym(val)
        other = z3.Function(f"{tab}_other_row", *[INT] * len(idx), val.sort())     # any other row: an unknown value
        return Sym(z3.If(g, val, other(*[lift(i) for i in idx])))
    return dict(sel=lambda c, a, b: Sym(ite(lift(c), lift(a), lift(b))), fmax=smax, fmin=smin, imax=smax, imin=smin,
                band=lambda a, b: Sym(z3.And(lift(a), lift(b))),
                idiv=lambda a, k: Sym(_floordiv(lift(a), k)) if isinstance(a, (Sym, Time)) else a // k,
                tof=lambda a: Sym(real(lift(a))), fpow=lambda x, y: Sym(POW(real(lift(x)), real(lift(y)))),
                fexp=lambda x: Sym(EXP(real(lift(x)))), flog=lambda x: Sym(LOG(real(lift(x)))), C=lambda x: x, ONE=1, ZERO=0.0, lk1=lk, lk2=lk)

class KernelDomain:
    """Symbolic evaluation of the kernel language. Concrete numbers are exact (integers and rationals);
    arithmetic on two of them is exact rational arithmetic, never floating point."""
    def __init__(s, sspec, ka): s.h = sym_helpers(sspec, ka)
    def integer(s, n): return int(n)
    def real(s, text): return Fraction(text)
    def neg(s, a): return -a
    def binop(s, op, a, b):
        if isinstance(a, bool) or isinstance(b, bool): raise lang.InvalidStep("arithmetic on a truth value")
        if _exact(a) and _exact(b):
            if op == "/":
                if b == 0: raise SpecError("the step divides by a literal zero")
                return Fraction(a) / Fraction(b)
            return a + b if op == "+" else a - b if op == "-" else a * b
        return a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b
    def compare(s, op, a, b):
        return {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}[op](a, b)
    def band(s, a, b): return (a and b) if isinstance(a, bool) and isinstance(b, bool) else s.h["band"](a, b)
    def call(s, name, args):
        if name == "sel" and isinstance(args[0], bool): return args[1] if args[0] else args[2]
        if name in ("fmax", "imax", "fmin", "imin") and _exact(args[0]) and _exact(args[1]): return (max if "max" in name else min)(args)
        if name == "tof" and _exact(args[0]): return Fraction(args[0])
        if name == "band": return s.band(*args)
        if name == "C": return args[0]
        return s.h[name](*args)

def symbolic_step(kspec, sspec, src=None, ka=None):
    """The step as a function on symbolic values: the validated tree evaluated node by node."""
    step = lang.parse_step(src or kspec.step_src); dom = KernelDomain(sspec, ka)
    return lambda *args: lang.run(step, [_norm(a) for a in args], dom)

def obligations(kspec, sspec, base, src=None):
    """(assumptions, [(label, kernel value, original value)], Original) for month 0 (`base`) or month t >= 1."""
    O = Original(sspec.source_path, sspec.int_cells); mp = sspec.setup(O)
    t = 0 if base else Time(0)
    cell = sspec.output_cell or parse_cell
    ocells = [O(*(lambda c: (c[0], t) + tuple(c[1:]))(cell(n))) for n in kspec.outputs]
    o_carry = sspec.carry(O, mp, t)
    assume, inv_now = [], (sspec.invariants(O, mp, t) if sspec.invariants else [])
    if base:
        state = [Sym(z3.Real(f"any_{s}")) for s in kspec.state]
    else:
        state = sspec.carry(O, mp, Time(-1))
        assume.append(TVAR >= 1)
        if sspec.invariants:
            assume += [lift(l) == lift(r) if z3.is_int(lift(l)) == z3.is_int(lift(r)) else real(lift(l)) == real(lift(r))
                       for _, l, r in sspec.invariants(O, mp, Time(-1))]
    if sspec.assumptions: assume += [lift(f) for _, f in sspec.assumptions(O, mp)]
    ka = sspec.kernel_args(O, mp, t)
    step = symbolic_step(kspec, sspec, src, ka)
    consts = [sspec.consts.get(c, c) for c in kspec.consts]
    r = step(0 if base else Sym(TVAR), base, *[ka[a] for a in kspec.row_args + kspec.scen_series + kspec.time_series],
             *state, *consts)
    obs = [(n, lift(r[k]), lift(ocells[k])) for k, n in enumerate(kspec.outputs)]
    obs += [(f"carried state: {s}", lift(r[kspec.n_out + k]), lift(o_carry[k])) for k, s in enumerate(kspec.state)]
    obs += [(f"invariant: {lab}", lift(l), lift(rr)) for lab, l, rr in inv_now]
    return assume, obs, O

def check(assume, lhs, rhs, timeout_ms=20000):
    """'proved', 'refuted' (with a counterexample model) or 'unknown'."""
    if z3.is_int(lhs) != z3.is_int(rhs): lhs, rhs = real(lhs), real(rhs)
    if z3.simplify(lhs).eq(z3.simplify(rhs)): return "proved", None
    if z3.is_real(lhs) and z3.is_true(z3.simplify(z3.simplify(lhs - rhs, som=True, hoist_ite=True, pull_cheap_ite=True) == 0)):
        return "proved", None                    # polynomial identity after normalisation
    s = z3.Solver(); s.set("timeout", timeout_ms); s.add(*assume); s.add(lhs != rhs)
    r = s.check()
    return ("proved", None) if r == z3.unsat else ("refuted", s.model()) if r == z3.sat else ("unknown", None)

def prove(kspec, sspec, src=None, stop_at_first=False, timeout_ms=20000, gate=True):
    """Prove the month-0 and month-t obligations. Returns a dict with counts, the labels refuted
    (with counterexample models) and the labels left unknown."""
    out = {"proved": 0, "refuted": [], "unknown": [], "total": 0, "seconds": 0.0}
    t0 = time.time()
    for base in (True, False):
        assume, obs, O = obligations(kspec, sspec, base, src)
        out["formulas"], out["axiomatised"] = O.n_formulas, list(O.axiomatised)      # formulas in the file; cells replaced by axioms
        out["cells_reached"] = sorted(set(out.get("cells_reached", [])) | {k[0] for k in O.memo})   # cells the obligations actually evaluate
        out["assumptions"] = [lab for lab, _ in sspec.assumptions(O, sspec.setup(Original(sspec.source_path, sspec.int_cells)))] if sspec.assumptions else []
        out["assumptions"] += [f"table {t} is the same for every value of index {p}" for t, ks in sspec.table_keys.items() for p, k in ks.items() if k is None]
        out["table_keys"] = {t: {p: k for p, k in ks.items() if k is not None} for t, ks in sspec.table_keys.items()}
        pending, lemmas = [], []
        for label, lhs, rhs in obs:
            out["total"] += 1
            verdict, model = check(assume, lhs, rhs, timeout_ms)
            tag = ("month 0: " if base else "month t>=1: ") + label
            if verdict == "proved":
                out["proved"] += 1; lemmas.append(real(lhs) == real(rhs))
            elif verdict == "refuted":
                out["refuted"].append((tag, model))
                if stop_at_first:
                    out["seconds"] = time.time() - t0
                    if gate:
                        from . import gates
                        gates.record("proof (stopped at the first refutation)", False, tag)
                    return out
            else: pending.append((tag, lhs, rhs))
        for tag, lhs, rhs in pending:            # retry with the equalities already proved as cut points
            verdict, model = check(assume + lemmas, lhs, rhs, timeout_ms)
            if verdict == "proved": out["proved"] += 1
            elif verdict == "refuted": out["refuted"].append((tag, model))
            else: out["unknown"].append(tag)
    out["seconds"] = time.time() - t0
    if gate:
        from . import gates
        gates.record(f"proof of {out['total']} obligations", out["proved"] == out["total"],
                     f"{out['proved']} proved, {len(out['refuted'])} refuted, {len(out['unknown'])} unknown")
    return out

def exit_obligations(kspec, sspec, src=None):
    """Obligations that justify early exit, generated from the exit rule the backends execute (`kspec.exit_when`).
    For period 0 and for an arbitrary period t >= 1, and for ANY incoming state: if the rule holds after
    the period, then in the next period every accumulated output is zero and the rule holds again.
    By induction nothing is lost by stopping. Returns [(label, assumptions, lhs, rhs)]."""
    if not kspec.exit_when: return []
    out, names = [], kspec.row_args + kspec.scen_series + kspec.time_series
    for base in (True, False):
        O = Original(sspec.source_path, sspec.int_cells); mp = sspec.setup(O)
        t0, t1 = (0, 1) if base else (Time(0), Time(1))
        assume = [] if base else [TVAR >= 1]
        if sspec.assumptions: assume += [lift(f) for _, f in sspec.assumptions(O, mp)]
        ka0, ka1 = sspec.kernel_args(O, mp, t0), sspec.kernel_args(O, mp, t1)
        step = symbolic_step(kspec, sspec, src, ka0); tv = lambda t: t if isinstance(t, int) else Sym(t.expr())
        consts = [sspec.consts.get(c, c) for c in kspec.consts]
        state = [Sym(z3.Real(f"any_{s}")) for s in kspec.state]
        r0 = step(tv(t0), base, *[ka0[a] for a in names], *state, *consts)
        r1 = step(tv(t1), False, *[ka1[a] for a in names], *r0[kspec.n_out:], *consts)
        tree = lang.parse_exit(kspec.exit_when, kspec.outputs, kspec.row_args); dom = KernelDomain(sspec, ka0)
        def rule(r, t):                                     # the exit rule, parsed against the kernel grammar: never run as Python
            v = lang.evaluate(tree, {"OUT": {n: _norm(r[kspec.ix(n)]) for n in kspec.outputs},
                                     "ROW": {a: ka0[a] for a in kspec.row_args}, "t": tv(t)}, dom)
            return lift(v)
        when = "period 0" if base else "period t>=1"
        hyp = assume + [rule(r0, t0)]
        for n in kspec.accumulate:
            out.append((f"exit after {when}: {n} is zero in the next period", hyp, lift(r1[kspec.ix(n)]), z3.RealVal(0)))
        out.append((f"exit after {when}: the exit rule holds again in the next period", hyp, rule(r1, t1), z3.BoolVal(True)))
    return out

def prove_lemmas(kspec, sspec, timeout_ms=20000, src=None, gate=True):
    """Extra obligations: the early-exit obligations generated from the executed exit rule, followed by any
    model-specific lemmas. Returns [(label, verdict)]."""
    obs = exit_obligations(kspec, sspec, src)
    res = [(label, check(assume, lhs, rhs, timeout_ms)[0]) for label, assume, lhs, rhs in obs]
    for k, (label, assume, lhs, rhs) in enumerate(obs):          # retry, using what is already proved under the same hypotheses
        if res[k][1] == "unknown":
            proved = [(l, r) for (_, a, l, r), (_, v) in zip(obs, res) if a is assume and v == "proved"]
            known = [real(l) == real(r) if z3.is_arith(l) else l == r for l, r in proved]
            same_sort = [(l, r) for l, r in proved if l.sort() == r.sort()]
            res[k] = (label, check(assume + known, z3.substitute(lhs, *same_sort) if same_sort else lhs, rhs, timeout_ms)[0])
    if sspec.lemmas:
        O = Original(sspec.source_path, sspec.int_cells); mp = sspec.setup(O)
        ka = sspec.kernel_args(O, mp, Time(0))
        res += [(label, check(assume, lhs, rhs, timeout_ms)[0]) for label, assume, lhs, rhs in sspec.lemmas(symbolic_step(kspec, sspec, src, ka), mp)]
    if gate and res:
        from . import gates
        gates.record(f"early-exit obligations and lemmas ({len(res)})", all(v == "proved" for _, v in res),
                     ", ".join(f"{l}: {v}" for l, v in res if v != "proved")[:300])
    return res

def refute_mutants(kspec, sspec, mutant_list, timeout_ms=5000):
    """For each seeded defect: is it refuted (a counterexample exists), proved equivalent, unknown, or
    invalid (not a runnable program, or outside what the interpreter models)? Invalid ones are not
    evidence either way and are counted separately."""
    rows, tally = [], {"refuted": 0, "proved": 0, "unknown": 0, "invalid": 0}
    for kind, desc, src in mutant_list:
        try:
            o = prove(kspec, sspec, src, stop_at_first=True, timeout_ms=timeout_ms, gate=False)
            v = "refuted" if o["refuted"] else "unknown" if o["unknown"] else "proved"
            cex = (o["refuted"][0][0], show_model(o["refuted"][0][1])) if o["refuted"] else None
        except Exception as e:
            v, cex = "invalid", ("not a valid program", repr(e)[:80])
        tally[v] += 1; rows.append((kind, desc, v, cex))
    return tally, rows

def show_model(model):
    vals = {d.name(): model[d] for d in model.decls() if d.arity() == 0 and not d.name().startswith(("prev:", "any_"))}
    return ", ".join(f"{k}={v}" for k, v in sorted(vals.items()))
