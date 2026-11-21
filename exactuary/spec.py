"""Description of a rewritten model ("kernel") that every backend and every check reads.

The kernel is one straight-line Python function, held as source text:

    def step(t, is_t0, <row_args>, <scen_series>, <time_series>, <state>, <consts>):
        ...
        return (<outputs>, <next state>)

It may use only arithmetic, comparisons and the helper operations listed in `HELPERS`.
It must contain no Python `if`, loop, or other data-dependent control flow: choices are
written with `sel(condition, a, b)`. This restriction is what allows the same source to be
compiled for a CPU, compiled for a GPU, executed symbolically and analysed for rounding error.

"Period" below means one projection step: a month in a monthly model, a year in an annual one.

Inputs are passed to every backend as one dictionary `p` holding:
    p[name]   for each name in row_args     array [row]
    p[name]   for each name in scen_series  array [scenario, period]
    p[name]   for each name in time_series  array [period]
    p[name]   for each name in consts       table (array) or scalar
    p["scen"]                               int array [row]: the scenario each row uses (all zeros if
                                            there are no scenario series; it is required regardless)
    p["T"]                                  number of periods to project
    p[discount]                             array [period]: discount factors (default name "DISC")

Two conventions that are easy to miss:
  * A truth value multiplied by a number in the original is written `sel(c, ONE, ZERO) * x`, not
    `sel(c, x, ZERO)`: the two differ when x is NaN or infinite, and the original produces the former.
  * For float32 every real literal is wrapped automatically, and a literal `1` on the left of an
    operator becomes the real `ONE`. Write integer expressions as `t + 1`, not `1 + t`.
"""
from dataclasses import dataclass, field
import re

HELPERS = {
    "sel(c, a, b)": "a if c else b",
    "fmax(a, b), fmin(a, b)": "larger / smaller of two real values",
    "imax(a, b), imin(a, b)": "larger / smaller of two integers",
    "band(a, b)": "logical and",
    "idiv(a, k)": "integer floor division by a positive constant",
    "tof(a)": "integer to real",
    "lk1(table, i), lk2(table, i, j)": "table lookup; any index outside the table gives zero",
    "fpow(x, y), fexp(x), flog(x)": "power, exponential, logarithm, for use only where the argument depends on "
                                    "carried state and so cannot be precomputed as a table. Different platforms' "
                                    "mathematical libraries may differ in the last bit, so outputs downstream of these "
                                    "may not be bit-identical; measure and report the difference",
    "C(x), ONE, ZERO": "real constants in working precision (inserted automatically for float32)",
}

@dataclass
class KernelSpec:
    step_src: str
    row_args: list            # per-policy inputs (one array each, length = number of rows)
    scen_series: list         # per-month inputs that depend on the scenario: array[scenario, t]
    time_series: list         # per-month inputs common to all rows: array[t]
    state: list               # values carried from one month to the next
    consts: list              # tables and scalars passed through unchanged
    outputs: list             # names of the step's outputs, in return order
    accumulate: list          # outputs whose discounted sum over months is reported
    discount: str = "DISC"    # name of the discount-factor array in the inputs
    exit_when: str = None     # optional early-exit test, e.g. "(OUT['pols'] == ZERO) & (ROW['dm0'] + t >= 0)"

    @property
    def n_out(self): return len(self.outputs)
    def ix(self, name): return self.outputs.index(name)
    def with_src(self, src):
        from dataclasses import replace
        return replace(self, step_src=src)
    def exit_expr(self, out=lambda k: f"r[{k}]", row=lambda n: f"R_{n}[i]"):
        """The exit test as backend source, regenerated from the parsed rule with OUT['name'] and ROW['name']
        replaced by the backend's own references."""
        import ast
        from .lang import parse_exit
        tree = parse_exit(self.exit_when, self.outputs, self.row_args)
        class Tr(ast.NodeTransformer):
            def visit_Subscript(s, n):
                ref = out(self.ix(n.slice.value)) if n.value.id == "OUT" else row(n.slice.value)
                return ast.parse(ref, mode="eval").body
        return ast.unparse(ast.fix_missing_locations(Tr().visit(tree)))

from .lang import InvalidStep, parse_step, parse_exit, clean_source

def validate_step(src, n_returns=None):
    """Check that the source is a step in the kernel language (see `exactuary/lang.py`). Returns its parameter names."""
    return parse_step(src, n_returns).params

def check_spec(spec):
    """Validate a KernelSpec as a whole: the step's language, its parameter list, its return arity, its
    names, and its exit rule (which is parsed against the same grammar and never evaluated as Python)."""
    import re
    names = spec.row_args + spec.scen_series + spec.time_series + spec.state + spec.consts
    for n in names + [spec.discount]:
        if not isinstance(n, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n): raise InvalidStep(f"`{n}` is not a valid name")
    for n in spec.outputs:
        if not isinstance(n, str) or "'" in n or "\\" in n or "\n" in n: raise InvalidStep(f"output name {n!r} is not acceptable")
    params = parse_step(spec.step_src, spec.n_out + len(spec.state)).params
    want = ["t", "is_t0"] + names
    if params != want: raise InvalidStep(f"`step` parameters {params} do not match the specification {want}")
    if len(set(spec.outputs)) != len(spec.outputs): raise InvalidStep("duplicate output name")
    for n in spec.accumulate:
        if n not in spec.outputs: raise InvalidStep(f"accumulated output `{n}` is not an output")
    if spec.exit_when is not None: parse_exit(spec.exit_when, spec.outputs, spec.row_args)

def make_step(helpers, src):
    """Build an executable step for a numeric backend. The source is parsed and validated; what is
    compiled is text regenerated from the validated tree, under globals holding the helper operations
    and nothing else (no builtins)."""
    code = clean_source(parse_step(src))
    ns = {"__builtins__": {}}; ns.update({k: v for k, v in helpers.items() if not k.startswith("__")})
    exec(compile(code, "<step>", "exec"), ns); return ns["step"]

def to_f32_src(src):
    """The same step with every real literal in single precision, so that the whole
    computation is carried out in float32 rather than silently promoted to float64."""
    import ast
    class Tr(ast.NodeTransformer):
        def visit_Constant(self, n):
            return ast.Call(ast.Name("C", ast.Load()), [n], []) if isinstance(n.value, float) else n
        def visit_BinOp(self, n):
            if isinstance(n.op, ast.Div) and isinstance(n.left, ast.Constant) and isinstance(n.right, ast.Constant):
                return ast.Call(ast.Name("C", ast.Load()), [ast.Constant(n.left.value / n.right.value)], [])
            if isinstance(n.left, ast.Constant) and isinstance(n.left.value, int) and n.left.value == 1:
                n.left = ast.Name("ONE", ast.Load())
            self.generic_visit(n); return n
    t = Tr().visit(ast.parse(src)); ast.fix_missing_locations(t)
    return ast.unparse(t)
