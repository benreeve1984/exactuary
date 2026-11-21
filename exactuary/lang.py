"""The kernel language: parsing, validation and evaluation.

A step is never executed as the Python text its author wrote. It is parsed, checked against the
grammar below, and then either
  * evaluated node by node by `run`, with every operation supplied by a "domain" (interval arithmetic,
    symbolic terms), so that nothing is computed by Python before the domain sees it; or
  * compiled, for the numeric backends, from source text regenerated from the checked tree, under
    globals that contain the helper operations and nothing else.

Grammar
    step      :=  def step(<names>):  <assignment>*  return (<expr>, ...)
    assignment:=  <name> = <expr>
    expr      :=  <number> | <name> | ONE | ZERO | -<expr> | <expr> (+|-|*|/) <expr>
               |  <expr> (==|!=|<|<=|>|>=) <expr> | <helper>(<expr>, ...)
    exit rule :=  <cmp> | (<cmp>) & (<cmp>) ...        with OUT['output'], ROW['row input'] and t as names

No annotations, defaults, decorators, attribute access, subscripts (other than OUT/ROW in an exit rule),
keyword arguments, strings, or calls to anything but the helpers.
"""
import ast
from dataclasses import dataclass

HELPERS = ("sel", "fmax", "fmin", "imax", "imin", "band", "idiv", "tof", "lk1", "lk2", "fpow", "fexp", "flog", "C")
ARITY = {"sel": 3, "fmax": 2, "fmin": 2, "imax": 2, "imin": 2, "band": 2, "idiv": 2, "tof": 1, "lk1": 2, "lk2": 3,
         "fpow": 2, "fexp": 1, "flog": 1, "C": 1}
CONSTANTS = ("ONE", "ZERO")
_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}

class InvalidStep(ValueError):
    """The text is not in the kernel language."""

@dataclass
class Step:
    params: list        # parameter names, in order
    body: list          # [(name, expression tree)]
    returns: list       # expression trees

def _check_expr(n, known, src, exit_rule=False):
    """Validate an expression tree in place; real literals are tagged with the text that was written."""
    if isinstance(n, ast.Constant):
        if isinstance(n.value, bool) or not isinstance(n.value, (int, float)): raise InvalidStep(f"constant {n.value!r} is not a number")
        if isinstance(n.value, float): n.text = (ast.get_source_segment(src, n) if src else None) or repr(n.value)
    elif isinstance(n, ast.Name):
        if not isinstance(n.ctx, ast.Load) or (n.id not in known and n.id not in CONSTANTS):
            raise InvalidStep(f"`{n.id}` is used before it is defined")
    elif isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        _check_expr(n.left, known, src, exit_rule); _check_expr(n.right, known, src, exit_rule)
    elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd) and exit_rule:
        _check_expr(n.left, known, src, True); _check_expr(n.right, known, src, True)
    elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        _check_expr(n.operand, known, src, exit_rule)
    elif isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMP:
        _check_expr(n.left, known, src, exit_rule); _check_expr(n.comparators[0], known, src, exit_rule)
    elif isinstance(n, ast.Call) and not exit_rule:
        if not isinstance(n.func, ast.Name) or n.func.id not in HELPERS or n.keywords:
            raise InvalidStep(f"call to `{ast.unparse(n.func)}` is not allowed")
        if len(n.args) != ARITY[n.func.id]: raise InvalidStep(f"`{n.func.id}` takes {ARITY[n.func.id]} arguments")
        if any(isinstance(a, ast.Starred) for a in n.args): raise InvalidStep("starred arguments are not allowed")
        for a in n.args: _check_expr(a, known, src)
    elif (exit_rule and isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in ("OUT", "ROW")
          and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)):
        if n.slice.value not in known[n.value.id]: raise InvalidStep(f"{n.value.id}['{n.slice.value}'] does not exist")
    else:
        raise InvalidStep(f"{type(n).__name__} is not allowed: {ast.unparse(n)[:60]}")

def parse_step(src, n_returns=None):
    """Parse and validate a step. Returns a `Step`."""
    try: tree = ast.parse(src)
    except SyntaxError as e: raise InvalidStep(f"not valid syntax: {e}")
    if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef or tree.body[0].name != "step":
        raise InvalidStep("the source must consist of exactly one function named `step`")
    f = tree.body[0]; a = f.args
    if (f.decorator_list or f.returns is not None or getattr(f, "type_params", None) or a.vararg or a.kwarg or a.kwonlyargs
            or a.defaults or a.kw_defaults or a.posonlyargs or any(x.annotation is not None or x.type_comment for x in a.args)):
        raise InvalidStep("`step` must take plain positional parameters only: no annotations, defaults or decorators")
    params = [x.arg for x in a.args]
    if len(set(params)) != len(params): raise InvalidStep("duplicate parameter name")
    for p in params:
        if p in HELPERS or p in CONSTANTS or p.startswith("__"): raise InvalidStep(f"`{p}` may not be a parameter name")
    known, body = set(params), []
    if not f.body: raise InvalidStep("empty step")
    for st in f.body[:-1]:
        if not (type(st) is ast.Assign and len(st.targets) == 1 and type(st.targets[0]) is ast.Name and st.type_comment is None):
            raise InvalidStep(f"only assignments to a single name are allowed: {ast.unparse(st)[:60]}")
        name = st.targets[0].id
        if name in HELPERS or name in CONSTANTS or name.startswith("__"): raise InvalidStep(f"`{name}` may not be assigned")
        _check_expr(st.value, known, src); known.add(name); body.append((name, st.value))
    ret = f.body[-1]
    if type(ret) is not ast.Return or type(ret.value) is not ast.Tuple: raise InvalidStep("`step` must end with `return (..., ...)`")
    for x in ret.value.elts: _check_expr(x, known, src)
    if n_returns is not None and len(ret.value.elts) != n_returns:
        raise InvalidStep(f"`step` returns {len(ret.value.elts)} values; the specification requires {n_returns} (outputs + state)")
    return Step(params, body, list(ret.value.elts))

def parse_exit(text, outputs, row_args):
    """Parse and validate an exit rule. Returns its expression tree."""
    try: tree = ast.parse(text, mode="eval").body
    except SyntaxError as e: raise InvalidStep(f"exit rule is not valid syntax: {e}")
    _check_expr(tree, {"OUT": set(outputs), "ROW": set(row_args), "t": True}, text, exit_rule=True)
    return tree

def clean_source(step):
    """Python source regenerated from a validated step: what the numeric backends compile."""
    lines = [f"def step({', '.join(step.params)}):"]
    lines += [f"    {name} = {ast.unparse(e)}" for name, e in step.body]
    lines.append("    return (" + ", ".join(ast.unparse(e) for e in step.returns) + ("," if len(step.returns) == 1 else "") + ")")
    return "\n".join(lines) + "\n"

def evaluate(n, env, dom):
    """Evaluate one expression tree in a domain. The domain supplies every operation:
        integer(n), real(text), neg(a), binop(op, a, b), compare(op, a, b), call(helper, args), band(a, b)."""
    if isinstance(n, ast.Constant):
        return dom.real(n.text) if isinstance(n.value, float) else dom.integer(n.value)
    if isinstance(n, ast.Name):
        if n.id in env: return env[n.id]
        return dom.real("1.0" if n.id == "ONE" else "0.0")
    if isinstance(n, ast.BinOp):
        a, b = evaluate(n.left, env, dom), evaluate(n.right, env, dom)
        return dom.band(a, b) if isinstance(n.op, ast.BitAnd) else dom.binop(_BIN[type(n.op)], a, b)
    if isinstance(n, ast.UnaryOp): return dom.neg(evaluate(n.operand, env, dom))
    if isinstance(n, ast.Compare):
        return dom.compare(_CMP[type(n.ops[0])], evaluate(n.left, env, dom), evaluate(n.comparators[0], env, dom))
    if isinstance(n, ast.Call): return dom.call(n.func.id, [evaluate(a, env, dom) for a in n.args])
    if isinstance(n, ast.Subscript): return env[n.value.id][n.slice.value]
    raise InvalidStep(type(n).__name__)

def run(step, args, dom):
    """Evaluate a validated step on positional arguments in a domain. Returns the tuple of returned values."""
    if len(args) != len(step.params): raise TypeError(f"step takes {len(step.params)} arguments, {len(args)} given")
    env = dict(zip(step.params, args))
    for name, e in step.body: env[name] = evaluate(e, env, dom)
    return tuple(evaluate(e, env, dom) for e in step.returns)
