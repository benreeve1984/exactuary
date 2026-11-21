"""Seeded defects: every single-edit variant ("mutant") of a step function, used to measure
how many defects a given check would catch."""
import ast, copy

def mutants(step_src, swaps=None):
    """Returns [(kind, description, mutated source)].
    Kinds: operator (+ <-> -, * <-> /), constant (perturbed), comparison (== -> !=, < -> <=, > -> >=),
    branch-swap (sel arms exchanged), min/max (exchanged), lookup-index (a table index off by one),
    wrong-variable (per `swaps`: {name: confusable name}). Edits that leave the program text unchanged
    (for example perturbing a literal zero) are omitted."""
    swaps = swaps or {}
    base = ast.parse(step_src); out = []
    nodes = [(n, ast.unparse(st)[:70]) for st in base.body[0].body for n in ast.walk(st)]
    for i, (n, stmt) in enumerate(nodes):
        def emit(kind, fn):
            tree = copy.deepcopy(base)
            tgt = [x for st in tree.body[0].body for x in ast.walk(st)][i]
            before = ast.unparse(tgt); fn(tgt); ast.fix_missing_locations(tree)
            if ast.unparse(tgt) != before:
                out.append((kind, f"{before}  ->  {ast.unparse(tgt)}   in: {stmt}", ast.unparse(tree)))
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            new = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}[type(n.op)]
            emit("operator", lambda t, new=new: setattr(t, "op", new()))
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            emit("constant", lambda t: setattr(t, "value", t.value + 1 if isinstance(t.value, int) else t.value * 1.01))
        elif isinstance(n, ast.Compare) and type(n.ops[0]) in (ast.Eq, ast.Lt, ast.Gt):
            new = {ast.Eq: ast.NotEq, ast.Lt: ast.LtE, ast.Gt: ast.GtE}[type(n.ops[0])]
            emit("comparison", lambda t, new=new: setattr(t, "ops", [new()]))
        elif isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("fmax", "fmin", "imin", "imax", "sel"):
            if n.func.id == "sel":
                emit("branch-swap", lambda t: setattr(t, "args", [t.args[0], t.args[2], t.args[1]]))
            else:
                flip = {"fmax": "fmin", "fmin": "fmax", "imin": "imax", "imax": "imin"}
                emit("min/max", lambda t: setattr(t.func, "id", flip[t.func.id]))
        elif isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("lk1", "lk2"):
            for k in range(1, len(n.args)):
                emit("lookup-index", lambda t, k=k: t.args.__setitem__(k, ast.BinOp(t.args[k], ast.Add(), ast.Constant(1))))
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in swaps:
            emit("wrong-variable", lambda t: setattr(t, "id", swaps[t.id]))
    return out
