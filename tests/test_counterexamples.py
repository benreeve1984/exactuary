"""Wrong kernels and false statements that adversarial testing showed the checks once accepted. Each must now be refused or refuted. Tests that need Lean use the LEAN environment
variable (or `lean` on the PATH) and are skipped without it."""
import os, sys, shutil, tempfile
import numpy as np, z3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exactuary import KernelSpec, symbolic as S, lean as L
from exactuary.spec import validate_step, InvalidStep
from exactuary.compare import compare_trace
from exactuary import harness

TMP = tempfile.mkdtemp()
LEAN = os.environ.get("LEAN") or shutil.which("lean")
def toy(original, src, setup=None, exit_when=None):
    path = os.path.join(TMP, f"orig_{abs(hash(original))}.py"); open(path, "w").write(original)
    setup = setup or (lambda O: O.axiom("x", lambda: S.Sym(z3.Real("x"))))
    ss = S.SymbolicSpec(path, setup, lambda O, mp, t: {"x": O("x")}, lambda O, mp, t: [])
    return KernelSpec(src, ["x"], [], [], [], [], ["cash"], ["cash"], exit_when=exit_when), ss
def refused(fn, *errors):
    try: fn()
    except errors: return True
    return False

def test_where_without_other_is_refused():
    K, ss = toy("def cash(t):\n    return x().where(x() > 0)\n", "def step(t, is_t0, x):\n    return (sel(x > ZERO, x, ZERO),)\n")
    assert refused(lambda: S.prove(K, ss), S.Unsupported)

def test_floor_division_of_a_real_is_refused():
    K, ss = toy("def cash(t):\n    return x() // 2\n", "def step(t, is_t0, x):\n    return (x / 2,)\n")
    assert refused(lambda: S.prove(K, ss), S.Unsupported)

def test_floor_division_by_a_negative_or_symbolic_divisor_is_refused():
    assert refused(lambda: S.Sym(z3.Int("n")) // -2, S.Unsupported)
    assert refused(lambda: S.Sym(z3.Int("n")) // S.Sym(z3.Int("k")), S.Unsupported, TypeError)

def test_python_branch_on_time_is_refused():
    K, ss = toy("def cash(t):\n    if t - 1:\n        return 1.0\n    return 0.0\n", "def step(t, is_t0, x):\n    return (ONE,)\n")
    assert refused(lambda: S.prove(K, ss), S.Unsupported)

def test_table_definition_that_ignores_its_row_is_refused_or_keyed():
    orig = "def cash(t):\n    return x() * rate()\n"
    def setup(O):
        O.axiom("x", lambda: S.Sym(z3.Real("x"))); O.axiom("rate", lambda: S.Sym(z3.Real("rate_of_this_policy")))
    def spec(keys):
        path = os.path.join(TMP, "orig_table.py"); open(path, "w").write(orig)
        return S.SymbolicSpec(path, setup, lambda O, mp, t: {"x": O("x"), "pk": S.Sym(z3.Int("pk"))}, lambda O, mp, t: [],
                              tables={"TAB": lambda i: O_rate}, table_keys=keys)
    O_rate = S.Sym(z3.Real("rate_of_this_policy"))
    def kern(index): return KernelSpec(f"def step(t, is_t0, x, pk, TAB):\n    return (x * lk1(TAB, {index}),)\n", ["x", "pk"], [], [], [], ["TAB"], ["cash"], ["cash"])
    assert refused(lambda: S.prove(kern("pk"), spec({})), S.SpecError)                 # undeclared: refused
    good = S.prove(kern("pk"), spec({"TAB": {0: "pk"}}))
    assert good["proved"] == good["total"] == 2
    bad = S.prove(kern("pk + 1"), spec({"TAB": {0: "pk"}}))                            # the wrong row is no longer provable
    assert bad["proved"] == 0 and len(bad["refuted"]) == 2

def test_exit_rule_that_stops_too_early_is_refuted():
    orig = "def cash(t):\n    return x()\n"; src = "def step(t, is_t0, x):\n    return (x,)\n"
    K, ss = toy(orig, src, exit_when="t >= 0")
    verdicts = dict(S.prove_lemmas(K, ss))
    assert verdicts and any(v == "refuted" for v in verdicts.values())
    K, ss = toy(orig, src, exit_when="OUT['cash'] == ZERO")       # a sound rule for this model: zero stays zero
    assert all(v == "proved" for _, v in S.prove_lemmas(K, ss))

def test_step_outside_the_restricted_language_is_rejected():
    for bad in ("def step(t, is_t0, x):\n    import os\n    return (x,)\n",
                "def step(t, is_t0, x):\n    y = x if x > 0 else 0\n    return (y,)\n",
                "def step(t, is_t0, x):\n    return (x.real,)\n",
                "def step(t, is_t0, x):\n    return (open('f'),)\n",
                "def step(t, is_t0, x):\n    return (y,)\n",
                "def step(t, is_t0, x):\n    for i in range(3):\n        x = x + 1\n    return (x,)\n"):
        assert refused(lambda: validate_step(bad), InvalidStep), bad

def test_comparison_refuses_mismatched_shapes_and_empty_input():
    assert refused(lambda: compare_trace(np.ones((1, 2, 1)), {"cash": np.ones((2, 3))}, ["cash"], verbose=False), ValueError)
    assert refused(lambda: compare_trace(np.ones((0, 2, 1)), {}, [], verbose=False), ValueError)
    assert refused(lambda: compare_trace(np.ones((1, 0, 1)), {"cash": np.ones((0, 1))}, ["cash"], verbose=False), ValueError)

def test_infinite_original_against_finite_kernel_is_caught_per_policy():
    K = KernelSpec("def step(t, is_t0, x):\n    return (x,)\n", ["x"], [], [], [], [], ["cash"], ["cash"])
    p = {"x": np.array([0.0, 2.0]), "scen": np.array([0, 0]), "T": 1, "DISC": np.array([1.0])}
    harness._init(K, {"probe": {"p": p, "trace": np.array([[[np.inf, 2.0]]]), "pv": np.array([[np.inf], [2.0]])}})
    assert harness._judge(K.step_src)["probe_pv"] is True

def test_max_and_min_treat_nan_alike_on_every_cpu_backend():
    try: import numba
    except ImportError: return
    from exactuary.backends import numpy_vec, numba_cpu
    a, b = numpy_vec.helpers(), numba_cpu.helpers()
    for f in ("fmax", "fmin"):
        for args in ((np.nan, 1.0), (1.0, np.nan), (2.0, 1.0), (np.inf, 1.0), (-np.inf, 1.0)):
            x, y = float(a[f](*args)), float(b[f](*args))
            assert (x == y) or (np.isnan(x) and np.isnan(y)), (f, args, x, y)

def _lean_fails(statement_lhs, statement_rhs, tag):
    path = os.path.join(TMP, tag + ".lean"); open(path, "w").write(L.theorem(tag, [], statement_lhs, statement_rhs, tag))
    n, failed = L.run_lean(path, LEAN); return n == 1 and len(failed) == 1

def test_false_statements_stay_false_after_export_to_lean():
    if not LEAN: return
    x = z3.Real("x"); fa = z3.Function("rate-a", z3.RealSort(), z3.RealSort()); fb = z3.Function("rate_a", z3.RealSort(), z3.RealSort())
    assert S.check([], fa(x), fb(x))[0] == "refuted" and _lean_fails(fa(x), fb(x), "uf_collision")
    assert S.check([], x / z3.RealVal(0), z3.RealVal(0))[0] == "refuted" and _lean_fails(x / z3.RealVal(0), z3.RealVal(0), "zero_division")
    y = z3.Real("y")
    assert S.check([], x / y * y, x)[0] == "refuted" and _lean_fails(x / y * y, x, "division_cancel")

def test_true_statement_with_guarded_division_still_checks_in_lean():
    if not LEAN: return
    x, y = z3.Real("x"), z3.Real("y"); path = os.path.join(TMP, "div_ok.lean")
    open(path, "w").write(L.theorem("div_ok", [y != 0], x / y + x / y, 2 * (x / y), "div_ok"))
    assert L.run_lean(path, LEAN) == (1, [])

def test_admitted_proofs_fail_however_they_are_spelled():
    if not LEAN: return
    for k, body in enumerate(("/-- false -/\ntheorem false_theorem : (0 : Nat) = 1 := by sorry\n",
                              "theorem falseDirect : False := sorryAx False true\n",
                              "axiom cheat : False\ntheorem viaAxiom : False := cheat\n")):
        path = os.path.join(TMP, f"admit{k}.lean"); open(path, "w").write(body)
        n, failed = L.run_lean(path, LEAN, gate=False)
        assert failed, body
    path = os.path.join(TMP, "honest.lean"); open(path, "w").write("/-- true -/\ntheorem honest (a b : Nat) : a + b = b + a := by omega\n")
    assert L.run_lean(path, LEAN, gate=False) == (1, [])

def test_three_way_distinct_keeps_all_its_operands():
    x, y, z = z3.Reals("x y z"); lhs, rhs = z3.Distinct(x, y, z), x != y
    assert S.check([], lhs, rhs)[0] == "refuted"
    if LEAN: assert _lean_fails(lhs, rhs, "nary_distinct")

def test_original_that_inspects_its_values_is_refused():
    for orig in ("def cash(t):\n    return 7.0 if isinstance(x(), float) else 0.0\n",
                 "def cash(t):\n    return float(x())\n",
                 "def cash(t):\n    return x().real\n",
                 "def cash(t):\n    import os\n    return x()\n",
                 "def cash(t):\n    return (lambda v: v)(x())\n",
                 "def cash(t):\n    return eval('x()')\n",
                 "def cash(t):\n    return x() if type(x()) is float else 0\n",
                 "def cash(t):\n    for i in range(2):\n        pass\n    return x()\n"):
        K, ss = toy(orig, "def step(t, is_t0, x):\n    return (ZERO,)\n")
        assert refused(lambda: S.prove(K, ss, gate=False), S.Unsupported), orig

def test_concrete_arithmetic_in_the_original_is_exact():
    K, ss = toy("def cash(t):\n    return x() * (1 / 3) + 0.1 / 3\n", "def step(t, is_t0, x):\n    return (x / 3 + 0.1 / 3,)\n")
    o = S.prove(K, ss, gate=False); assert o["proved"] == o["total"] == 2
    K, ss = toy("def cash(t):\n    return x() * (1 / 3)\n", "def step(t, is_t0, x):\n    return (x * 0.3333333333333333,)\n")
    assert len(S.prove(K, ss, gate=False)["refuted"]) == 2      # the nearest float is not one third

def test_no_code_runs_from_a_step_or_an_exit_rule():
    marker = os.path.join(TMP, "executed.txt")
    for src in (f"def step(t, is_t0, x: __import__('pathlib').Path({marker!r}).write_text('x')):\n    return (x,)\n",
                f"def step(t, is_t0, x) -> __import__('pathlib').Path({marker!r}).write_text('x'):\n    return (x,)\n",
                f"def step(t, is_t0, x=__import__('pathlib').Path({marker!r}).write_text('x')):\n    return (x,)\n",
                f"@(lambda f: __import__('pathlib').Path({marker!r}).write_text('x') or f)\ndef step(t, is_t0, x):\n    return (x,)\n"):
        assert refused(lambda: validate_step(src), InvalidStep)
        from exactuary.backends import numpy_vec
        from exactuary.spec import make_step
        assert refused(lambda: make_step(numpy_vec.helpers(), src), InvalidStep)
    from exactuary.spec import check_spec
    K = KernelSpec("def step(t, is_t0, x):\n    return (x,)\n", ["x"], [], [], [], [], ["cash"], ["cash"],
                   exit_when=f'(__import__("pathlib").Path({marker!r}).write_text("x") > 0)')
    assert refused(lambda: check_spec(K), InvalidStep)
    _, ss = toy("def cash(t):\n    return x()\n", K.step_src)
    assert refused(lambda: S.prove_lemmas(K, ss, gate=False), InvalidStep)
    assert not os.path.exists(marker)

def test_declared_table_key_cannot_be_switched_off():
    orig = "def cash(t):\n    return x() * rate()\n"; rate = S.Sym(z3.Real("rate_of_this_policy"))
    def setup(O):
        O.axiom("x", lambda: S.Sym(z3.Real("x"))); O.axiom("rate", lambda: rate)
    path = os.path.join(TMP, "orig_table2.py"); open(path, "w").write(orig)
    def spec(table, keys): return S.SymbolicSpec(path, setup, lambda O, mp, t: {"x": O("x"), "pk": S.Sym(z3.Int("pk"))}, lambda O, mp, t: [],
                                                 tables={"TAB": table}, table_keys=keys)
    kern = lambda index: KernelSpec(f"def step(t, is_t0, x, pk, TAB):\n    return (x * lk1(TAB, {index}),)\n", ["x", "pk"], [], [], [], ["TAB"], ["cash"], ["cash"])
    sneaky = lambda i: rate + (i - i)                                       # uses the index only trivially
    bad = S.prove(kern("pk + 2"), spec(sneaky, {"TAB": {0: "pk"}}), gate=False)
    assert bad["proved"] == 0 and len(bad["refuted"]) == 2                  # the declared key still applies
    assert refused(lambda: S.prove(kern("pk + 2"), spec(sneaky, {}), gate=False), S.SpecError)    # and without one it is refused

def test_comparison_refuses_a_trace_of_the_wrong_rank():
    assert refused(lambda: compare_trace(np.ones((1, 6)), {"cash": np.ones(6)}, ["cash"], verbose=False, gate=False), ValueError)

def test_a_refutation_is_recorded_even_when_the_proof_stops_early():
    from exactuary import gates
    K, ss = toy("def cash(t):\n    return x()\n", "def step(t, is_t0, x):\n    return (x + 1,)\n")
    gates.reset(); S.prove(K, ss, stop_at_first=True)
    assert gates.results() and not gates.results()[-1][1]; gates.reset()

def test_missing_required_gate_fails_the_run():
    from exactuary import gates
    gates.reset(); gates.record("exact comparison of 1 outputs", True)
    assert gates.missing(["exact comparison", "present values"]) == ["present values"]; gates.reset()

def test_round_three_originals_are_refused():
    # each once let a numerically wrong kernel be proved
    for orig in ('def cash(t):\n    return 7 if np.maximum.__name__ == "maximum" else 0\n',
                 'def cash(t):\n    return 7 if np.minimum.__qualname__ else 0\n',
                 "def cash(t):\n    g = (i for i in range(3))\n    return sum(g) + sum(g)\n",
                 "def twice(f):\n    return f\n@twice\ndef cash(t):\n    return 3\n",
                 "def amount():\n    return 2\ndef cash(t, amount=amount()):\n    return amount\n",
                 "def cash(t):\n    return 1\ndef cash(t):\n    return 2\n",
                 "def cash(t):\n    return x().__class__\n",
                 "def cash(t):\n    return np.sqrt(x())\n"):
        K, ss = toy(orig, "def step(t, is_t0, x):\n    return (ZERO,)\n")
        assert refused(lambda: S.prove(K, ss, gate=False), S.Unsupported), orig
    K, ss = toy("def cash(t):\n    return sum(i * x() for i in range(3)) + sum([1, 2])\n", "def step(t, is_t0, x):\n    return (3 * x + 3,)\n")
    o = S.prove(K, ss, gate=False); assert o["proved"] == o["total"] == 2      # the supported forms still work

def test_unsupported_sorts_are_not_exported():
    a, b, c = [z3.Const(n, z3.BitVecSort(2)) for n in "abc"]
    assert refused(lambda: L.theorem("bv", [], z3.Distinct(a, b, c), z3.BoolVal(False), "bv"), NotImplementedError)
    P = z3.DeclareSort("Policy"); u, v = z3.Consts("u v", P)
    assert refused(lambda: L.theorem("us", [], u == v, z3.BoolVal(True), "us"), NotImplementedError)
    f = z3.Function("f", P, z3.RealSort())
    assert refused(lambda: L.theorem("uf", [], f(u), f(v), "uf"), NotImplementedError)

def test_declarations_in_unaudited_forms_fail():
    if not LEAN: return
    for k, last in enumerate(("  theorem falseIndented : False := unproved", "@[simp] theorem falseAttributed : False := unproved",
                              "protected theorem falseProt : False := unproved", "lemma falseLemma : False := unproved")):
        path = os.path.join(TMP, f"form{k}.lean"); open(path, "w").write("axiom unproved : False\ntheorem honest : True := True.intro\n" + last + "\n")
        assert L.run_lean(path, LEAN, gate=False)[1], last

def test_exit_rule_is_validated_on_every_path():
    from exactuary import errbound
    from exactuary.backends import numpy_vec
    for rule in ("abs(t) >= 0", "(t + 1) ** 2 > 0", "t if t else 0"):
        K = KernelSpec("def step(t, is_t0, x):\n    return (x,)\n", ["x"], [], [], [], [], ["cash"], ["cash"], exit_when=rule)
        p = {"x": np.array([1.0]), "scen": np.array([0]), "T": 2, "DISC": np.ones(2)}
        assert refused(lambda: errbound.run_pv_bounds(K, p), InvalidStep), rule
    K = KernelSpec("def step(t, is_t0, x):\n    return (x,)\n", ["x"], [], [], [], [], ["cash"], ["cash"], exit_when='OUT [ "cash" ] == ZERO')
    assert K.exit_expr() == "r[0] == ZERO"          # regenerated from the parsed rule, whatever its spelling

def test_distance_refuses_wider_floats():
    from exactuary.interval import distance, NotCertifiable
    enc = {"lo": np.array([[2.0 ** 53]]), "hi": np.array([[2.0 ** 53]])}
    if np.dtype(np.longdouble) != np.float64:       # on some machines long double is just float64
        assert refused(lambda: distance(np.array([[np.longdouble(2 ** 53) + np.longdouble(1)]]), enc), NotCertifiable)
    assert refused(lambda: distance(np.array([[1.0]], dtype=np.float16), enc), NotCertifiable)

def test_round_four_variants():
    # a generator nested inside an accepted one must not inherit permission to be consumed
    K, ss = toy("def cash(t):\n    return sum(sum(g) + sum(g) for g in [(i for i in range(3))])\n", "def step(t, is_t0, x):\n    return (6 * ONE,)\n")
    assert refused(lambda: S.prove(K, ss, gate=False), S.Unsupported)
    # band means logical and on every backend
    try: import numba
    except ImportError: numba = None
    if numba:
        from exactuary.backends import numpy_vec, numba_cpu
        a, b = numpy_vec.helpers(), numba_cpu.helpers()
        for args in ((2, 1), (1, 2), (1.0, -3.0), (0, 5), (True, False)):
            assert bool(a["band"](*args)) == bool(b["band"](*args)) and int(b["band"](*args)) in (0, 1), args
    # the NumPy trace validates the whole specification, exit rule included
    from exactuary.backends import numpy_vec
    K = KernelSpec("def step(t, is_t0, x):\n    return (x,)\n", ["x"], [], [], [], [], ["cash"], ["cash"], exit_when="max(t, 0) >= 0")
    assert refused(lambda: numpy_vec.run_trace(K, {"x": np.array([1.0]), "scen": np.array([0]), "T": 2, "DISC": np.ones(2)}), InvalidStep)
    # Lean: other declaration prefixes fail, even for true statements
    if LEAN:
        for k, body in enumerate(("theorem honest : True := True.intro\nnonrec theorem additional : (2 : Nat) = 2 := by rfl\n",
                                  "theorem honest : True := True.intro\nexample : (2 : Nat) = 2 := by rfl\n")):
            path = os.path.join(TMP, f"r4_{k}.lean"); open(path, "w").write(body)
            assert L.run_lean(path, LEAN, gate=False)[1], body

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
