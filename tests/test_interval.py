"""The interval arithmetic checked against exact rational arithmetic, and adversarial cases in which an
earlier first-order rounding estimate, or an earlier version of the enclosure, failed to bound the error."""
import os, sys, random
from fractions import Fraction
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exactuary import KernelSpec
from exactuary.interval import IV, enclose, distance, combine, Domain, NotCertifiable, _literal

def inside(iv, q, k=None):
    lo, hi = (float(np.asarray(iv.lo).ravel()[k or 0]), float(np.asarray(iv.hi).ravel()[k or 0]))
    return (lo == -np.inf or Fraction(lo) <= q) and (hi == np.inf or q <= Fraction(hi))

def test_random_programs_contain_the_exact_result():
    rnd = random.Random(20261121); checked = 0
    for trial in range(400):
        vals = [rnd.choice([rnd.uniform(-1e6, 1e6), rnd.uniform(-1, 1), rnd.uniform(0, 1) * 10.0 ** rnd.randint(-320, 300),
                            0.0, 1.0, 0.1, 1e-300, 5e-324, 1e16 + 1]) for _ in range(4)]
        ivs = [IV(np.float64(v), np.float64(v)) for v in vals]; qs = [Fraction(v) for v in vals]
        for _ in range(rnd.randint(3, 25)):
            i, j = rnd.randrange(len(ivs)), rnd.randrange(len(ivs)); op = rnd.choice("+-**/")
            if op == "/":
                if qs[j] == 0 or not (ivs[j].lo > 0 or ivs[j].hi < 0): continue
                a, q = ivs[i] / ivs[j], qs[i] / qs[j]
            elif op == "+": a, q = ivs[i] + ivs[j], qs[i] + qs[j]
            elif op == "-": a, q = ivs[i] - ivs[j], qs[i] - qs[j]
            else: a, q = ivs[i] * ivs[j], qs[i] * qs[j]
            if q.numerator.bit_length() + q.denominator.bit_length() > 20000: break
            assert inside(a, q), (trial, op, vals)
            ivs.append(a); qs.append(q); checked += 1
    assert checked > 3000

def test_literals_are_the_decimals_written():
    assert inside(_literal("0.1"), Fraction(1, 10)) and _literal("0.1").lo < _literal("0.1").hi
    assert _literal("0.5").lo == _literal("0.5").hi == 0.5
    assert inside(Domain().binop("/", 1, 12), Fraction(1, 12))

def _run(src, x):
    K = KernelSpec(src, ["x"], [], [], [], [], ["cash"], ["cash"])
    return enclose(K, {"x": np.array([x]), "scen": np.array([0]), "T": 1, "DISC": np.array([1.0])})

def test_branch_that_flips_under_rounding_keeps_both_outcomes():
    # exact: 1 + 2^-53 > 1, so the cash is 1,000,000; in float64 the sum rounds to 1 and the cash is 0
    e = _run("def step(t, is_t0, x):\n    y = x + 1.1102230246251565e-16\n    return (sel(y > ONE, 1000000.0, ZERO),)\n", 1.0)
    assert e["lo"][0, 0] <= 0.0 and e["hi"][0, 0] >= 1000000.0 and e["undecided"] == 1
    assert distance(np.array([[0.0]]), e)[0, 0] >= 1000000.0

def test_underflow_is_enclosed():
    e = _run("def step(t, is_t0, x):\n    return (x * 0.5,)\n", 5e-324)
    assert e["lo"][0, 0] <= 0.0 < e["hi"][0, 0]

def test_power_is_enclosed_over_the_whole_interval():
    try: import mpmath
    except ImportError: return
    y = Domain().call("fpow", [IV(np.float64(0.9), np.float64(1.1)), IV(np.float64(2.0), np.float64(2.0))])
    assert float(y.lo) <= 0.81 and float(y.hi) >= 1.21

def test_division_by_possible_zero_is_not_certified():
    e = _run("def step(t, is_t0, x):\n    return (ONE / (x - x * ONE),)\n", 3.0)
    assert not e["certified"][0]
    e = _run("def step(t, is_t0, x):\n    y = x * 0.1\n    return (ONE / (y - 0.3),)\n", 3.0)
    assert not e["certified"][0]

def test_exact_cancellation_of_the_same_quantity_is_zero():
    e = _run("def step(t, is_t0, x):\n    y = x * 0.1\n    m = sel(x > ZERO, y, ZERO)\n    return ((y - m) * 1e300 * 1e300,)\n", 3.0)
    assert e["certified"][0] and e["lo"][0, 0] == 0.0 == e["hi"][0, 0]

def _encl(src, x, T=1, consts=None, k=None):
    names = ["k"] if k is not None else []
    K = KernelSpec(src, ["x"], [], [], [], names, ["cash"], ["cash"])
    p = {"x": np.array([x]), "scen": np.array([0]), "T": T, "DISC": np.ones(T)}
    if k is not None: p["k"] = k
    return enclose(K, p)
def _holds(e, q): return (not e["certified"][0]) or (Fraction(float(e["lo"][0, 0])) <= q <= Fraction(float(e["hi"][0, 0])))
def _refused(f):
    try: f()
    except NotCertifiable: return True
    return False

def test_nothing_is_computed_before_it_is_tracked():
    # each once gave a zero-width interval that missed the exact value
    e = _encl("def step(t, is_t0, x):\n    return (sel(t == 3, t / 10, ZERO),)\n", 1.0, T=4)
    assert _holds(e, Fraction(3, 10)) and e["lo"][0, 0] < e["hi"][0, 0]
    e = _encl("def step(t, is_t0, x):\n    return (x / 10,)\n", 3)                     # integer policy input
    assert _holds(e, Fraction(3, 10)) and e["lo"][0, 0] < e["hi"][0, 0]
    e = _encl("def step(t, is_t0, x, k):\n    return (x * (k / 10),)\n", 1.0, k=3)      # integer scalar constant
    assert _holds(e, Fraction(3, 10)) and e["lo"][0, 0] < e["hi"][0, 0]
    assert _holds(_encl("def step(t, is_t0, x):\n    return (1 / 3 + 0.1 / 3,)\n", 1.0), Fraction(1, 3) + Fraction(1, 30))

def test_integer_overflow_is_refused():
    assert _refused(lambda: _encl("def step(t, is_t0, x):\n    return (tof(x * x),)\n", 2 ** 32))
    assert _refused(lambda: _encl("def step(t, is_t0, x):\n    return (tof(x + x),)\n", 2 ** 62))
    assert _refused(lambda: _encl("def step(t, is_t0, x):\n    return (tof(x),)\n", 2 ** 53 + 1))

def test_float32_inputs_are_refused():
    assert _refused(lambda: _encl("def step(t, is_t0, x):\n    return (x,)\n", np.float32(0.1)))

def test_distance_and_combine_refuse_values_they_cannot_represent():
    enc = {"lo": np.array([[2.0 ** 53, 2.0 ** 53]]), "hi": np.array([[2.0 ** 53, 2.0 ** 53]])}
    assert _refused(lambda: distance(np.array([[2 ** 53 + 1, 0]], dtype=np.int64), enc))
    lo, hi = combine({"lo": np.array([[1.0, 1.0]]), "hi": np.array([[1.0, 1.0]])}, [1, 2 ** 53])
    assert lo[0] <= 2 ** 53 + 1 <= hi[0] and lo[0] < hi[0]
    assert _refused(lambda: combine({"lo": np.array([[1.0, 1.0]]), "hi": np.array([[1.0, 1.0]])}, [1, 2 ** 53 + 1]))

if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
