"""lifelib `basiclife/BasicTerm_ME` (level-premium term life, new and in-force business).
Everything specific to this model lives in this file: test portfolios, the step function,
input preparation, and the symbolic description used for the proofs."""
import os, time, numpy as np, pandas as pd
from exactuary import KernelSpec
from exactuary.oracle import cell_name, capture

# ----------------------------------------------------------------- original model (the oracle)
T_CELLS = ["age", "claim_pp", "claims", "commissions", "duration", "duration_mth", "expenses", "lapse_rate",
           "mort_rate", "mort_rate_mth", "net_cf", "pols_death", "pols_if", "pols_lapse", "pols_maturity",
           "pols_new_biz", "premiums"]
ARG_CELLS = [("pols_if_at", k) for k in ("BEF_MAT", "BEF_NB", "BEF_DECR")]
OUTPUTS = [cell_name(c) for c in T_CELLS] + [cell_name(c, k) for c, k in ARG_CELLS]
PV_NAMES = ["PV Premiums", "PV Claims", "PV Expenses", "PV Commissions", "PV Net Cashflow"]
ACCUMULATE = ["premiums", "claims", "expenses", "commissions", "pols_if_at[BEF_DECR]"]

def lib_path():
    import lifelib
    return os.path.join(os.path.dirname(lifelib.__file__), "libraries", "basiclife")
def source_path(): return os.path.join(lib_path(), "BasicTerm_ME")
def formulas_path(): return os.path.join(source_path(), "Projection", "__init__.py")
def load_original():
    import modelx as mx
    return mx.read_model(source_path())
def load_unvectorised():
    import modelx as mx
    return mx.read_model(os.path.join(lib_path(), "BasicTerm_SE"))

COLS = ["age_at_entry", "sex", "policy_term", "policy_count", "sum_assured", "duration_mth"]
def _index(mp):
    mp = mp.reset_index(drop=True); mp.index = mp.index + 1; mp.index.name = "policy_id"
    return mp
def portfolio_sample(m, copies=1):
    """lifelib's own 10,000 sample model points, repeated `copies` times."""
    mp = m.Projection.model_point_table
    return mp.copy() if copies == 1 else _index(mp.iloc[np.tile(np.arange(len(mp)), copies)])
def portfolio_edge(n=2000, seed=7):
    """Random model points chosen to reach edge cases while every value stays finite: new business,
    in-force, future-dated (negative duration), at and beyond maturity, far beyond term (ages past the
    end of the mortality table), ages below the mortality table, zero and extreme counts and sums assured.
    Entry age and term stay inside the premium table (see portfolio_edge_nan for the rest)."""
    r = np.random.default_rng(seed)
    term = r.choice([10, 15, 20], n)
    dur = np.where(r.random(n) < 0.25, 0, r.integers(-40, 260, n))
    u = r.random(n)
    dur = np.where(u < 0.15, term * 12 + r.integers(-2, 3, n), dur)                 # around maturity
    dur = np.where((u >= 0.15) & (u < 0.22), r.integers(700, 1300, n), dur)        # far beyond term: age > 120
    age = r.integers(20, 60, n)
    age = np.where(r.random(n) < 0.2, r.choice([20, 21, 59], n), age)
    mp = pd.DataFrame({"age_at_entry": age, "sex": r.choice(["M", "F"], n), "policy_term": term,
                       "policy_count": r.choice([0, 1, 37, 100, 10**9], n, p=[.1, .2, .3, .3, .1]),
                       "sum_assured": r.choice([0, 1, 1000, 622_000, 999_999, 5 * 10**9], n),
                       "duration_mth": dur})
    return _index(mp)
def portfolio_edge_nan(n=600, seed=11):
    """Model points for which the original itself produces NaN: entry age or term outside the premium
    table (pandas reindex without a fill value gives NaN premiums), term 0 (maturity and new business in
    the same month), and durations so negative that the lapse rate exceeds 1 (NaN from a fractional
    power of a negative number). The exact comparison treats NaN = NaN as identical."""
    r = np.random.default_rng(seed)
    term = r.choice([0, 1, 2, 5, 10, 12, 15, 20, 30], n)
    dur = np.where(r.random(n) < 0.3, 0, r.integers(-60, 400, n))
    dur = np.where(r.random(n) < 0.1, r.integers(-640, -530, n), dur)
    dur = np.where(r.random(n) < 0.1, term * 12 + r.integers(-1, 2, n), dur)
    mp = pd.DataFrame({"age_at_entry": r.integers(5, 80, n), "sex": r.choice(["M", "F"], n), "policy_term": term,
                       "policy_count": r.choice([0, 1, 37, 100], n), "sum_assured": r.choice([0, 1000, 622_000], n),
                       "duration_mth": dur})
    return _index(mp)
def configure(m, mp):
    P = m.Projection; P.model_point_table = mp
    return P
def run_original(P, trace=True):
    t0 = time.perf_counter(); pv = P.result_pv(); sec = time.perf_counter() - t0
    T = P.max_proj_len()
    res = {"pv": pv[PV_NAMES].to_numpy(dtype=np.float64), "T": T, "seconds_pv": sec}
    if trace:
        res["trace"] = capture(P, OUTPUTS, T)
        res["inflation_factor"] = np.array([P.inflation_factor(t) for t in range(T)])
        res["disc_factors"] = np.asarray(P.disc_factors(), dtype=np.float64)
        res["premium_pp"] = np.asarray(P.premium_pp(), dtype=np.float64)
        res["pv_pols_if"] = np.asarray(P.pv_pols_if(), dtype=np.float64)
        res["net_premium_pp"] = np.asarray(P.net_premium_pp(), dtype=np.float64)
        res["proj_len"] = np.asarray(P.proj_len(), dtype=np.int64)
    return res
def reported(acc):
    """The original's five reported present values from the accumulated outputs (same expression as pv_net_cf)."""
    return np.column_stack([acc[:, 0], acc[:, 1], acc[:, 2], acc[:, 3], acc[:, 0] - acc[:, 1] - acc[:, 2] - acc[:, 3]])
def net_premium(acc):
    """The original's net_premium_pp from the accumulated outputs (same expression)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(acc[:, 1] / acc[:, 4])

# ----------------------------------------------------------------- the rewrite
STEP_SRC = '''
def step(t, is_t0, age0, term, pcount, sa, dm0, prem, infl, s_p_bef_mat, TAB_MORT, TAB_MORTMTH, TAB_LAPSEMTH, DUR_MIN, AGE_MIN):
    dm = dm0 + t
    dur = idiv(dm, 12)
    age = age0 + dur
    term12 = term * 12
    p_bef_mat = sel(is_t0, sel(dm0 > 0, pcount, ZERO), s_p_bef_mat)
    p_mat = sel(dm == term12, ONE, ZERO) * p_bef_mat
    p_bef_nb = p_bef_mat - p_mat
    p_nb = sel(dm == 0, pcount, ZERO)
    p_decr = p_bef_nb + p_nb
    d5 = imin(dur, 5)
    mort = lk2(TAB_MORT, age - AGE_MIN, d5)
    q = lk2(TAB_MORTMTH, age - AGE_MIN, d5)
    p_death = p_decr * q
    lapse_rate = fmax(0.1 - 0.02 * tof(dur), 0.02)
    lq = lk1(TAB_LAPSEMTH, dur - DUR_MIN)
    p_lapse = (p_decr - p_death) * lq
    p_next = p_decr - p_lapse - p_death
    premiums = prem * p_decr
    claims = sa * p_death
    comm = sel(dur == 0, ONE, ZERO) * premiums
    expenses = 300.0 * p_nb + p_decr * 60.0 / 12.0 * infl
    net_cf = premiums - claims - expenses - comm
    return (tof(age), sa, claims, comm, tof(dur), tof(dm), expenses, lapse_rate,
            mort, q, net_cf, p_death, p_bef_mat, p_lapse, p_mat,
            p_nb, premiums,
            p_bef_mat, p_bef_nb, p_decr,
            p_next)
'''
# The same step with the algebra rearranged. It is not bit-identical to the original; it is
# proved equal over the real numbers and its rounding is bounded.
_REARRANGE = [("p_lapse = (p_decr - p_death) * lq", "p_surv = p_decr * (1 - q)\n    p_lapse = p_surv * lq"),
              ("p_next = p_decr - p_lapse - p_death", "p_next = p_surv - p_lapse"),
              ("expenses = 300.0 * p_nb + p_decr * 60.0 / 12.0 * infl", "expenses = 300.0 * p_nb + p_decr * (5.0 * infl)")]
STEP_REARRANGED_SRC = STEP_SRC
for _a, _b in _REARRANGE:
    assert _a in STEP_REARRANGED_SRC; STEP_REARRANGED_SRC = STEP_REARRANGED_SRC.replace(_a, _b)

KERNEL = KernelSpec(
    step_src=STEP_SRC,
    row_args=["age0", "term", "pcount", "sa", "dm0", "prem"],
    scen_series=[], time_series=["infl"], state=["s_p_bef_mat"],
    consts=["TAB_MORT", "TAB_MORTMTH", "TAB_LAPSEMTH", "DUR_MIN", "AGE_MIN"],
    outputs=OUTPUTS, accumulate=ACCUMULATE,
    exit_when="(OUT['pols_if_at[BEF_DECR]'] == ZERO) & (ROW['dm0'] + t >= 0)")
KERNEL_REARRANGED = KERNEL.with_src(STEP_REARRANGED_SRC)

# names a careless translator might confuse, for the wrong-variable seeded defects
SWAPS = {"p_bef_mat": "p_bef_nb", "p_bef_nb": "p_decr", "p_decr": "p_bef_mat", "p_death": "p_lapse", "p_lapse": "p_death",
         "q": "lq", "lq": "q", "dm": "dm0", "dur": "dm", "p_next": "p_decr", "pcount": "sa", "sa": "pcount", "prem": "sa",
         "mort": "q", "premiums": "claims", "claims": "premiums", "age": "age0", "term12": "term", "p_nb": "p_mat",
         "p_mat": "p_nb", "s_p_bef_mat": "pcount", "dm0": "dm", "comm": "premiums"}

def extract_tables(P):
    """Read the assumption tables out of the original model."""
    mt = P.mort_table; pt = P.premium_table
    return {"mort": mt.to_numpy(dtype=np.float64), "mort_age0": int(mt.index.min()),
            "prem": {(int(a), int(b)): float(v) for (a, b), v in pt.items()},
            "disc_rate_ann": P.disc_rate_ann.copy()}

def prep(mp, tb, dtype=np.float64):
    """Turn a model point table into the kernel's inputs. Every derived table uses the same expression
    as the original, so that the kernel's inputs are bit-identical. The premium per policy is an input
    of the projection (premium table rate x sum assured, rounded to 2 places with the original's NumPy
    call); the rate is looked up here with a dictionary, not with the original's pandas reindex, and
    the result is checked against the original's premium_pp cell."""
    age0 = mp["age_at_entry"].to_numpy(np.int64); term = mp["policy_term"].to_numpy(np.int64)
    dm0 = mp["duration_mth"].to_numpy(np.int64); sa = mp["sum_assured"].to_numpy()
    T = int(np.max(np.maximum(12 * term - dm0 + 1, 0)))
    get = tb["prem"].get
    rate = np.array([get((a, b), np.nan) for a, b in zip(age0.tolist(), term.tolist())], dtype=np.float64)
    prem = np.around(sa * rate, 2)
    dur_min, dur_max = int(dm0.min() // 12), int((dm0.max() + T) // 12)
    d = np.arange(dur_min, dur_max + 1)
    dra = tb["disc_rate_ann"]
    disc_rate_mth = np.array(list(((1 + dra[t // 12]) ** (1 / 12) - 1 for t in range(T))))
    DISC = np.array(list(((1 + disc_rate_mth[t]) ** (-t) for t in range(T))))
    INFL = np.array([(1 + 0.01) ** (t / 12) for t in range(T)])
    f = lambda a: np.ascontiguousarray(a, dtype=dtype)
    with np.errstate(invalid="ignore"):
        lapse_mth = 1 - (1 - np.maximum(0.1 - 0.02 * d, 0.02)) ** (1 / 12)
    return {"age0": age0, "term": term, "pcount": f(mp["policy_count"]), "sa": f(sa), "dm0": dm0, "prem": f(prem),
            "scen": np.zeros(len(mp), np.int64), "infl": f(INFL), "DISC": f(DISC), "T": T,
            "TAB_MORT": f(tb["mort"]), "TAB_MORTMTH": f(1 - (1 - tb["mort"]) ** (1 / 12)),
            "TAB_LAPSEMTH": f(lapse_mth), "DUR_MIN": dur_min, "AGE_MIN": tb["mort_age0"]}
SHARED = ("T", "infl", "DISC", "TAB_LAPSEMTH", "DUR_MIN")      # inputs that depend on the whole portfolio

# ----------------------------------------------------------------- symbolic description
def symbolic_spec(path=None):
    import z3
    from exactuary.symbolic import SymbolicSpec, Sym, smax, lift, tval, POW
    I, R = z3.Int, z3.Real
    MORT = z3.Function("MORT", z3.IntSort(), z3.IntSort(), z3.RealSort())     # mortality table lookup
    twelfth = z3.Q(1, 12)                # exactly one twelfth, as the original's `1/12` denotes
    def setup(O):
        mp = {"age_at_entry": Sym(I("age_at_entry")), "policy_term": Sym(I("policy_term")), "policy_count": Sym(R("policy_count")),
              "sum_assured": Sym(R("sum_assured")), "duration_mth": Sym(I("duration_mth"))}
        # Axioms: the inputs, and the two pandas table lookups (checked exhaustively by enumeration instead).
        O.axiom("model_point", lambda: mp)
        O.axiom("mort_rate", lambda t: Sym(MORT(lift(O("age", t)), lift(O.ns["np"].minimum(O("duration", t), 5)))))
        O.axiom("premium_pp", lambda: Sym(R("premium_pp")))
        return mp
    def kernel_args(O, mp, t):
        return {"age0": mp["age_at_entry"], "term": O("policy_term"), "pcount": mp["policy_count"], "sa": O("sum_assured"),
                "dm0": mp["duration_mth"], "prem": O("premium_pp"), "infl": O("inflation_factor", t)}
    def carry(O, mp, t):
        return [O("pols_if_at", t, "BEF_DECR") - O("pols_lapse", t) - O("pols_death", t)]
    def invariants(O, mp, t):
        return [("duration_mth(t) = duration_mth(0) + t", O("duration_mth", t), mp["duration_mth"] + tval(t))]
    DUR_MIN = Sym(I("DUR_MIN"))
    def lapse_tab(i):       # how the hoisted monthly lapse table is defined, in terms of duration = i + DUR_MIN
        lr = smax(0.1 - 0.02 * Sym(z3.ToReal(lift(i + DUR_MIN))), 0.02)
        return Sym(1 - POW(1 - lift(lr), twelfth))
    tables = {"TAB_MORT": lambda i, j: Sym(MORT(lift(i + 18), lift(j))),
              "TAB_MORTMTH": lambda i, j: Sym(1 - POW(1 - MORT(lift(i + 18), lift(j)), twelfth)),
              "TAB_LAPSEMTH": lapse_tab}
    def lemmas(step, mp):
        """Early exit: if no policies are in force and the policy has started, then next month every
        accumulated cashflow is zero and the same holds again."""
        t = z3.Int("t")
        a = {"age0": mp["age_at_entry"], "term": Sym(I("term")), "pcount": mp["policy_count"], "sa": mp["sum_assured"],
             "dm0": mp["duration_mth"], "prem": Sym(R("premium_pp")), "infl": Sym(R("infl"))}
        r = step(Sym(t), False, *[a[k] for k in KERNEL.row_args + KERNEL.scen_series + KERNEL.time_series],
                 0.0, "TAB_MORT", "TAB_MORTMTH", "TAB_LAPSEMTH", DUR_MIN, 18)
        assume = [t >= 1, lift(mp["duration_mth"]) + t >= 1]
        goals = [(n, r[KERNEL.ix(n)]) for n in ACCUMULATE + ["net_cf"]] + [("policies carried forward", r[KERNEL.n_out])]
        return [(f"early exit: {n} is zero", assume, lift(g), z3.RealVal(0)) for n, g in goals]
    return SymbolicSpec(source_path=path or formulas_path(), setup=setup, kernel_args=kernel_args, carry=carry,
                        invariants=invariants, tables=tables, consts={"DUR_MIN": DUR_MIN, "AGE_MIN": 18},
                        int_cells=("duration_mth",), lemmas=lemmas)

COMPOSITE = {"net_cf": ["premiums", "claims", "expenses", "commissions"]}
