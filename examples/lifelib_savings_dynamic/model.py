"""lifelib `savings/CashValue_ME_EX2`: variable-annuity style savings product with a guaranteed
maturity benefit (GMAB), guaranteed death benefit (GMDB), Monte Carlo over investment-return
scenarios, a simulation-parameter table (`sim_id` 1..5) that switches fees / mortality / lapse on,
and (sim_id 5) a dynamic lapse rate that depends on the account value, i.e. on carried state.

Everything specific to this model lives in this file: test portfolios, the step functions,
input preparation, and the symbolic description used for the proofs."""
import os, time, numpy as np, pandas as pd
from exactuary import KernelSpec
from exactuary.oracle import cell_name, capture

# ----------------------------------------------------------------- original model (the oracle)
T_CELLS = ["age", "av_change", "coi", "coi_pp", "coi_rate", "commissions", "csv_pp", "duration", "duration_mth",
           "expenses", "inv_income", "inv_income_pp", "inv_return_mth", "lapse_rate", "maint_fee", "maint_fee_pp",
           "margin_expense", "margin_mortality", "mort_rate", "mort_rate_mth", "net_amt_at_risk", "net_cf",
           "pols_death", "pols_if", "pols_lapse", "pols_maturity", "pols_new_biz", "prem_to_av", "prem_to_av_pp",
           "premium_pp", "premiums", "surr_charge", "surr_charge_rate"]
ARG_CELLS = ([("av_at", k) for k in ("BEF_MAT", "BEF_NB", "BEF_FEE")]
             + [("av_pp_at", k) for k in ("BEF_PREM", "BEF_FEE", "BEF_INV", "MID_MTH")]
             + [("pols_if_at", k) for k in ("BEF_MAT", "BEF_NB", "BEF_DECR")]
             + [("claim_pp", k) for k in ("DEATH", "LAPSE", "MATURITY")]
             + [("claim_net_pp", k) for k in ("DEATH", "LAPSE", "MATURITY")]
             + [("claims", k) for k in ("DEATH", "LAPSE", "MATURITY", None)]
             + [("claims_from_av", k) for k in ("DEATH", "LAPSE", "MATURITY")]
             + [("claims_over_av", k) for k in ("DEATH", "LAPSE", "MATURITY")])
OUTPUTS = [cell_name(c) for c in T_CELLS] + [cell_name(c, k) for c, k in ARG_CELLS]
# reported present values: the nine of result_pv, the three behind monte_carlo, and the other pv_* cells
PV_NAMES = ["Premiums", "Death", "Surrender", "Maturity", "Expenses", "Commissions",
            "Investment Income", "Change in AV", "Net Cashflow"]
PV_ALL = PV_NAMES + ["GMDB (claims over AV, death)", "GMAB (claims over AV, maturity)", "PV Fees",
                     "Claims from AV, death", "Claims from AV, lapse", "Claims from AV, maturity", "PV policies in force"]
ACCUMULATE = ["premiums", "claims[DEATH]", "claims[LAPSE]", "claims[MATURITY]", "expenses",
              "commissions", "inv_income", "av_change", "claims[ALL]",
              "claims_over_av[DEATH]", "claims_over_av[MATURITY]", "maint_fee",
              "claims_from_av[DEATH]", "claims_from_av[LAPSE]", "claims_from_av[MATURITY]", "pols_if_at[BEF_DECR]"]
N_RAND_T = 242           # the original draws 242 months of normals per scenario; projections cannot be longer
SIM_IDS = (1, 2, 3, 4, 5)

def source_path():
    import lifelib
    return os.path.join(os.path.dirname(lifelib.__file__), "libraries", "savings", "CashValue_ME_EX2")
def formulas_path(): return os.path.join(source_path(), "Projection", "__init__.py")
def load_original():
    import modelx as mx
    return mx.read_model(source_path())

def _finish(df):
    df.index = pd.RangeIndex(1, len(df) + 1, name="point_id"); df["accum_prem_init_pp"] = 0
    return df.astype({c: np.int64 for c in df.columns if c not in ("spec_id", "sex")})
def portfolio_sample1(m):
    """The model's own model point (the notebook's configuration)."""
    return m.Projection.model_point_1.copy()
def portfolio_money(m):
    """The model's own in-the-moneyness model points."""
    return m.Projection.model_point_moneyness.copy()
def portfolio_edge(n, last_age, wild=False, seed=7):
    """Random model points chosen to reach edge cases: in-force, future-dated and past-term policies,
    whole-of-life products, ages at and beyond the mortality table, zero premiums and account values.
    The original draws only 242 months of random numbers, so every projection is kept within 241 months.
    wild=False keeps the sum assured at least a quarter of the funding so that the dynamic lapse rate
    stays below 1; wild=True draws the sum assured independently (including zero), which drives the
    original's lapse rate above 1 or to infinity and its results to NaN."""
    r = np.random.default_rng(seed)
    spec = r.choice(list("ABCD"), n); wl = np.isin(spec, ["C", "D"])
    term = r.choice([1, 2, 5, 10, 15, 20, 30], n)
    age = r.integers(18, 75, n)
    age = np.where(wl & (r.random(n) < 0.4), last_age - r.integers(-3, 21, n), age)
    age = np.where(~wl & (r.random(n) < 0.1), r.integers(95, 125, n), age)
    eff_term = np.where(wl, last_age - age, term)
    dur = np.where(r.random(n) < 0.35, 0, r.integers(-30, 400, n))
    dur = np.where(r.random(n) < 0.15, eff_term * 12 + r.integers(-2, 3, n), dur)
    dur = np.maximum(dur, 12 * eff_term - 240)
    prem = r.choice([0, 100, 3100, 6700, 422_000, 900_000], n)
    av = np.where(dur > 0, r.choice([0, 10, 40_000, 2_000_000], n), 0)
    if wild:
        sa = r.choice([0, 1000, 500_000, 5_000_000], n)
    else:
        single = np.isin(spec, ["A", "B"])
        funding = av + prem * np.where(single, 1, np.maximum(12 * eff_term - dur, 0))
        sa = np.maximum((funding * r.choice([0.25, 0.5, 1.0, 2.0, 10.0], n)).astype(np.int64), 1000)
    return _finish(pd.DataFrame({"spec_id": spec, "age_at_entry": age, "sex": r.choice(["M", "F"], n), "policy_term": term,
                                 "policy_count": r.integers(0, 200, n), "sum_assured": sa, "duration_mth": dur,
                                 "premium_pp": prem, "av_pp_init": av}))
def portfolio_scale(n, seed=11):
    """New-business single-premium policies around the model's own model point, for timing at scale."""
    r = np.random.default_rng(seed)
    return _finish(pd.DataFrame({"spec_id": r.choice(list("AB"), n), "age_at_entry": r.integers(40, 80, n), "sex": "M",
                                 "policy_term": 10, "policy_count": 100, "sum_assured": 500_000, "duration_mth": 0,
                                 "premium_pp": r.integers(300_000, 500_001, n), "av_pp_init": 0}))

def configure(m, mp, n_scen, sim_id):
    """Only inputs are set: the model point table, the number of scenarios and the simulation id."""
    P = m.Projection; P.model_point_table = mp; P.scen_size = n_scen; P.sim_id = sim_id
    return P
def read_flags(P):
    return {k: bool(getattr(P, k)()) for k in ("has_fees", "has_mortality", "has_lapse", "is_lapse_dynamic")}
def run_original(P, trace=True):
    t0 = time.perf_counter(); pv = P.result_pv(); t1 = time.perf_counter(); mc = P.monte_carlo(); t2 = time.perf_counter()
    T = P.max_proj_len(); n = len(P.model_point())
    cols = ([pv[c].to_numpy(dtype=np.float64) for c in PV_NAMES]
            + [P.pv_claims_over_av("DEATH"), P.pv_claims_over_av("MATURITY"), P.pv_maint_fee()]
            + [P.pv_claims_from_av(k) for k in ("DEATH", "LAPSE", "MATURITY")] + [P.pv_pols_if()])
    res = {"pv": np.column_stack([np.asarray(c, dtype=np.float64) for c in cols]), "T": T,
           "seconds_pv": t1 - t0, "seconds_mc": t2 - t1, "monte_carlo": mc.to_numpy(dtype=np.float64)}
    if trace:
        tr = capture(P, OUTPUTS, T)
        for k, v in tr.items():
            # exactuary.oracle.capture returns [month] for cells that are scalars (coi_rate, claim_net_pp[LAPSE])
            # and [month, scenario] for inv_return_mth; bring both to [month, row] (rows are point-major)
            if v.ndim == 1: tr[k] = np.repeat(v[:, None], n, axis=1)
            elif v.shape[1] != n: tr[k] = np.tile(v, (1, n // v.shape[1]))
        res["trace"] = tr
        res["inflation_factor"] = np.array([P.inflation_factor(t) for t in range(T)])
        res["disc_factors"] = np.asarray(P.disc_factors(), dtype=np.float64)
    return res
def reported(acc):
    """The sixteen reported present values from the sixteen accumulated outputs (net cashflow is derived
    with the original's own expression)."""
    net = acc[:, 0] + acc[:, 6] - acc[:, 8] - acc[:, 4] - acc[:, 5] - acc[:, 7]
    return np.column_stack([acc[:, k] for k in range(8)] + [net] + [acc[:, k] for k in range(9, 16)])
def monte_carlo_from(pv, mp, n_scen):
    """The original's `monte_carlo` expression applied to per-row present values."""
    idx = pd.MultiIndex.from_product([mp.index, range(1, n_scen + 1)], names=["point_id", "scen_id"])
    death, mat, fee = pv[:, 9], pv[:, 10], pv[:, 11]
    result = pd.DataFrame({"GMDB": death, "GMAB": mat, "GMxB Total": death + mat, "PV Fees": fee},
                          index=idx).groupby(level="point_id").mean()
    result["Coverage Ratio"] = result["PV Fees"] / result["GMxB Total"]
    return result.to_numpy(dtype=np.float64)

# ----------------------------------------------------------------- the rewrite
# One step with two variants of the lapse block. In simulations without dynamic lapse the lapse rate
# depends only on duration, so its twelfth root is hoisted into a table (exact). With dynamic lapse
# the rate depends on the account value carried from month to month, so the root is taken in the step.
# `(dm == term12) * pols` in the original is a boolean times a number: False * NaN is NaN, hence
# sel(c, x, ZERO * x) rather than sel(c, x, ZERO).
_STEP = '''
def step(t, is_t0, age0, term, pcount, sa, dm0, prem, av0, is_single, is_level, has_sc, sc_col, load,
         inv_ret, infl, s_av_prem, s_p_bef_mat,
         TAB_MORT, TAB_MORTMTH, TAB_LAPSEMTH, TAB_SC, DUR_MIN, AGE_MIN, FEE_RATE, HAS_LAPSE, TWELFTH):
    dm = dm0 + t
    dur = idiv(dm, 12)
    age = age0 + dur
    term12 = term * 12
    av_prem = sel(is_t0, av0, s_av_prem)
    p_bef_mat = sel(is_t0, sel(dm0 > 0, pcount, ZERO), s_p_bef_mat)
    p_mat = sel(dm == term12, p_bef_mat, ZERO * p_bef_mat)
    p_bef_nb = p_bef_mat - p_mat
    p_nb = sel(dm == 0, pcount, ZERO)
    p_decr = p_bef_nb + p_nb
    prem_pp = sel(band(is_single, dm == 0), prem, ZERO) + sel(band(is_level, dm < term12), prem, ZERO)
    prem_av_pp = (1 - load) * prem_pp
    av_fee = av_prem + prem_av_pp
    mfee_pp = FEE_RATE * av_fee
    nar = fmax(sa - av_fee, ZERO)
    d5 = imin(dur, 5)
    mort = lk2(TAB_MORT, age - AGE_MIN, d5)
    q = lk2(TAB_MORTMTH, age - AGE_MIN, d5)
    coi_rate = ZERO
    coi_pp = coi_rate * nar
    av_inv = av_fee - mfee_pp - coi_pp
    iipp = inv_ret * av_inv
    av_mid = av_inv + 0.5 * iipp
    p_death = p_decr * q
    sc_rate = sel(has_sc, lk2(TAB_SC, imin(dur, 10), sc_col), ZERO)
    csv = (1 - sc_rate) * av_mid
#LAPSE#
    p_lapse = (p_decr - p_death) * lq
    p_next = p_decr - p_lapse - p_death
    av_next = av_inv + iipp
    premiums = prem_pp * p_decr
    cl_death_pp = fmax(sa, av_mid)
    cl_mat_pp = fmax(sa, av_prem)
    cl_death = cl_death_pp * p_death
    cfa_death = av_mid * p_death
    cfa_lapse = av_mid * p_lapse
    cfa_mat = av_prem * p_mat
    surr = sc_rate * av_mid * p_lapse
    cl_lapse = cfa_lapse - surr
    cl_mat = cl_mat_pp * p_mat
    cl_all = ZERO + cl_death + cl_lapse + cl_mat
    expenses = 5000.0 * p_nb + p_decr * 500.0 / 12.0 * infl
    commissions = 0.05 * premiums
    inv_income = iipp * p_next + 0.5 * iipp * (p_death + p_lapse)
    av_bef_mat = av_prem * p_bef_mat
    av_change = av_next * p_next - av_bef_mat
    net_cf = premiums + inv_income - cl_all - expenses - commissions - av_change
    coi = coi_pp * p_decr
    mfee = mfee_pp * p_decr
    margin_exp = load * prem_pp * p_decr + surr + mfee - commissions - expenses
    coa_death = cl_death - cfa_death
    margin_mort = coi - coa_death
    return (tof(age), av_change, coi, coi_pp, coi_rate, commissions, csv, tof(dur), tof(dm),
            expenses, inv_income, iipp, inv_ret, lapse_rate, mfee, mfee_pp,
            margin_exp, margin_mort, mort, q, nar, net_cf,
            p_death, p_bef_mat, p_lapse, p_mat, p_nb, prem_av_pp * p_decr, prem_av_pp,
            prem_pp, premiums, surr, sc_rate,
            av_bef_mat, av_prem * p_bef_nb, av_fee * p_decr,
            av_prem, av_fee, av_inv, av_mid,
            p_bef_mat, p_bef_nb, p_decr,
            cl_death_pp, av_mid, cl_mat_pp,
            cl_death_pp - av_mid, ZERO, cl_mat_pp - av_prem,
            cl_death, cl_lapse, cl_mat, cl_all,
            cfa_death, cfa_lapse, cfa_mat,
            coa_death, cl_lapse - cfa_lapse, cl_mat - cfa_mat,
            av_next, p_next)
'''
_LAPSE_STATIC = '''    lapse_rate = sel(HAS_LAPSE, fmax(0.1 - 0.01 * tof(dur), 0.02), ZERO)
    lq = lk1(TAB_LAPSEMTH, dur - DUR_MIN)'''
# The exponent 1/12 is passed as the constant TWELFTH rather than written as a literal: exactuary.errbound
# cannot evaluate fpow with a literal exponent in the float32 source (to_f32_src wraps it in C(), which the
# error analysis turns into an interval object that fpow then rejects).
_LAPSE_DYNAMIC = '''    lapse_rate = csv / sa * fmax(0.1 - 0.01 * tof(dur), 0.02)
    lq = 1 - fpow(1 - lapse_rate, TWELFTH)'''
STEP_STATIC_SRC = _STEP.replace("#LAPSE#", _LAPSE_STATIC)
STEP_DYNAMIC_SRC = _STEP.replace("#LAPSE#", _LAPSE_DYNAMIC)

_SPEC = dict(row_args=["age0", "term", "pcount", "sa", "dm0", "prem", "av0", "is_single", "is_level", "has_sc", "sc_col", "load"],
             scen_series=["inv_ret"], time_series=["infl"], state=["s_av_prem", "s_p_bef_mat"],
             consts=["TAB_MORT", "TAB_MORTMTH", "TAB_LAPSEMTH", "TAB_SC", "DUR_MIN", "AGE_MIN", "FEE_RATE", "HAS_LAPSE", "TWELFTH"],
             outputs=OUTPUTS, accumulate=ACCUMULATE,
             exit_when="(OUT['pols_if_at[BEF_DECR]'] == ZERO) & (ROW['dm0'] + t >= 0)")
KERNEL_STATIC = KernelSpec(step_src=STEP_STATIC_SRC, **_SPEC)
KERNEL_DYNAMIC = KernelSpec(step_src=STEP_DYNAMIC_SRC, **_SPEC)
def is_dynamic(flags): return bool(flags["has_lapse"] and flags["is_lapse_dynamic"])
def kernel_for(flags): return KERNEL_DYNAMIC if is_dynamic(flags) else KERNEL_STATIC

# names a careless translator might confuse, for the wrong-variable seeded defects
SWAPS = {"av_prem": "av_fee", "av_fee": "av_inv", "av_inv": "av_mid", "av_mid": "av_inv", "p_bef_mat": "p_bef_nb",
         "p_bef_nb": "p_decr", "p_decr": "p_bef_mat", "p_death": "p_lapse", "p_lapse": "p_death", "q": "lq", "lq": "q",
         "prem_pp": "prem_av_pp", "cfa_death": "cfa_lapse", "cl_death": "cl_lapse", "dm": "dm0", "dur": "dm",
         "p_next": "p_decr", "av_next": "av_inv", "iipp": "av_inv", "s_av_prem": "av0", "pcount": "sa",
         "cl_mat_pp": "cl_death_pp", "cl_death_pp": "cl_mat_pp", "cfa_mat": "cfa_death", "csv": "av_mid", "sa": "av_mid"}

def extract_tables(P):
    """Read the assumption tables out of the original model."""
    mt = P.mort_table
    return {"mort": mt.to_numpy(dtype=np.float64), "mort_age0": int(mt.index.min()),
            "mort_last_age": int(P.mort_table_last_age()), "sc": P.surr_charge_table.to_numpy(dtype=np.float64),
            "sc_cols": list(P.surr_charge_table.columns), "spec": P.product_spec_table.copy(),
            "disc_rate_ann": P.disc_rate_ann.copy(), "sim_param_table": P.sim_param_table.copy()}
def flags_of(tb, sim_id):
    return {k: bool(tb["sim_param_table"][k][sim_id]) for k in ("has_fees", "has_mortality", "has_lapse", "is_lapse_dynamic")}
def normals(n_scen):
    """The original's own random draw (its `std_norm_rand` cell), as an array [scenario, month]."""
    return np.random.default_rng(1234).standard_normal(n_scen * N_RAND_T).reshape(n_scen, N_RAND_T)

def prep(mp, n_scen, Z, tb, flags, dtype=np.float64, clip_T=False):
    """Turn a model point table, a number of scenarios, the normal draws and the simulation switches into
    the kernel's inputs: one row per (model point, scenario), model point major, as in the original.
    Every derived table uses the same expression as the original, so that the inputs are bit-identical."""
    ext = mp.join(tb["spec"], on="spec_id"); n_pts = len(ext)
    rep = np.repeat(np.arange(n_pts), n_scen)
    age0 = ext["age_at_entry"].to_numpy(np.int64); is_wl = ext["is_wl"].to_numpy(bool)
    term = np.where(is_wl, tb["mort_last_age"] - age0, ext["policy_term"].to_numpy(np.int64))
    dm0 = ext["duration_mth"].to_numpy(np.int64)
    T = int(np.max(np.maximum(12 * term - dm0 + 1, 0)))
    if clip_T: T = min(T, N_RAND_T)
    if T > N_RAND_T: raise ValueError(f"projection of {T} months exceeds the {N_RAND_T} months of random numbers the original draws")
    sc_col = ext["surr_charge_id"].map({c: i for i, c in enumerate(tb["sc_cols"])}).fillna(-1).to_numpy(np.int64)
    dur_min, dur_max = int(dm0.min() // 12), int((dm0.max() + T) // 12)
    d = np.arange(dur_min, dur_max + 1)
    mu, sigma, dt = 0.02, 0.03, 1 / 12
    INV = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dt ** 0.5 * Z) - 1
    dra = tb["disc_rate_ann"]
    disc_rate_mth = np.array(list(((1 + dra[t // 12]) ** (1 / 12) - 1 for t in range(T))))
    DISC = np.array(list(((1 + disc_rate_mth[t]) ** (-t) for t in range(T))))
    INFL = np.array([(1 + 0.01) ** (t / 12) for t in range(T)])
    mort = tb["mort"] if flags["has_mortality"] else np.zeros(tb["mort"].shape, dtype=np.int64)
    lapse = 1 * np.maximum(0.1 - 0.01 * d, 0.02) if flags["has_lapse"] else np.zeros(len(d), dtype=np.int64)
    f = lambda a: np.ascontiguousarray(a, dtype=dtype)
    row = lambda a, dt_: np.ascontiguousarray(np.asarray(a)[rep], dtype=dt_)
    return {"age0": row(age0, np.int64), "term": row(term, np.int64), "pcount": row(ext["policy_count"], dtype),
            "sa": row(ext["sum_assured"], dtype), "dm0": row(dm0, np.int64),
            "prem": row(ext["premium_pp"], dtype), "av0": row(ext["av_pp_init"], dtype),
            "is_single": row(ext["premium_type"] == "SINGLE", bool), "is_level": row(ext["premium_type"] == "LEVEL", bool),
            "has_sc": row(ext["has_surr_charge"], bool), "sc_col": row(sc_col, np.int64), "load": row(ext["load_prem_rate"], dtype),
            "scen": np.tile(np.arange(n_scen, dtype=np.int64), n_pts), "inv_ret": f(INV[:, :T]), "infl": f(INFL), "DISC": f(DISC), "T": T,
            "TAB_MORT": f(mort), "TAB_MORTMTH": f(1 - (1 - mort) ** (1 / 12)),
            "TAB_LAPSEMTH": f(1 - (1 - lapse) ** (1 / 12)),
            "TAB_SC": f(tb["sc"]), "DUR_MIN": dur_min, "AGE_MIN": tb["mort_age0"],
            "FEE_RATE": dtype(0.01 / 12 if flags["has_fees"] else 0), "HAS_LAPSE": bool(flags["has_lapse"]), "TWELFTH": dtype(1 / 12)}
SHARED_KEYS = ("T", "inv_ret", "infl", "DISC", "TAB_LAPSEMTH", "DUR_MIN")     # inputs that depend on the whole portfolio
def subset(p, idx):
    """The kernel inputs restricted to some rows (portfolio-wide inputs unchanged)."""
    n = len(p["age0"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n and k not in ("infl", "DISC") else v) for k, v in p.items()}

# ----------------------------------------------------------------- symbolic description
def symbolic_spec(flags, path=None):
    """The symbolic description for one setting of the simulation switches. The switches are concrete
    (the original branches on them with Python `if`), so the proof is run once per sim_id."""
    import z3
    from exactuary.symbolic import SymbolicSpec, Sym, Enum, smin, smax, lift, tval, POW
    I, R, B = z3.Int, z3.Real, z3.Bool
    MORT = z3.Function("MORT", z3.IntSort(), z3.IntSort(), z3.RealSort())     # mortality table lookup
    SC = z3.Function("SC", z3.IntSort(), z3.IntSort(), z3.RealSort())         # surrender charge lookup
    INVRET = z3.Function("INVRET", z3.IntSort(), z3.RealSort())               # scenario return in month t
    twelfth = z3.Q(1, 12)                # exactly one twelfth, as the original's `1/12` denotes
    K = kernel_for(flags)
    def setup(O):
        mp = {"age_at_entry": Sym(I("age_at_entry")), "policy_term": Sym(I("policy_term")), "policy_count": Sym(R("policy_count")),
              "sum_assured": Sym(R("sum_assured")), "duration_mth": Sym(I("duration_mth")), "premium_pp": Sym(R("premium_pp")),
              "av_pp_init": Sym(R("av_pp_init")), "has_surr_charge": Sym(B("has_surr_charge")), "is_wl": Sym(B("is_wl")),
              "load_prem_rate": Sym(R("load_prem_rate")), "_sc_col": Sym(I("surr_charge_col")),
              "premium_type": Enum({"SINGLE": Sym(B("is_single")), "LEVEL": Sym(B("is_level"))})}
        # Axioms: inputs (model point, simulation switches, random returns) and the pandas table lookups.
        O.axiom("model_point", lambda: mp)
        O.axiom("mort_table_last_age", lambda: Sym(I("mort_table_last_age")))
        for k in ("has_fees", "has_mortality", "has_lapse", "is_lapse_dynamic"):
            O.axiom(k, (lambda v: lambda: v)(bool(flags[k])))
        if flags["has_mortality"]:
            O.axiom("mort_rate", lambda t: Sym(MORT(lift(O("age", t)), lift(smin(O("duration", t), 5)))))
        else:       # the original returns pd.Series(0, ...), which cannot be executed symbolically
            O.axiom("mort_rate", lambda t: Sym(z3.IntVal(0)))
        if not flags["has_lapse"]:      # likewise pd.Series(0, ...); with lapse on, the formula is executed as written
            O.axiom("lapse_rate", lambda t: Sym(z3.IntVal(0)))
        O.axiom("surr_charge_rate", lambda t: Sym(z3.If(lift(mp["has_surr_charge"]),
                SC(lift(mp["_sc_col"]), lift(smin(O("duration", t), 10))), z3.RealVal(0))))
        O.axiom("inv_return_mth", lambda t: Sym(INVRET(lift(t))))
        return mp
    def kernel_args(O, mp, t):
        return {"age0": mp["age_at_entry"], "term": O("policy_term"), "pcount": mp["policy_count"], "sa": mp["sum_assured"],
                "dm0": mp["duration_mth"], "prem": mp["premium_pp"], "av0": mp["av_pp_init"],
                "is_single": mp["premium_type"] == "SINGLE", "is_level": mp["premium_type"] == "LEVEL",
                "has_sc": mp["has_surr_charge"], "sc_col": mp["_sc_col"], "load": mp["load_prem_rate"],
                "inv_ret": O("inv_return_mth", t), "infl": O("inflation_factor", t)}
    def carry(O, mp, t):
        return [O("av_pp_at", t, "BEF_INV") + O("inv_income_pp", t),
                O("pols_if_at", t, "BEF_DECR") - O("pols_lapse", t) - O("pols_death", t)]
    def invariants(O, mp, t):
        return [("duration_mth(t) = duration_mth(0) + t", O("duration_mth", t), mp["duration_mth"] + tval(t))]
    DUR_MIN = Sym(I("DUR_MIN")); age_min = Sym(I("AGE_MIN"))      # table origins: any integers
    def mort_at(i, j): return MORT(lift(i + age_min), lift(j)) if flags["has_mortality"] else z3.RealVal(0)
    def lapse_tab(i):       # the hoisted monthly lapse table, in terms of duration = i + DUR_MIN (static lapse only)
        lr = lift(smax(0.1 - 0.01 * Sym(z3.ToReal(lift(i + DUR_MIN))), 0.02)) if flags["has_lapse"] else z3.RealVal(0)
        return Sym(1 - POW(1 - lr, twelfth))
    tables = {"TAB_MORT": lambda i, j: Sym(mort_at(i, j)),
              "TAB_MORTMTH": lambda i, j: Sym(1 - POW(1 - mort_at(i, j), twelfth)),
              "TAB_LAPSEMTH": lapse_tab, "TAB_SC": lambda i, j: Sym(SC(lift(j), lift(i)))}
    from fractions import Fraction
    fee_rate = Fraction(1, 1200) if flags["has_fees"] else 0       # 0.01/12, exactly
    consts = {"DUR_MIN": DUR_MIN, "AGE_MIN": age_min, "FEE_RATE": fee_rate, "HAS_LAPSE": bool(flags["has_lapse"]), "TWELFTH": Fraction(1, 12)}
    def lemmas(step, mp):
        """Early exit: if no policies are in force and the policy has started, then next month every
        accumulated cashflow is zero and the same holds again."""
        t = z3.Int("t")
        a = {"age0": mp["age_at_entry"], "term": Sym(I("term")), "pcount": mp["policy_count"], "sa": mp["sum_assured"],
             "dm0": mp["duration_mth"], "prem": mp["premium_pp"], "av0": mp["av_pp_init"], "is_single": Sym(B("is_single")),
             "is_level": Sym(B("is_level")), "has_sc": mp["has_surr_charge"], "sc_col": mp["_sc_col"],
             "load": mp["load_prem_rate"], "inv_ret": Sym(R("inv_ret")), "infl": Sym(R("infl"))}
        r = step(Sym(t), False, *[a[k] for k in K.row_args + K.scen_series + K.time_series],
                 Sym(R("any_av")), 0.0, *[consts.get(c, c) for c in K.consts])
        assume = [t >= 1, lift(mp["duration_mth"]) + t >= 1]
        goals = [(n, r[K.ix(n)]) for n in ACCUMULATE + ["net_cf"]] + [("policies carried forward", r[K.n_out + 1])]
        return [(f"early exit: {n} is zero", assume, lift(g), z3.RealVal(0)) for n, g in goals]
    return SymbolicSpec(source_path=path or formulas_path(), setup=setup, kernel_args=kernel_args, carry=carry,
                        invariants=invariants, tables=tables, consts=consts, int_cells=("duration_mth",), lemmas=lemmas,
                        table_keys={**({} if flags["has_mortality"] else {"TAB_MORT": {0: None, 1: None}, "TAB_MORTMTH": {0: None, 1: None}}),
                                    **({} if flags["has_lapse"] else {"TAB_LAPSEMTH": {0: None}})})   # declared constant (all zeros) when switched off

COMPOSITE = {"claims[ALL]": ["claims[DEATH]", "claims[LAPSE]", "claims[MATURITY]"],
             "net_cf": ["premiums", "inv_income", "claims[ALL]", "expenses", "commissions", "av_change"]}
