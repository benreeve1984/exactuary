"""Worked example: lifelib `savings/CashValue_ME` (universal-life style savings product).
Everything specific to this model lives in this file: test portfolios, the step function,
input preparation, and the symbolic description used for the proofs."""
import os, time, numpy as np, pandas as pd
from exactuary import KernelSpec
from exactuary.oracle import cell_name, capture

# ----------------------------------------------------------------- original model (the oracle)
T_CELLS = ["age", "av_change", "coi", "coi_pp", "coi_rate", "commissions", "duration", "duration_mth",
           "expenses", "inv_income", "inv_income_pp", "inv_return_mth", "lapse_rate", "maint_fee", "maint_fee_pp",
           "margin_expense", "margin_mortality", "mort_rate", "mort_rate_mth", "net_amt_at_risk", "net_cf",
           "pols_death", "pols_lapse", "pols_maturity", "pols_new_biz", "prem_to_av", "prem_to_av_pp",
           "premium_pp", "premiums", "surr_charge", "surr_charge_rate"]
ARG_CELLS = ([("av_at", k) for k in ("BEF_MAT", "BEF_NB", "BEF_FEE")]
             + [("av_pp_at", k) for k in ("BEF_PREM", "BEF_FEE", "BEF_INV", "MID_MTH")]
             + [("pols_if_at", k) for k in ("BEF_MAT", "BEF_NB", "BEF_DECR")]
             + [("claim_pp", k) for k in ("DEATH", "LAPSE", "MATURITY")]
             + [("claims", k) for k in ("DEATH", "LAPSE", "MATURITY", None)]
             + [("claims_from_av", k) for k in ("DEATH", "LAPSE", "MATURITY")]
             + [("claims_over_av", k) for k in ("DEATH", "LAPSE", "MATURITY")])
OUTPUTS = [cell_name(c) for c in T_CELLS] + [cell_name(c, k) for c, k in ARG_CELLS]
PV_NAMES = ["Premiums", "Death", "Surrender", "Maturity", "Expenses", "Commissions",
            "Investment Income", "Change in AV", "Net Cashflow"]
ACCUMULATE = ["premiums", "claims[DEATH]", "claims[LAPSE]", "claims[MATURITY]", "expenses",
              "commissions", "inv_income", "av_change", "claims[ALL]"]

def source_path():
    import lifelib
    return os.path.join(os.path.dirname(lifelib.__file__), "libraries", "savings", "CashValue_ME")
def formulas_path(): return os.path.join(source_path(), "Projection", "__init__.py")
def load_original():
    import modelx as mx
    return mx.read_model(source_path())

def normals(n_scen, n_t=1300, seed=20261121):
    return np.random.default_rng(seed).standard_normal((n_scen, n_t))
def expand(mp, n_scen):
    n = len(mp); out = mp.iloc[np.repeat(np.arange(n), n_scen)].reset_index(drop=True); out.index.name = "row"
    return out, np.tile(np.arange(n_scen), n)
def portfolio_sample(m, n_points, n_scen):
    """lifelib's own sample model points, each under n_scen scenarios."""
    mp = m.Projection.model_point_10000.iloc[:n_points].copy(); mp["accum_prem_init_pp"] = 0
    return expand(mp, n_scen)
def portfolio_edge(n_points, n_scen, seed=7):
    """Random model points chosen to reach edge cases: in-force, future-dated and past-term
    policies, extreme sums assured, pre-funded account values."""
    r = np.random.default_rng(seed); n = n_points
    term = r.choice([1, 2, 5, 10, 15, 20, 30], n)
    dur = np.where(r.random(n) < 0.35, 0, r.integers(-30, 400, n))
    dur = np.where(r.random(n) < 0.15, term * 12 + r.integers(-2, 3, n), dur)
    mp = pd.DataFrame({"spec_id": r.choice(list("ABCD"), n), "age_at_entry": r.integers(18, 75, n),
                       "sex": r.choice(["M", "F"], n), "policy_term": term, "policy_count": r.integers(1, 200, n),
                       "sum_assured": r.choice([0, 1000, 50_000, 500_000, 5_000_000], n), "duration_mth": dur,
                       "premium_pp": r.choice([0, 100, 3100, 6700, 422_000, 900_000], n),
                       "av_pp_init": np.where(dur > 0, r.choice([0, 10, 40_000, 2_000_000], n), 0),
                       "accum_prem_init_pp": 0})
    return expand(mp, n_scen)
def configure(m, mp, scen, Z):
    P = m.Projection; P.model_point_table = mp; P.scen_id = scen; P.std_norm_rand = Z
    return P
def run_original(P, trace=True):
    t0 = time.perf_counter(); pv = P.result_pv(); sec = time.perf_counter() - t0
    T = P.max_proj_len()
    res = {"pv": pv[PV_NAMES].to_numpy(dtype=np.float64), "T": T, "seconds_pv": sec}
    if trace:
        res["trace"] = capture(P, OUTPUTS, T)
        res["inflation_factor"] = np.array([P.inflation_factor(t) for t in range(T)])
        res["disc_factors"] = np.asarray(P.disc_factors(), dtype=np.float64)
    return res
def net_cf_pv(acc):
    """The nine reported present values from the nine accumulated outputs."""
    return np.column_stack([acc[:, k] for k in range(8)] + [acc[:, 0] + acc[:, 6] - acc[:, 8] - acc[:, 4] - acc[:, 5] - acc[:, 7]])

# ----------------------------------------------------------------- the rewrite
STEP_SRC = '''
def step(t, is_t0, age0, term, pcount, sa, dm0, prem, av0, is_single, is_level, has_sc, sc_col, load,
         inv_ret, infl, s_av_prem, s_p_bef_mat, TAB_MORT, TAB_MORTMTH, TAB_LAPSEMTH, TAB_SC, DUR_MIN, AGE_MIN):
    dm = dm0 + t
    dur = idiv(dm, 12)
    age = age0 + dur
    term12 = term * 12
    av_prem = sel(is_t0, av0, s_av_prem)
    p_bef_mat = sel(is_t0, sel(dm0 > 0, pcount, ZERO), s_p_bef_mat)
    p_mat = sel(dm == term12, p_bef_mat, ZERO)
    p_bef_nb = p_bef_mat - p_mat
    p_nb = sel(dm == 0, pcount, ZERO)
    p_decr = p_bef_nb + p_nb
    prem_pp = sel(band(is_single, dm == 0), prem, ZERO) + sel(band(is_level, dm < term12), prem, ZERO)
    prem_av_pp = (1 - load) * prem_pp
    av_fee = av_prem + prem_av_pp
    mfee_pp = (0.01 / 12) * av_fee
    nar = fmax(sa - av_fee, ZERO)
    d5 = imin(dur, 5)
    mort = lk2(TAB_MORT, age - AGE_MIN, d5)
    q = lk2(TAB_MORTMTH, age - AGE_MIN, d5)
    coi_rate = 1.1 * q
    coi_pp = coi_rate * nar
    av_inv = av_fee - mfee_pp - coi_pp
    iipp = inv_ret * av_inv
    av_mid = av_inv + 0.5 * iipp
    p_death = p_decr * q
    lapse_rate = fmax(0.1 - 0.02 * tof(dur), 0.02)
    lq = lk1(TAB_LAPSEMTH, dur - DUR_MIN)
    p_lapse = (p_decr - p_death) * lq
    p_next = p_decr - p_lapse - p_death
    av_next = av_inv + iipp
    premiums = prem_pp * p_decr
    cl_death_pp = fmax(sa, av_mid)
    cl_death = cl_death_pp * p_death
    cfa_death = av_mid * p_death
    cfa_lapse = av_mid * p_lapse
    sc_rate = sel(has_sc, lk2(TAB_SC, imin(dur, 10), sc_col), ZERO)
    surr = sc_rate * av_mid * p_lapse
    cl_lapse = cfa_lapse - surr
    cl_mat = av_prem * p_mat
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
    return (tof(age), av_change, coi, coi_pp, coi_rate, commissions, tof(dur), tof(dm),
            expenses, inv_income, iipp, inv_ret, lapse_rate, mfee, mfee_pp,
            margin_exp, margin_mort, mort, q, nar, net_cf,
            p_death, p_lapse, p_mat, p_nb, prem_av_pp * p_decr, prem_av_pp,
            prem_pp, premiums, surr, sc_rate,
            av_bef_mat, av_prem * p_bef_nb, av_fee * p_decr,
            av_prem, av_fee, av_inv, av_mid,
            p_bef_mat, p_bef_nb, p_decr,
            cl_death_pp, av_mid, av_prem,
            cl_death, cl_lapse, cl_mat, cl_all,
            cfa_death, cfa_lapse, cl_mat,
            coa_death, cl_lapse - cfa_lapse, cl_mat - cl_mat,
            av_next, p_next)
'''
# The same step with the algebra rearranged. It is not bit-identical to the original; it is
# proved equal over the real numbers and its rounding is bounded.
_REARRANGE = [("av_mid = av_inv + 0.5 * iipp", "av_mid = av_inv * (1 + 0.5 * inv_ret)"),
              ("p_lapse = (p_decr - p_death) * lq", "p_surv = p_decr * (1 - q)\n    p_lapse = p_surv * lq"),
              ("p_next = p_decr - p_lapse - p_death", "p_next = p_surv - p_lapse"),
              ("av_next = av_inv + iipp", "av_next = av_inv * (1 + inv_ret)"),
              ("inv_income = iipp * p_next + 0.5 * iipp * (p_death + p_lapse)", "inv_income = iipp * (p_next + 0.5 * (p_death + p_lapse))")]
STEP_REARRANGED_SRC = STEP_SRC
for _a, _b in _REARRANGE:
    assert _a in STEP_REARRANGED_SRC; STEP_REARRANGED_SRC = STEP_REARRANGED_SRC.replace(_a, _b)

KERNEL = KernelSpec(
    step_src=STEP_SRC,
    row_args=["age0", "term", "pcount", "sa", "dm0", "prem", "av0", "is_single", "is_level", "has_sc", "sc_col", "load"],
    scen_series=["inv_ret"], time_series=["infl"], state=["s_av_prem", "s_p_bef_mat"],
    consts=["TAB_MORT", "TAB_MORTMTH", "TAB_LAPSEMTH", "TAB_SC", "DUR_MIN", "AGE_MIN"],
    outputs=OUTPUTS, accumulate=ACCUMULATE,
    exit_when="(OUT['pols_if_at[BEF_DECR]'] == ZERO) & (ROW['dm0'] + t >= 0)")
KERNEL_REARRANGED = KERNEL.with_src(STEP_REARRANGED_SRC)

# names a careless translator might confuse, for the wrong-variable seeded defects
SWAPS = {"av_prem": "av_fee", "av_fee": "av_inv", "av_inv": "av_mid", "av_mid": "av_inv", "p_bef_mat": "p_bef_nb",
         "p_bef_nb": "p_decr", "p_decr": "p_bef_mat", "p_death": "p_lapse", "p_lapse": "p_death", "q": "lq", "lq": "q",
         "prem_pp": "prem_av_pp", "cfa_death": "cfa_lapse", "cl_death": "cl_lapse", "dm": "dm0", "dur": "dm",
         "p_next": "p_decr", "av_next": "av_inv", "iipp": "av_inv", "s_av_prem": "av0", "pcount": "sa"}

def extract_tables(P):
    """Read the assumption tables out of the original model."""
    mt = P.mort_table
    return {"mort": mt.to_numpy(dtype=np.float64), "mort_age0": int(mt.index.min()),
            "mort_last_age": int(P.mort_table_last_age()), "sc": P.surr_charge_table.to_numpy(dtype=np.float64),
            "sc_cols": list(P.surr_charge_table.columns), "spec": P.product_spec_table.copy(),
            "disc_rate_ann": P.disc_rate_ann.copy()}

def prep(mp, scen, Z, tb, dtype=np.float64):
    """Turn a model point table, scenario ids and normal draws into the kernel's inputs. Every derived
    table uses the same expression as the original, so that the kernel's inputs are bit-identical."""
    ext = mp.join(tb["spec"], on="spec_id")
    age0 = ext["age_at_entry"].to_numpy(np.int64); is_wl = ext["is_wl"].to_numpy(bool)
    term = np.where(is_wl, tb["mort_last_age"] - age0, ext["policy_term"].to_numpy(np.int64))
    dm0 = ext["duration_mth"].to_numpy(np.int64)
    T = int(np.max(np.maximum(12 * term - dm0 + 1, 0)))
    sc_col = ext["surr_charge_id"].map({c: i for i, c in enumerate(tb["sc_cols"])}).fillna(-1).to_numpy(np.int64)
    dur_min, dur_max = int(dm0.min() // 12), int((dm0.max() + T) // 12)
    d = np.arange(dur_min, dur_max + 1)
    mu, sigma, dt = 0.02, 0.03, 1 / 12
    INV = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dt ** 0.5 * Z) - 1
    dra = tb["disc_rate_ann"]
    disc_rate_mth = np.array(list(((1 + dra[t // 12]) ** (1 / 12) - 1 for t in range(T))))
    DISC = np.array(list(((1 + disc_rate_mth[t]) ** (-t) for t in range(T))))
    INFL = np.array([(1 + 0.01) ** (t / 12) for t in range(T)])
    f = lambda a: np.ascontiguousarray(a, dtype=dtype)
    return {"age0": age0, "term": term, "pcount": f(ext["policy_count"]), "sa": f(ext["sum_assured"]), "dm0": dm0,
            "prem": f(ext["premium_pp"]), "av0": f(ext["av_pp_init"]),
            "is_single": (ext["premium_type"] == "SINGLE").to_numpy(bool),
            "is_level": (ext["premium_type"] == "LEVEL").to_numpy(bool),
            "has_sc": ext["has_surr_charge"].to_numpy(bool), "sc_col": sc_col, "load": f(ext["load_prem_rate"]),
            "scen": np.asarray(scen, np.int64), "inv_ret": f(INV[:, :T]), "infl": f(INFL), "DISC": f(DISC), "T": T,
            "TAB_MORT": f(tb["mort"]), "TAB_MORTMTH": f(1 - (1 - tb["mort"]) ** (1 / 12)),
            "TAB_LAPSEMTH": f(1 - (1 - np.maximum(0.1 - 0.02 * d, 0.02)) ** (1 / 12)),
            "TAB_SC": f(tb["sc"]), "DUR_MIN": dur_min, "AGE_MIN": tb["mort_age0"]}

# ----------------------------------------------------------------- symbolic description
def symbolic_spec(path=None):
    import z3
    from exactuary.symbolic import SymbolicSpec, Sym, Enum, smin, smax, lift, tval, POW
    I, R, B = z3.Int, z3.Real, z3.Bool
    MORT = z3.Function("MORT", z3.IntSort(), z3.IntSort(), z3.RealSort())     # mortality table lookup
    SC = z3.Function("SC", z3.IntSort(), z3.IntSort(), z3.RealSort())         # surrender charge lookup
    INVRET = z3.Function("INVRET", z3.IntSort(), z3.RealSort())               # scenario return in month t
    twelfth = z3.Q(1, 12)                # exactly one twelfth, as the original's `1/12` denotes
    def setup(O):
        mp = {"age_at_entry": Sym(I("age_at_entry")), "policy_term": Sym(I("policy_term")), "policy_count": Sym(R("policy_count")),
              "sum_assured": Sym(R("sum_assured")), "duration_mth": Sym(I("duration_mth")), "premium_pp": Sym(R("premium_pp")),
              "av_pp_init": Sym(R("av_pp_init")), "has_surr_charge": Sym(B("has_surr_charge")), "is_wl": Sym(B("is_wl")),
              "load_prem_rate": Sym(R("load_prem_rate")), "_sc_col": Sym(I("surr_charge_col")),
              "premium_type": Enum({"SINGLE": Sym(B("is_single")), "LEVEL": Sym(B("is_level"))})}
        # Axioms: inputs, and the two pandas table lookups (checked exhaustively by enumeration instead).
        O.axiom("model_point", lambda: mp)
        O.axiom("mort_table_last_age", lambda: Sym(I("mort_table_last_age")))
        O.axiom("mort_rate", lambda t: Sym(MORT(lift(O("age", t)), lift(smin(O("duration", t), 5)))))
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
    DUR_MIN = Sym(I("DUR_MIN"))
    def lapse_tab(i):       # how the hoisted monthly lapse table is defined, in terms of duration = i + DUR_MIN
        lr = smax(0.1 - 0.02 * Sym(z3.ToReal(lift(i + DUR_MIN))), 0.02)
        return Sym(1 - POW(1 - lift(lr), twelfth))
    tables = {"TAB_MORT": lambda i, j: Sym(MORT(lift(i + 18), lift(j))),
              "TAB_MORTMTH": lambda i, j: Sym(1 - POW(1 - MORT(lift(i + 18), lift(j)), twelfth)),
              "TAB_LAPSEMTH": lapse_tab, "TAB_SC": lambda i, j: Sym(SC(lift(j), lift(i)))}
    def lemmas(step, mp):
        """Early exit: if no policies are in force and the policy has started, then next month every
        accumulated cashflow is zero and the same holds again."""
        t = z3.Int("t")
        a = {"age0": mp["age_at_entry"], "term": Sym(I("term")), "pcount": mp["policy_count"], "sa": mp["sum_assured"],
             "dm0": mp["duration_mth"], "prem": mp["premium_pp"], "av0": mp["av_pp_init"], "is_single": Sym(B("is_single")),
             "is_level": Sym(B("is_level")), "has_sc": mp["has_surr_charge"], "sc_col": mp["_sc_col"],
             "load": mp["load_prem_rate"], "inv_ret": Sym(R("inv_ret")), "infl": Sym(R("infl"))}
        r = step(Sym(t), False, *[a[k] for k in KERNEL.row_args + KERNEL.scen_series + KERNEL.time_series],
                 Sym(R("any_av")), 0.0, "TAB_MORT", "TAB_MORTMTH", "TAB_LAPSEMTH", "TAB_SC", DUR_MIN, 18)
        assume = [t >= 1, lift(mp["duration_mth"]) + t >= 1]
        goals = [(n, r[KERNEL.ix(n)]) for n in ACCUMULATE + ["pols_if_at[BEF_DECR]", "net_cf"]] + [("policies carried forward", r[KERNEL.n_out + 1])]
        return [(f"early exit: {n} is zero", assume, lift(g), z3.RealVal(0)) for n, g in goals]
    return SymbolicSpec(source_path=path or formulas_path(), setup=setup, kernel_args=kernel_args, carry=carry,
                        invariants=invariants, tables=tables, consts={"DUR_MIN": DUR_MIN, "AGE_MIN": 18},
                        int_cells=("duration_mth",), lemmas=lemmas)

COMPOSITE = {"claims[ALL]": ["claims[DEATH]", "claims[LAPSE]", "claims[MATURITY]"],
             "net_cf": ["premiums", "inv_income", "claims[ALL]", "expenses", "commissions", "av_change"]}
