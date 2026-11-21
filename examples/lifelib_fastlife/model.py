"""lifelib `fastlife` (annual-step traditional life model: term, whole life, endowment; several modelx spaces).
Everything specific to this model lives in this file: test portfolios, the oracle capture, the step
function, input preparation, and the symbolic description used for the proofs.

Scope: every cell of the `Projection` space that the original can evaluate (policies in force, decrements,
premiums, benefits, expenses, commissions, accumulated cashflow), the per-year `Policy` cells they use
(net-level-premium reserve rate on the premium basis, surrender charge, cash value rate), the per-year
`Assumptions` cells (mortality factor, surrender rate) and the four working present-value cells.
Out of scope: the valuation-basis reserve and everything downstream of it, and the ten PV cells that
raise in the original (see REPORT.md)."""
import os, time, dataclasses, numpy as np, pandas as pd
from itertools import combinations
from exactuary import KernelSpec
from exactuary.oracle import parse_cell

# ----------------------------------------------------------------- original model (the oracle)
ZERO_CELLS = ["BenefitAccDth", "BenefitAccHosp", "BenefitAnn", "BenefitLiving", "BenefitMat", "BenefitOther",
              "BenefitSickHosp", "BenefitSurg", "ExpsOther", "PolsAccDeath", "PolsAccHosp", "PolsAnnuity",
              "PolsLiving", "PolsOther", "PolsRenewal", "PolsSickHosp", "PolsSurg", "ReserveHospRsrvEnd",
              "ReserveUernPremEnd", "SizeBenefitAccDth", "SizeBenefitAccHosp", "SizeBenefitAnn", "SizeBenefitLiving",
              "SizeBenefitMat", "SizeBenefitOther", "SizeBenefitSickHosp", "SizeBenefitSurg", "SizeExpsOther",
              "SizeReserveUernPremAftMat", "SizeReserveUernPremEnd"]
# output name -> expression in the step. Names without a prefix are cells of the Projection space;
# "pol." / "asmp." are cells of its child spaces; "pvflow_" are the three terms the original discounts.
OUTPUT_EXPR = {
    "AccumCF": "accum", "AttAge": "tof(age)", "BaseMortRate": "q", "BenefitDeath": "b_death", "BenefitSurr": "b_surr",
    "BenefitTotal": "b_total", "DiscRate": "disc_rate", "ExpsAcq": "e_acq", "ExpsAcqTotal": "e_comm + e_acq",
    "ExpsCommInit": "e_ci", "ExpsCommRen": "e_cr", "ExpsCommTotal": "e_comm", "ExpsMaint": "e_maint",
    "ExpsMaintTotal": "e_maint", "ExpsTotal": "e_total", "InflFactor": "infl", "InsurIF_Beg1": "p_beg1 * sa",
    "InsurIF_End": "p_end * sa", "IntAccumCF": "int_acc", "InvstRetRate": "disc_rate", "NetInsurCF": "net_cf",
    "PolsDeath": "p_death", "PolsIF_AftMat": "p_aft", "PolsIF_Beg": "p_aft", "PolsIF_Beg1": "p_beg1",
    "PolsIF_End": "p_end", "PolsMaturity": "p_mat", "PolsNewBiz": "p_nb", "PolsSurr": "p_surr", "PremIncome": "prem",
    "SizeAnnPrem": "sap", "SizeBenefitDeath": "sa", "SizeBenefitSurr": "sbs", "SizeExpsAcq": "s_acq",
    "SizeExpsCommInit": "s_ci", "SizeExpsCommRen": "s_cr", "SizeExpsMaint": "s_maint", "SizePremium": "sp",
    "SizeSumAssured": "sa",
    "pol.ReserveNLP_Rate[PREM]": "rsv0", "pol.SurrCharge": "sc0", "pol.CashValueRate": "cv0",
    "asmp.MortFactor": "mf", "asmp.SurrRate": "sr",
    "pvflow_prem": "sel(exist, prem, ZERO)", "pvflow_exps": "sel(exist, e_total, ZERO)", "pvflow_ben": "b_total * vdisc",
    **{c: "ZERO" for c in ZERO_CELLS}}
OUTPUTS = list(OUTPUT_EXPR)
ACCUMULATE = ["pvflow_prem", "pvflow_exps", "pvflow_ben"]
PV_CELLS = ["PV_PremIncome", "PV_ExpsTotal", "PV_BenefitTotal", "PV_NetCashflow"]
# Projection cells that raise in the original for any portfolio of more than one policy, and are therefore not converted
BROKEN = {"reserve (KeyError 'NetPremRate_VAL')": ["ChangeRsrv", "IncomeTotal", "InvstIncome", "ProfitBefTax",
          "ReservePremRsrvEnd", "ReserveTotal_End", "SizeInvstIncome", "SizeReservePremRsrvAftMat",
          "SizeReservePremRsrvEnd", "SizeReserveTotalAftMat"],
          "per-benefit PV (truth value of a Series)": ["InterestNetCF", "PV_BenefitDeath", "PV_BenefitMat",
          "PV_BenefitSurr", "PV_Check", "PV_ExpsAcq", "PV_ExpsCommTotal", "PV_ExpsMaint", "PV_NetCashflowForCheck",
          "PV_SumInsurIF"]}
# the three discounted terms, as the original's PV cells write them (they are sub-expressions, not cells)
DERIVED = {"pvflow_prem": lambda P, t: (t <= P.last_t()) * P.PremIncome(t),
           "pvflow_exps": lambda P, t: (t <= P.last_t()) * P.ExpsTotal(t),
           "pvflow_ben": lambda P, t: P.BenefitTotal(t) * (1 / (1 + P.DiscRate(t)))}
ATTR_CELLS = {"gpr": ("Policy", "GrossPremRate"), "netp": ("Policy", "NetPremRate", "PREM"), "apr": ("Policy", "AnnPremRate"),
              "isc": ("Policy", "InitSurrCharge"), "g2": ("Policy", "LoadMaintSA2"), "last_t": (None, "last_t"),
              "cip": ("Assumptions", "CommInitPrem"), "crp": ("Assumptions", "CommRenPrem"), "crt": ("Assumptions", "CommRenTerm"),
              "acq_ap": ("Assumptions", "ExpsAcqAnnPrem"), "acq_sa": ("Assumptions", "ExpsAcqSA"), "acq_pol": ("Assumptions", "ExpsAcqPol"),
              "mnt_ap": ("Assumptions", "ExpsMaintAnnPrem"), "mnt_sa": ("Assumptions", "ExpsMaintSA"), "mnt_pol": ("Assumptions", "ExpsMaintPol")}

def lib_path(lib="fastlife"):
    import lifelib
    return os.path.join(os.path.dirname(lifelib.__file__), "libraries", lib, "model")
def formulas_path(space="Projection"): return os.path.join(lib_path(), *space.split("."), "__init__.py")
def load_original():
    import modelx as mx
    return mx.read_model(lib_path())

class Data:
    """Stands in for the model's PandasData object: called, it returns the model point table."""
    def __init__(self, value): self.value = value
    def __call__(self): return self.value

def sample_table(m): return m.Projection.Policy.PolicyData().copy()
def portfolio_sample(m, reps=1):
    """lifelib's own 300 model points, repeated `reps` times."""
    df = sample_table(m)
    out = pd.concat([df] * reps, ignore_index=True); out.index = np.arange(1, len(out) + 1); out.index.name = "Policy"
    return out
VALID = [("TERM", 1, 1), ("TERM", 1, 2), ("WL", 1, 1), ("ENDW", 1, 1), ("ENDW", 1, 2), ("ENDW", 2, 1), ("ENDW", 2, 2)]
LAST_AGE = {"M": 113, "F": 117}      # table 3, the only table the product specs select
def _frame(prod, polt, gen, sex, age, term, freq, count, sa):
    n = len(age)
    df = pd.DataFrame({"Product": prod, "PolicyType": np.asarray(polt, np.int64), "Gen": np.asarray(gen, np.int64),
                       "Channel": np.nan, "Duration": 0, "Sex": sex, "IssueAge": np.asarray(age, np.int64), "PaymentMode": 1,
                       "PremFreq": np.asarray(freq, np.int64), "PolicyTerm": np.asarray(term, np.int64),
                       "MaxPolicyTerm": np.asarray(term, np.int64), "PolicyCount": np.asarray(count, np.int64),
                       "SumAssured": np.asarray(sa, np.int64)}, index=pd.RangeIndex(1, n + 1, name="Policy"))
    return df
def portfolio_edge(n=1500, seed=7):
    """Random model points chosen to reach unusual cases: every product / policy type / generation the
    specification tables accept, issue ages from 0 to the last age of the mortality table, terms from one
    year to the end of the commutation columns (so that policies outlive the mortality table), terms at
    and around the commission and surrender-charge breakpoints, zero and extreme counts and sums assured."""
    r = np.random.default_rng(seed)
    k = r.integers(0, len(VALID), n); prod = np.array([VALID[i][0] for i in k], dtype=object)
    polt = np.array([VALID[i][1] for i in k]); gen = np.array([VALID[i][2] for i in k])
    sex = r.choice(["M", "F"], n); last = np.where(sex == "M", LAST_AGE["M"], LAST_AGE["F"])
    u = r.random(n)
    age = np.where(u < 0.6, r.integers(0, 91, n), np.where(u < 0.85, r.integers(90, 118, n), last - r.integers(0, 4, n)))
    age = np.minimum(age, last)
    cap = 131 - age
    v = r.random(n)
    term = np.where(v < 0.25, r.integers(1, 7, n), np.where(v < 0.45, r.integers(8, 13, n),
            np.where(v < 0.7, r.integers(1, 60, n), np.where(v < 0.85, cap, last - age + r.integers(-2, 3, n)))))
    term = np.clip(term, 1, cap)
    return _frame(prod, polt, gen, sex, age, term, r.choice([1, 2, 4, 12], n), r.choice([0, 1, 3, 250], n, p=[.05, .5, .25, .2]),
                  r.choice([0, 1, 1000, 1_000_000, 50_000_000], n, p=[.05, .05, .2, .5, .2]))
def portfolio_tables():
    """Synthetic portfolio that enumerates the index space of every lookup: each accepted product key,
    both sexes, every issue age up to the table's last age, shortest, mid and longest term."""
    rows = [(p, pt, g, s, x, n) for p, pt, g in VALID for s in "MF" for x in range(0, LAST_AGE[s] + 1)
            for n in sorted({1, min(12, 131 - x), 131 - x})]
    c = list(zip(*rows)); n = len(rows)
    return _frame(np.array(c[0], dtype=object), c[1], c[2], np.array(c[3], dtype=object), c[4], c[5], [12] * n, [1] * n, [1_000_000] * n)

def edge_inputs(m):
    """Assumptions for the edge runs. lifelib's sample assumptions are flat (one mortality factor, one lapse
    rate, one interest rate, no tax, no inflation), which would hide a wrong index into any of them.
    The formulas are untouched; only input data objects are replaced by varied copies."""
    P = m.Projection; A = P.Assumptions
    asmp = dict(A.Assumption); tabs = dict(A.AssumptionTables); scen = dict(m.Economic.Scenarios)
    asmp[("MortFactor", "WL", None, None)] = "MortAsmp1"
    asmp[("CnsmpTax", None, None, None)] = 0.1
    asmp[("InflRate", None, None, None)] = 0.02
    asmp[("ExpsMaintSA", None, None, None)] = 0.0001
    asmp[("CommRenTerm", "ENDW", None, None)] = 8
    asmp[("BaseMort", "ENDW", 1, None)] = 1        # projection mortality table that ends before the premium table does
    for y in range(20): tabs[("LapseRate1", y)] = 0.03 + 0.005 * y
    for t in range(151): scen[(11, "IntRate", t)] = 0.005 + 0.001 * (t % 30)
    A.Assumption = asmp; A.AssumptionTables = tabs; m.Economic.Scenarios = scen
    P.ScenID = 11
    return 11
def configure(m, table):
    d = Data(table); m.Projection.Policy.PolicyData = d; m.Projection.Assumptions.PolicyData = d
    return m.Projection

def cell_value(P, name, t):
    if name in DERIVED: return DERIVED[name](P, t)
    if name.startswith("pol."):
        c = parse_cell(name[4:]); return getattr(P.Policy, c[0])(*c[1:], t)      # e.g. ReserveNLP_Rate('PREM', t)
    if name.startswith("asmp."): return getattr(P.Assumptions, name[5:])(t)
    return getattr(P, name)(t)
def capture(P, names, T, n):
    """As exactuary.oracle.capture, for a model whose cells live in several spaces, take their arguments
    in a different order, and return a scalar (not a Series) when a value is the same for every policy."""
    return {nm: np.array([np.broadcast_to(np.asarray(cell_value(P, nm, t), dtype=np.float64), (n,)) for t in range(T)])
            for nm in names}
def run_original(P, trace=True):
    n = len(P.Policy.PolicyData())
    t0 = time.perf_counter(); pvn = P.PV_NetCashflow(0); sec = time.perf_counter() - t0
    T = int(P.last_t().max()) + 1
    res = {"T": T, "seconds_pv": sec, "pv": np.column_stack([np.asarray(getattr(P, c)(0), dtype=np.float64) for c in PV_CELLS])}
    if trace:
        res["trace"] = capture(P, OUTPUTS, T, n)
        res["pv_t"] = {c: np.array([np.broadcast_to(np.asarray(getattr(P, c)(t), dtype=np.float64), (n,)) for t in range(T + 1)])
                       for c in PV_CELLS}
        res["attrs"] = {k: np.asarray(getattr(P if sp is None else getattr(P, sp), c)(*a)) for k, (sp, c, *a) in ATTR_CELLS.items()}
    return res

def pv_backward(tr, K, p):
    """The four present-value cells for every year, computed from a kernel trace with the original's own
    backward recursion (same operations, same order). Returns {cell: array [year 0..T, row]}."""
    T = p["T"]; n = tr.shape[2]; r = p["disc_rate"]
    prem, exps, ben = tr[K.ix("pvflow_prem")], tr[K.ix("pvflow_exps")], tr[K.ix("BenefitTotal")]
    P_, E_, B_ = (np.zeros((T + 1, n)) for _ in range(3))
    for t in range(T - 1, -1, -1):
        P_[t] = prem[t] + P_[t + 1] / (1 + r[t])
        E_[t] = -exps[t] + E_[t + 1] / (1 + r[t])
        B_[t] = (-ben[t] + B_[t + 1]) / (1 + r[t])
    return {"PV_PremIncome": P_, "PV_ExpsTotal": E_, "PV_BenefitTotal": B_, "PV_NetCashflow": P_ + E_ + B_}
def pv_from_acc(acc):
    """The four reported present values from the three accumulated (running-sum) outputs."""
    return np.column_stack([acc[:, 0], -acc[:, 1], -acc[:, 2], acc[:, 0] - acc[:, 1] - acc[:, 2]])

# ----------------------------------------------------------------- the rewrite
# `pk` is the policy's product key: the row of every table that depends on (product, policy type, generation, sex).
# (The compiled backend accepts at most 30 step parameters, so values that depend only on the key are looked up
# in TAB_KF rather than passed one array each.)
ROW_ARGS = ["age0", "term", "last_t", "dz", "crt", "pk", "pcount", "sa", "gpr", "freq", "apr", "isc", "netp"]
KF_COLS = ["cip", "crp", "acq_ap", "acq_sa", "acq_pol", "mnt_ap", "mnt_sa", "mnt_pol", "g2"]
TIME_SERIES = ["disc_rate", "vdisc", "infl"]
STATE = ["s_pols_end", "s_accum"]
CONSTS = ["TAB_Q", "TAB_MF", "TAB_SR", "TAB_M", "TAB_N", "TAB_D", "TAB_KF", "TAX"]
_BODY = '''
    age = age0 + t
    age1 = age + 1
    agen = age0 + term
    exist = t <= last_t
    q = sel(exist, lk2(TAB_Q, pk, age), ZERO)
    mf = lk2(TAB_MF, pk, t)
    sr = lk2(TAB_SR, pk, t)
    cip = lk2(TAB_KF, pk, 0)
    crp = lk2(TAB_KF, pk, 1)
    acq_ap = lk2(TAB_KF, pk, 2)
    acq_sa = lk2(TAB_KF, pk, 3)
    acq_pol = lk2(TAB_KF, pk, 4)
    mnt_ap = lk2(TAB_KF, pk, 5)
    mnt_sa = lk2(TAB_KF, pk, 6)
    mnt_pol = lk2(TAB_KF, pk, 7)
    g2 = lk2(TAB_KF, pk, 8)
    m_n = lk2(TAB_M, pk, agen)
    n_n = lk2(TAB_N, pk, agen)
    ok0 = age < dz
    dd0 = sel(ok0, lk2(TAB_D, pk, age), ONE)
    axn0 = sel(ok0, (lk2(TAB_M, pk, age) - m_n) / dd0, ZERO)
    w0 = sel(ok0, (n_n - n_n) / dd0, ZERO)
    ann0 = sel(ok0, (lk2(TAB_N, pk, age) - n_n) / dd0, ZERO)
    rsv0 = sel(t <= term, axn0 + (g2 * w0 - netp * ann0), ZERO)
    ok1 = age1 < dz
    dd1 = sel(ok1, lk2(TAB_D, pk, age1), ONE)
    axn1 = sel(ok1, (lk2(TAB_M, pk, age1) - m_n) / dd1, ZERO)
    w1 = sel(ok1, (n_n - n_n) / dd1, ZERO)
    ann1 = sel(ok1, (lk2(TAB_N, pk, age1) - n_n) / dd1, ZERO)
    rsv1 = sel(t + 1 <= term, axn1 + (g2 * w1 - netp * ann1), ZERO)
    m10 = imin(term, 10)
    sc0 = isc * fmax(tof(m10 - t) / tof(m10), ZERO)
    sc1 = isc * fmax(tof(m10 - (t + 1)) / tof(m10), ZERO)
    cv0 = fmax(rsv0 - sc0, ZERO)
    cv1 = fmax(rsv1 - sc1, ZERO)
    sp = sa * gpr * freq
    sap = sa * apr
    sbs = sa * (cv0 + cv1) / 2.0
    s_acq = sel(is_t0, sap * acq_ap + (sa * acq_sa + acq_pol) * infl / infl, ZERO)
    s_ci = sel(is_t0, sp * cip * (1 + TAX), ZERO)
    s_cr = sel(is_t0, ZERO, sel(t < crt, sp, ZERO) * crp * (1 + TAX))
    s_maint = sap * mnt_ap + (sa * mnt_sa + mnt_pol) * infl
    p_end = sel(is_t0, ZERO, s_pols_end)
    p_mat = sel(term == t, p_end, ZERO)
    p_aft = p_end - p_mat
    p_nb = sel(is_t0, pcount, ZERO)
    p_beg1 = p_aft + p_nb
    p_death = p_beg1 * q * mf
    p_surr = p_beg1 * sr
    prem = sp * p_beg1
    b_death = sa * p_death
    b_surr = sbs * p_surr
    b_total = b_death + b_surr
    e_acq = s_acq * p_nb
    e_ci = s_ci * p_beg1
    e_cr = s_cr * p_beg1
    e_comm = e_ci + e_cr
    e_maint = s_maint * p_beg1
    e_total = e_comm + e_acq + e_maint
    accum = sel(is_t0, ZERO, s_accum)
    int_acc = (accum + prem - e_total) * disc_rate
    net_cf = prem - b_total - e_total
'''
def _step_src(body, consts):
    return (f"def step(t, is_t0, {', '.join(ROW_ARGS + TIME_SERIES + STATE + consts)}):" + body
            + "    return (" + ", ".join(OUTPUT_EXPR[n] for n in OUTPUTS)
            + ",\n            p_beg1 - p_death - p_surr, accum + int_acc + net_cf)\n")
STEP_SRC = _step_src(_BODY, CONSTS)
EXIT_WHEN = "OUT['PolsIF_Beg1'] == ZERO"
KERNEL = KernelSpec(step_src=STEP_SRC, row_args=ROW_ARGS, scen_series=[], time_series=TIME_SERIES, state=STATE, consts=CONSTS,
                    outputs=OUTPUTS, accumulate=ACCUMULATE, exit_when=EXIT_WHEN)

# The same step with every division by a commutation value D replaced by a multiplication by its hoisted
# reciprocal. It is not bit-identical to the original; it is proved equal over the real numbers, and
# (unlike the reference step) the rounding analysis can bound it in single precision.
_REARRANGE = [(f"dd{k} = sel(ok{k}, lk2(TAB_D, pk, {a}), ONE)", f"dd{k} = lk2(TAB_RD, pk, {a})") for k, a in ((0, "age"), (1, "age1"))] \
           + [(f" / dd{k}", f" * dd{k}") for k in (0, 1)]
_body_r = _BODY
for _a, _b in _REARRANGE:
    assert _a in _body_r; _body_r = _body_r.replace(_a, _b)
CONSTS_R = [c if c != "TAB_D" else "TAB_RD" for c in CONSTS]
STEP_REARRANGED_SRC = _step_src(_body_r, CONSTS_R)
KERNEL_REARRANGED = dataclasses.replace(KERNEL, step_src=STEP_REARRANGED_SRC, consts=CONSTS_R)

# names a careless translator might confuse, for the wrong-variable seeded defects
SWAPS = {"age": "age1", "age1": "age", "rsv0": "rsv1", "cv0": "cv1", "cv1": "cv0", "sc0": "sc1", "sp": "sap", "sap": "sp",
         "p_end": "p_aft", "p_aft": "p_beg1", "p_beg1": "p_end", "p_death": "p_surr", "p_surr": "p_death", "mf": "sr",
         "sr": "mf", "e_ci": "e_cr", "e_cr": "e_ci", "e_acq": "e_maint", "b_death": "b_surr", "m_n": "n_n", "n_n": "m_n",
         "acq_sa": "mnt_sa", "acq_pol": "mnt_pol", "mnt_ap": "acq_ap", "cip": "crp", "term": "last_t", "last_t": "term",
         "accum": "s_accum", "p_mat": "p_end", "gpr": "apr", "ok0": "ok1", "dd1": "dd0", "ann0": "axn0", "axn1": "ann1",
         "disc_rate": "vdisc", "prem": "sp", "s_pols_end": "pcount"}

N_AGE = 132          # commutation columns cover ages 0..131 (the original cannot go further: qx stops at 130)
def extract_tables(m):
    """Read the input data out of the original model (after any edge_inputs)."""
    P = m.Projection; mt = m.Input.MortalityTables()
    assert list(mt.index) == list(range(len(mt)))
    return {"spec": dict(P.Policy.ProductSpec), "asmp": dict(P.Assumptions.Assumption),
            "asmp_tables": dict(P.Assumptions.AssumptionTables), "scen": dict(m.Economic.Scenarios),
            "mort": mt.to_numpy(dtype=np.float64), "mort_cols": list(mt.columns), "scen_id": int(P.ScenID)}

def lookup(table, name, prod=None, polt=None, gen=None):
    """The original's SpecLookup / AsmpLookup: most specific key first, wildcards (None) allowed."""
    key = [prod, polt, gen]
    for match_len in range(3, -1, -1):
        for idxs in combinations(range(3), match_len):
            masked = [None] * 3
            for idx in idxs: masked[idx] = key[idx]
            value = table.get((name, *masked))
            if value is not None: return value
    return None

def commutation(tb, sex, rate, table_id):
    """Columns D, M, N of the original's LifeTable[sex, rate, table_id], ages 0..131, with the same
    scalar operations in the same order (including Python's float ** int for the discount powers)."""
    q = tb["mort"][:, tb["mort_cols"].index((table_id, sex))]
    l = [100000]
    for x in range(1, N_AGE): l.append(l[x - 1] - l[x - 1] * q[x - 1])
    disc = 1 / (1 + rate)
    D = [l[x] * disc ** x for x in range(N_AGE)]
    C = [l[x] * q[x] * disc ** (x + 1 / 2) for x in range(N_AGE - 1)]
    M, N = [None] * N_AGE, [None] * N_AGE
    for x in range(N_AGE - 1, -1, -1):
        M[x] = D[x] if x >= 110 else M[x + 1] + C[x]
        N[x] = D[x] if x >= 110 else N[x + 1] + D[x]
    return np.array(D, np.float64), np.array(M, np.float64), np.array(N, np.float64)

def prep(table, tb, dtype=np.float64):
    """Turn a model point table and the model's input data into the kernel's inputs. Everything derived
    here (specification and assumption lookups, commutation columns, premium rates) uses the original's
    expressions in the original's order, so that the kernel's inputs are bit-identical."""
    spec, asmp = tb["spec"], tb["asmp"]
    prod = table["Product"].to_numpy(); polt = table["PolicyType"].to_numpy(); gen = table["Gen"].to_numpy()
    sex = table["Sex"].to_numpy(); x = table["IssueAge"].to_numpy(np.int64); n = table["PolicyTerm"].to_numpy(np.int64)
    freq = table["PremFreq"].to_numpy(np.int64)
    if (n < 1).any() or (freq < 1).any(): raise ValueError("policy term and premium frequency must be at least 1")
    if (x + n >= N_AGE).any(): raise KeyError("issue age + term beyond the mortality table (the original raises too)")
    keys = pd.MultiIndex.from_arrays([prod, polt, gen, sex]); code, uniq = pd.factorize(keys)
    # --- per product key: specification and assumption lookups
    cols = {k: [] for k in ("a1", "a2", "lmp1", "lmp2", "gamma", "g2", "rate", "tid", "sc1", "sc2", "cip", "crp", "crt",
                            "acq_ap", "acq_sa", "acq_pol", "mnt_ap", "mnt_sa", "mnt_pol", "mort_id", "mf_name", "sr_name")}
    def need(v, what):
        if v is None: raise ValueError(f"{what} not found")
        return v
    for p, pt, g, s in uniq:
        pt, g = int(pt), int(g)
        cols["a1"].append(need(lookup(spec, "LoadAcqSAParam1", p), "LoadAcqSAParam1"))
        cols["a2"].append(need(lookup(spec, "LoadAcqSAParam2", p), "LoadAcqSAParam2"))
        cols["lmp1"].append(lookup(spec, "LoadMaintPremParam1", p)); cols["lmp2"].append(lookup(spec, "LoadMaintPremParam2", p))
        cols["gamma"].append(need(lookup(spec, "LoadMaintSA", p, pt, g), "LoadMaintSA"))
        cols["g2"].append(need(lookup(spec, "LoadMaintSA2", p, pt, g), "LoadMaintSA2"))
        cols["rate"].append(need(lookup(spec, "IntRatePrem", p, pt, g), "IntRatePrem"))
        cols["tid"].append(need(lookup(spec, "MortTablePrem", p, pt, g), "MortTablePrem"))
        cols["sc1"].append(need(lookup(spec, "SurrChargeParam1", p, pt, g), "SurrChargeParam"))
        cols["sc2"].append(need(lookup(spec, "SurrChargeParam2", p, pt, g), "SurrChargeParam"))
        for k, nm in (("cip", "CommInitPrem"), ("crp", "CommRenPrem"), ("crt", "CommRenTerm"), ("acq_ap", "ExpsAcqAnnPrem"),
                      ("acq_sa", "ExpsAcqSA"), ("acq_pol", "ExpsAcqPol"), ("mnt_ap", "ExpsMaintPrem"), ("mnt_sa", "ExpsMaintSA"),
                      ("mnt_pol", "ExpsMaintPol"), ("mort_id", "BaseMort"), ("mf_name", "MortFactor"), ("sr_name", "Surrender")):
            cols[k].append(need(lookup(asmp, nm, p, pt, pt), nm))        # the original passes the policy type twice
    U = lambda k, dt=np.float64: np.asarray(cols[k], dtype=dt)[code]
    usex = [u[3] for u in uniq]
    # --- commutation columns per (sex, premium interest rate, premium mortality table)
    lts = sorted({(s, float(r), int(t)) for s, r, t in zip(usex, cols["rate"], cols["tid"])})
    com = [commutation(tb, *k) for k in lts]
    TD, TM, TN = (np.stack([c[i] for c in com]) for i in range(3))
    dzs = []
    for D in TD:
        dz = int(np.argmax(D == 0)) if (D == 0).any() else N_AGE
        assert (D[:dz] != 0).all() and (D[dz:] == 0).all(); dzs.append(dz)
    klt = [lts.index((s, float(r), int(t))) for s, r, t in zip(usex, cols["rate"], cols["tid"])]     # life table of each key
    lt = np.asarray(klt, np.int64)[code]
    dz = np.asarray(dzs, np.int64)[lt]
    if (x >= dz).any(): raise ZeroDivisionError("issue age beyond the premium mortality table (the original raises too)")
    # --- policy attributes (Policy space), vectorised with the original's operation order
    alpha = U("a1") + U("a2") * np.minimum(n / 10, 1)                                   # LoadAcqSA
    lmp1 = np.asarray([np.nan if v is None else v for v in cols["lmp1"]], np.float64)[code]
    lmp2 = np.asarray([np.nan if v is None else v for v in cols["lmp2"]], np.float64)[code]
    if (np.isnan(lmp1) & np.isnan(lmp2)).any(): raise ValueError("LoadMaintPrem parameters not found")
    beta = np.where(np.isnan(lmp1), (lmp2 + np.minimum(10, n)) / 100, lmp1)             # LoadMaintPrem
    delta = np.where(n < 5, 0.0005, np.where(n < 10, 0.001, 0.002))                      # LoadMaintPrem2
    gamma, g2 = U("gamma"), U("g2")
    isc = U("sc1") + U("sc2") * np.minimum(n / 10, 1)                                   # InitSurrCharge
    Dx, Dxn = TD[lt, x], TD[lt, x + n]
    axn = (TM[lt, x] - TM[lt, x + n]) / Dx                                              # Axn(x, n)
    ann1 = (TN[lt, x] - TN[lt, x + n]) / Dx                                             # AnnDuenx(x, n)
    with np.errstate(all="ignore"):
        annk = np.where(freq > 1, ann1 - (freq - 1) / (2 * freq) * (1 - Dxn / Dx), ann1)    # AnnDuenx(x, n, freq)
    wait = (TN[lt, x + n] - TN[lt, x + n]) / Dx                                         # AnnDuenx(x, 0, 1, n)
    is_endw = prod == "ENDW"
    if not np.isin(prod, ["TERM", "WL", "ENDW"]).all(): raise ValueError("invalid product")
    num = np.where(is_endw, Dxn / Dx + axn + alpha + gamma * annk + g2 * wait, axn + alpha + gamma * annk + g2 * wait)
    gpr = num / (1 - beta - delta) / freq / annk                                        # GrossPremRate
    netp = (axn + g2 * wait) / ann1                                                     # NetPremRate('PREM')
    apr = gpr * np.where(freq == 0, 1 / 10, freq)                                       # AnnPremRate
    # --- projection assumptions (Assumptions space)
    mort_id = U("mort_id", np.int64)
    kcol = [tb["mort_cols"].index((mid, u[3])) for u, mid in zip(uniq, cols["mort_id"])]            # mortality column of each key
    last_age = np.argmax(tb["mort"], axis=0)[np.asarray(kcol, np.int64)[code]]                                      # TableLastAge = idxmax
    last_t = np.minimum(last_age - x, n)
    T = int(last_t.max()) + 1
    def factor_table(names):                # one row per key
        tab = np.full((len(names), T + 1), np.nan)
        for i, nm in enumerate(names):
            for y in range(T + 1):
                v = tb["asmp_tables"].get((nm, y), np.nan)
                tab[i, y] = tab[i, y - 1] if (y > 0 and v != v) else v                  # missing years repeat the previous year
        return tab
    TAB_MF, TAB_SR = factor_table(cols["mf_name"]), factor_table(cols["sr_name"])
    TAB_KF = np.column_stack([np.asarray(cols[k], np.float64) for k in KF_COLS])
    # --- economic scenario (Economic space)
    rate = np.array([tb["scen"][(tb["scen_id"], "IntRate", t)] for t in range(T)], np.float64)
    infl_rate = lookup(asmp, "InflRate"); infl = [1]
    for t in range(1, T): infl.append(infl[t - 1] / (1 + infl_rate))
    vdisc = 1 / (1 + rate); DISC = np.ones(T)
    for t in range(1, T): DISC[t] = DISC[t - 1] * vdisc[t - 1]
    with np.errstate(all="ignore"): RD = np.where(TD != 0, 1 / TD, 0.0)
    f = lambda a: np.ascontiguousarray(a, dtype=dtype)
    return {"age0": x, "term": n, "last_t": last_t.astype(np.int64), "dz": dz, "crt": U("crt", np.int64), "pk": code.astype(np.int64),
            "pcount": f(table["PolicyCount"]), "sa": f(table["SumAssured"]), "gpr": f(gpr), "freq": f(freq), "apr": f(apr),
            "isc": f(isc), "netp": f(netp),
            "scen": np.zeros(len(x), np.int64), "disc_rate": f(rate), "vdisc": f(vdisc), "infl": f(infl), "DISC": f(DISC), "T": T,
            "TAB_Q": f(tb["mort"].T[kcol]), "TAB_MF": f(TAB_MF), "TAB_SR": f(TAB_SR), "TAB_M": f(TM[klt]), "TAB_N": f(TN[klt]),
            "TAB_D": f(TD[klt]), "TAB_RD": f(RD[klt]), "TAB_KF": f(TAB_KF), "TAX": dtype(lookup(asmp, "CnsmpTax")),
            "_keys": [tuple(u) for u in uniq], "_klt": [lts[i] for i in klt],
            "_attrs": {**{k: U(k) for k in KF_COLS}, "gpr": gpr, "netp": netp, "apr": apr, "isc": isc, "last_t": last_t, "crt": U("crt")}}

# ----------------------------------------------------------------- symbolic description
AXIOMS = {
    "Projection": ["BaseMortRate (pandas row lookup): If(t <= last_t, Q(mortality column, IssueAge + t), 0)"],
    "Policy": ["IssueAge, PolicyTerm, PolicyCount, SumAssured, PremFreq (model point)",
               "GrossPremRate, AnnPremRate, InitSurrCharge (per-policy values computed in prep; uninterpreted)",
               "ReserveNLP_Rate('PREM', t): NOT executed symbolically (row-wise apply with Python branches over LifeTable "
               "item spaces); replaced by a hand transcription of Policy.ReserveNLP_Rate + LifeTable.Axn/AnnDuenx over "
               "uninterpreted commutation columns M, N, D",
               "table fact: the commutation column D is non-zero exactly at ages below the policy's `dz` (asserted by prep)"],
    "Assumptions": ["CommInitPrem, CommRenPrem, CommRenTerm, ExpsAcqAnnPrem, ExpsAcqSA, ExpsAcqPol, ExpsMaintAnnPrem, "
                    "ExpsMaintSA, ExpsMaintPol, CnsmpTax, LastAge (dictionary lookups; uninterpreted)",
                    "MortFactor(y), SurrRate(y) (dictionary lookups with carry-forward; uninterpreted functions of the year)"],
    "Economic": ["DiscRate(t) (scenario data; uninterpreted function of the year)", "AsmpLookup('InflRate') (uninterpreted)"],
    "definitions": ["pvflow_prem, pvflow_exps, pvflow_ben: the three terms the PV cells discount, defined from Projection cells"]}

def symbolic_spec(path=None, fold_zero=False):
    """fold_zero: evaluate `(PolicyTerm == t) * 0` (a boolean Series times the literal 0, which the original
    writes in year 0) to the literal 0 instead of `if PolicyTerm == t then 0 else 0`. Sound, and irrelevant
    to Z3; it exists only because the Lean tactic cannot see through an integer if-then-else cast to a rational."""
    import z3
    from exactuary.symbolic import SymbolicSpec, Sym, Original, lift, tval, ite, real
    I, R = z3.Int, z3.Real
    IS, RS = z3.IntSort(), z3.RealSort()
    Q = z3.Function("Q", IS, IS, RS)             # mortality table [column, age]
    MF = z3.Function("MF", IS, IS, RS)           # mortality factor table [id, year]
    SR = z3.Function("SR", IS, IS, RS)           # surrender rate table [id, year]
    MX = z3.Function("MX", IS, IS, RS)           # commutation columns [life table, age]
    NX = z3.Function("NX", IS, IS, RS)
    D0 = z3.Function("D0", IS, IS, RS)
    def DX(l, a):
        """Commutation column D. Table fact built into its definition (SymbolicSpec has no place for a stated
        assumption about a table): D is non-zero exactly at ages below the policy's `dz`. prep asserts this
        of every D column it builds, on every run."""
        d = D0(l, a)
        return z3.If(a < I("dz"), z3.If(d == 0, z3.RealVal(1), d), z3.RealVal(0))
    DR = z3.Function("DR", IS, RS)               # scenario interest rate in year t
    class Flag(Sym):                             # the result of `t < CommRenTerm()`: a Series the original calls .any() on
        def any(s): return True                  # sound: where no policy satisfies it, both branches of the original give 0
    class Term(Sym):                             # CommRenTerm(): comparisons return a Flag
        def __gt__(s, o): return Flag(Sym.__gt__(s, o).e)
        def __lt__(s, o): return Flag(Sym.__lt__(s, o).e)
    class ZeroFlag(Sym):
        def __mul__(s, o): return 0 if (type(o) is int and o == 0) else Sym.__mul__(s, o)
    class TermSym(Sym):
        def __eq__(s, o): return ZeroFlag(Sym.__eq__(s, o).e)
        __hash__ = None
    class Proxy:                                 # `pol.X()` / `asmp.X()` / `scen[ScenID].X()` -> cells of another space
        def __init__(s, get): s._get = get
        def __getattr__(s, name): return lambda *a: s._get(name, *a)
        def __getitem__(s, key): return s
    S = lambda e: Sym(e)
    def setup(O):
        mp = {k: S(I(k)) for k in ("IssueAge", "PolicyTerm", "LastAge", "pk", "dz")}
        mp.update({k: S(R(k)) for k in ("PolicyCount", "SumAssured", "PremFreq", "GrossPremRate", "AnnPremRate", "InitSurrCharge",
                                        "NetPremRate_PREM", "LoadMaintSA2", "CommInitPrem", "CommRenPrem", "ExpsAcqAnnPrem",
                                        "ExpsAcqSA", "ExpsAcqPol", "ExpsMaintAnnPrem", "ExpsMaintSA", "ExpsMaintPol", "CnsmpTax",
                                        "InflRate")})
        mp["CommRenTerm"] = Term(I("CommRenTerm"))
        if fold_zero: mp["PolicyTerm"] = TermSym(I("PolicyTerm"))
        # --- Policy space: CashValueRate and SurrCharge are executed as written; the rest are axioms
        OP = Original(formulas_path("Projection.Policy"))
        for k in ("IssueAge", "PolicyTerm", "PolicyCount", "SumAssured", "PremFreq", "GrossPremRate", "AnnPremRate", "InitSurrCharge"):
            OP.axiom(k, (lambda v: lambda: v)(mp[k]))
        OP.ns["PremTerm"] = OP.ns["PolicyTerm"]
        def reserve(basis, t):
            assert basis == "PREM"
            x, n, l = mp["IssueAge"], mp["PolicyTerm"], lift(mp["pk"]); tt = tval(t)
            a, an = lift(x + tt), lift(x + n)
            D = DX(l, a); dead = D == 0                                              # LifeTable: `if Dx(x) == 0: return 0`
            axn = S(z3.If(dead, z3.RealVal(0), (MX(l, a) - MX(l, an)) / D))          # Axn(x+t, n-t)
            wait = S(z3.If(dead, z3.RealVal(0), (NX(l, an) - NX(l, an)) / D))        # AnnDuenx(x+t, n-m, 1, m-t), m = n
            ann = S(z3.If(dead, z3.RealVal(0), (NX(l, a) - NX(l, an)) / D))          # AnnDuenx(x+t, m-t)
            return S(ite(lift(tt <= n), lift(axn + (mp["LoadMaintSA2"] * wait - mp["NetPremRate_PREM"] * ann)), z3.IntVal(0)))
        OP.axiom("ReserveNLP_Rate", reserve)
        # --- Economic space: InflFactor and InvstRetRate are executed as written
        OE = Original(formulas_path("Economic"))
        OE.axiom("DiscRate", lambda t: S(DR(lift(tval(t)))))
        OE.axiom("AsmpLookup", lambda name: mp[name])
        # --- Assumptions space: all cells are dictionary lookups
        A = {k: (lambda v: lambda: v)(mp[k]) for k in ("CommInitPrem", "CommRenPrem", "CommRenTerm", "ExpsAcqAnnPrem", "ExpsAcqSA",
                                                        "ExpsAcqPol", "ExpsMaintAnnPrem", "ExpsMaintSA", "ExpsMaintPol", "CnsmpTax", "LastAge")}
        A["MortFactor"] = lambda y: S(MF(lift(mp["pk"]), lift(tval(y))))
        A["SurrRate"] = lambda y: S(SR(lift(mp["pk"]), lift(tval(y))))
        O.ns.update(pol=Proxy(OP), asmp=Proxy(lambda name, *a: A[name](*a)), scen=Proxy(OE), ScenID=1)
        O.axiom("BaseMortRate", lambda t: S(z3.If(lift(tval(t) <= O("last_t")), Q(lift(mp["pk"]), lift(mp["IssueAge"] + tval(t))), z3.RealVal(0))))
        # outputs that are cells of the child spaces, and the three discounted terms
        O.axiom("pol_ReserveNLP_Rate", lambda t, basis: OP("ReserveNLP_Rate", basis, t))
        O.axiom("pol_SurrCharge", lambda t: OP("SurrCharge", t))
        O.axiom("pol_CashValueRate", lambda t: OP("CashValueRate", t))
        O.axiom("asmp_MortFactor", lambda t: A["MortFactor"](t))
        O.axiom("asmp_SurrRate", lambda t: A["SurrRate"](t))
        O.axiom("pvflow_prem", lambda t: (tval(t) <= O("last_t")) * O("PremIncome", t))
        O.axiom("pvflow_exps", lambda t: (tval(t) <= O("last_t")) * O("ExpsTotal", t))
        O.axiom("pvflow_ben", lambda t: O("BenefitTotal", t) * (1 / (1 + O("DiscRate", t))))
        mp["_O"], mp["_OP"], mp["_OE"] = O, OP, OE
        return mp
    def kernel_args(O, mp, t):
        return {"age0": mp["IssueAge"], "term": S(lift(mp["PolicyTerm"])), "last_t": O("last_t"), "dz": mp["dz"],
                "crt": S(lift(mp["CommRenTerm"])), "pk": mp["pk"], "pcount": mp["PolicyCount"], "sa": mp["SumAssured"],
                "gpr": mp["GrossPremRate"], "freq": mp["PremFreq"], "apr": mp["AnnPremRate"], "isc": mp["InitSurrCharge"],
                "netp": mp["NetPremRate_PREM"],
                "disc_rate": O("DiscRate", t), "vdisc": 1 / (1 + O("DiscRate", t)), "infl": O("InflFactor", t)}
    def carry(O, mp, t):
        return [O("PolsIF_Beg1", t) - O("PolsDeath", t) - O("PolsSurr", t),
                O("AccumCF", t) + O("IntAccumCF", t) + O("NetInsurCF", t)]
    KF = dict(zip(KF_COLS, ("CommInitPrem", "CommRenPrem", "ExpsAcqAnnPrem", "ExpsAcqSA", "ExpsAcqPol", "ExpsMaintAnnPrem",
                            "ExpsMaintSA", "ExpsMaintPol", "LoadMaintSA2")))
    def tab_kf(i, j):                           # row = the policy's own key; column j = one per-key value (a model-point symbol)
        return S(R(KF[KF_COLS[j]]))
    tables = {"TAB_KF": tab_kf, "TAB_Q": lambda i, j: S(Q(lift(i), lift(j))), "TAB_MF": lambda i, j: S(MF(lift(i), lift(j))),
              "TAB_SR": lambda i, j: S(SR(lift(i), lift(j))), "TAB_M": lambda i, j: S(MX(lift(i), lift(j))),
              "TAB_N": lambda i, j: S(NX(lift(i), lift(j))), "TAB_D": lambda i, j: S(DX(lift(i), lift(j))),
              "TAB_RD": lambda i, j: S(z3.If(DX(lift(i), lift(j)) != 0, 1 / DX(lift(i), lift(j)), z3.RealVal(0)))}   # as prep defines it
    def output_cell(name):
        if name.startswith(("pol.", "asmp.")): return parse_cell(name.replace(".", "_"))
        return (name,)
    def lemmas(step, mp):
        """Early exit. P(t): no policies in force at the start of year t (PolsIF_Beg1(t) = 0).
        (1) if P(t), the kernel carries zero policies out of year t;
        (2) if it carries zero policies into year t >= 1, every accumulated term of year t is zero and P(t) holds."""
        O = mp["_O"]; t = z3.Int("t"); a = kernel_args(O, mp, 0); a.update(disc_rate=S(R("disc_rate")), vdisc=S(R("vdisc")), infl=S(R("infl")))
        args = [a[k] for k in KERNEL.row_args + KERNEL.time_series]; cs = KERNEL.consts[:-1] + [S(R("TAX"))]
        out = []
        for label, t0 in (("in year 0", True), ("in a year t>=1", False)):
            r = step(0 if t0 else S(t), t0, *args, S(R("any_pols")), S(R("any_accum")), *cs)
            out.append((f"early exit (1) {label}: nothing in force -> nothing carried forward",
                        ([] if t0 else [t >= 1]) + [real(lift(r[KERNEL.ix("PolsIF_Beg1")])) == 0], lift(r[KERNEL.n_out]), z3.RealVal(0)))
        r = step(S(t), False, *args, 0.0, S(R("any_accum")), *cs)
        for nm in ACCUMULATE + ["PolsIF_Beg1"]:
            out.append((f"early exit (2): {nm} is zero", [t >= 1], lift(r[KERNEL.ix(nm)]), z3.RealVal(0)))
        return out
    return SymbolicSpec(source_path=path or formulas_path(), setup=setup, kernel_args=kernel_args, carry=carry, tables=tables,
                        consts={"TAX": S(R("CnsmpTax"))}, output_cell=output_cell, lemmas=lemmas,
                        table_keys={"TAB_KF": {0: "pk"}})        # its symbolic definition is per policy: valid only in the policy's own row

COMPOSITE = {"BenefitTotal": ["BenefitDeath", "BenefitSurr"], "ExpsCommTotal": ["ExpsCommInit", "ExpsCommRen"],
             "ExpsTotal": ["ExpsCommInit", "ExpsCommRen", "ExpsAcq", "ExpsMaint"],
             "NetInsurCF": ["PremIncome", "BenefitTotal", "ExpsTotal"],
             "carried state: s_accum": ["AccumCF", "IntAccumCF", "NetInsurCF"]}
