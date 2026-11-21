"""The four example models behind one interface, for the timing and certification scripts.

Each adaptor returns a dict:
    kernel      the KernelSpec
    p           {"float64": inputs, "float32": inputs}
    prep        seconds of input preparation (float64)
    period      "month" or "year"
    original    (rows, seconds, description) for the original model, or None if not timed
    net         weights that combine the accumulated outputs into the net cashflow present value
`big=True` builds about ten million rows; `big=False` builds the workload the original was run on.
"""
import sys, os, time, pickle, importlib.util
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)

def load(example):
    """Import examples/<example>/model.py under its own name (every example calls its module `model`)."""
    d = os.path.join(ROOT, "examples", example); sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(f"model_{example}", os.path.join(d, "model.py"))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M); sys.path.remove(d)
    return M, os.path.join(d, "data")
def _both(f):
    t = time.perf_counter(); p = {"float64": f(np.float64)}; t_prep = time.perf_counter() - t
    p["float32"] = f(np.float32); return p, t_prep

def savings(big=True, original=True):
    M, data = load("lifelib_savings"); tb = pickle.load(open(f"{data}/tables.pkl", "rb")); orig = None
    if original:
        m = M.load_original(); mp, scen = M.portfolio_sample(m, 400, 100); P = M.configure(m, mp, scen, M.normals(100))
        t = time.perf_counter(); P.result_pv(); orig = (len(P.model_point()), time.perf_counter() - t, "400 model points x 100 scenarios")
    m = M.load_original(); pts, sc = (10000, 1000) if big else (400, 100)
    mp, scen = M.portfolio_sample(m, pts, sc); Z = M.normals(sc)
    p, t_prep = _both(lambda dt: M.prep(mp, scen, Z, tb, dt))
    return dict(kernel=M.KERNEL, p=p, prep=t_prep, period="month", original=orig, net=[1, 0, 0, 0, -1, -1, 1, -1, -1])

def savings_dynamic(big=True, original=True):
    M, data = load("lifelib_savings_dynamic"); tb = pickle.load(open(f"{data}/tables.pkl", "rb")); orig = None
    sim = next(s for s in M.SIM_IDS if M.is_dynamic(M.flags_of(tb, s))); flags = M.flags_of(tb, sim)
    m = M.load_original(); money = M.portfolio_money(m)
    if original:
        P = M.configure(m, money, 10000, sim); r = M.run_original(P, False)
        orig = (len(money) * 10000, float(r["seconds_pv"]) + float(r["seconds_mc"]), f"{len(money)} model points x 10,000 scenarios, sim_id {sim}")
    m.close()
    mp, sc = (M.portfolio_scale(1000), 10000) if big else (money, 1000)
    Z = M.normals(sc)
    p, t_prep = _both(lambda dt: M.prep(mp, sc, Z, tb, flags, dt))
    K = M.kernel_for(flags)
    return dict(kernel=K, p=p, prep=t_prep, period="month", original=orig, net=[1, 0, 0, 0, -1, -1, 1, -1, -1] + [0] * (len(K.accumulate) - 9))

def basicterm(big=True, original=True):
    M, data = load("lifelib_basicterm"); tb = pickle.load(open(f"{data}/tables.pkl", "rb")); orig = None
    if original:
        m = M.load_original(); mp = M.portfolio_sample(m, 10); P = M.configure(m, mp)
        t = time.perf_counter(); P.result_pv(); orig = (len(mp), time.perf_counter() - t, "lifelib's 10,000 model points x 10")
    m = M.load_original(); mp = M.portfolio_sample(m, 1000 if big else 1)
    p, t_prep = _both(lambda dt: M.prep(mp, tb, dt))
    return dict(kernel=M.KERNEL, p=p, prep=t_prep, period="month", original=orig, net=[1, -1, -1, -1, 0])

def fastlife(big=True, original=True):
    M, data = load("lifelib_fastlife"); orig = None
    if original:
        m = M.load_original(); table = M.portfolio_sample(m, 34); P = M.configure(m, table)
        orig = (len(table), float(M.run_original(P, False)["seconds_pv"]), "lifelib's 300 model points x 34")
    m = M.load_original(); mp = M.portfolio_sample(m, 33334 if big else 34); tb = M.extract_tables(m)
    p, t_prep = _both(lambda dt: M.prep(mp, tb, dt))
    return dict(kernel=M.KERNEL, p=p, prep=t_prep, period="year", original=orig, net=[1, -1, -1])

MODELS = {"lifelib_savings": savings, "lifelib_savings_dynamic": savings_dynamic,
          "lifelib_basicterm": basicterm, "lifelib_fastlife": fastlife}

def subset(spec, p, idx):
    """The inputs restricted to some rows; everything that is not per row is shared."""
    q = dict(p)
    for a in spec.row_args + ["scen"]: q[a] = np.asarray(p[a])[idx]
    return q
