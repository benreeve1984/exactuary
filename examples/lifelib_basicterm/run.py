"""Reproduce every result for lifelib basiclife/BasicTerm_ME.

    python run.py oracle     run the original model and store every intermediate value
    python run.py exact      compiled CPU kernel: exact comparison with the original, and timings
    python run.py harness    how good is the checking: exhaustive tables, coverage, seeded defects
    python run.py prove      symbolic proof of equivalence for all policies and months (Z3)
    python run.py precision  GPU and single precision: timings, accuracy, rounding bounds
    python run.py lean       re-check the proof obligations with Lean (set LEAN=/path/to/lean)
    python run.py baseline   BasicTerm_SE (one policy at a time): agreement with BasicTerm_ME, and timing
    python run.py reprice    two-pass demonstration: premium rates priced from a first projection
"""
import sys, os, pickle, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import model as M
from exactuary.compare import same, compare_trace
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)
SETS = [("edge", True), ("edgenan", True), ("sample", True), ("sample_x10", False)]
TRACED = [k for k, tr in SETS if tr]
def save(name, obj): json.dump(obj, open(os.path.join(RES, name + ".json"), "w"), indent=1, default=str)
def load_set(tag):
    inp = pickle.load(open(f"{DATA}/inputs_{tag}.pkl", "rb")); orc = np.load(f"{DATA}/oracle_{tag}.npz")
    return inp, orc
def tables(): return pickle.load(open(f"{DATA}/tables.pkl", "rb"))
def best(f, k=3):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
def portfolio(m, kind):
    return {"edge": lambda: M.portfolio_edge(), "edgenan": lambda: M.portfolio_edge_nan(),
            "sample": lambda: M.portfolio_sample(m), "sample_x10": lambda: M.portfolio_sample(m, 10)}[kind]()
def relmax(a, b):
    with np.errstate(all="ignore"):
        r = np.where(same(a, b), 0, np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
    return float(np.nanmax(r)) if r.size else 0.0

def oracle():
    m = M.load_original(); pickle.dump(M.extract_tables(m.Projection), open(f"{DATA}/tables.pkl", "wb"))
    for kind, trace in SETS:
        secs = []
        for rep in range(1 if trace else 2):
            m = M.load_original(); mp = portfolio(m, kind)
            res = M.run_original(M.configure(m, mp), trace); secs.append(res["seconds_pv"])
        pickle.dump({"mp": mp}, open(f"{DATA}/inputs_{kind}.pkl", "wb"))
        out = {"pv": res["pv"], "T": res["T"], "seconds_pv": min(secs)}
        if trace:
            out.update({"t:" + k: v for k, v in res["trace"].items()})
            for k in ("inflation_factor", "disc_factors", "premium_pp", "pv_pols_if", "net_premium_pp", "proj_len"): out[k] = res[k]
        np.savez(f"{DATA}/oracle_{kind}.npz", **out)
        nan = int(np.isnan(res["pv"]).any(1).sum())
        print(f"{kind}: {len(mp):,} rows x {res['T']} months; original took {min(secs):.2f}s; rows with NaN present values: {nan}")

def exact():
    from exactuary.backends import numba_cpu
    tb = tables(); K = M.KERNEL; run_pv, run_trace = numba_cpu.build(K); res = {}
    for kind in TRACED:
        inp, orc = load_set(kind); p = M.prep(inp["mp"], tb)
        assert p["T"] == int(orc["T"]) and (p["infl"] == orc["inflation_factor"]).all() and (p["DISC"] == orc["disc_factors"]).all()
        prem_same = bool(same(p["prem"], orc["premium_pp"]).all())
        print(f"[{kind}] {len(inp['mp']):,} rows x {p['T']} months; premium_pp input identical to the original's cell: {prem_same}")
        tr = run_trace(p); bad, total = compare_trace(tr, {k: orc["t:" + k] for k in K.outputs}, K.outputs)
        from exactuary import gates; gates.early_exit_agrees(run_pv, p)
        acc = np.column_stack([tr[K.ix(k)].T @ p["DISC"] for k in K.accumulate])
        pv_blas = bool(same(M.reported(acc), orc["pv"]).all())
        pvif_blas = bool(same(acc[:, 4], orc["pv_pols_if"]).all()); np_blas = bool(same(M.net_premium(acc), orc["net_premium_pp"]).all())
        from exactuary import gates; gates.record("inputs prepared for the kernel equal the original's cells", prem_same); gates.record("present values identical when summed as the original sums them", pv_blas and pvif_blas and np_blas)
        racc = run_pv(p); pv = M.reported(racc)
        r_pv, r_np = relmax(pv, orc["pv"]), relmax(M.net_premium(racc), orc["net_premium_pp"])
        nan_same = bool((np.isnan(pv) == np.isnan(orc["pv"])).all())
        noexit = M.reported(run_pv(p, False)); exit_same = bool(same(noexit, pv).all())
        print(f"  summed as the original sums them: 5 present values identical: {pv_blas}; pv_pols_if identical: {pvif_blas}; net_premium_pp identical: {np_blas}")
        print(f"  as a running sum: max relative difference in present values {r_pv:.1e}, in net_premium_pp {r_np:.1e}; NaN pattern identical: {nan_same}; "
              f"early exit gives the same bits as no early exit: {exit_same}")
        res[kind] = {"rows": len(inp["mp"]), "months": p["T"], "values": total, "outputs": len(K.outputs), "outputs_differing": len(bad),
                     "differing": {k: v for k, v in bad.items()}, "premium_pp_identical": prem_same,
                     "pv_identical_same_summation": pv_blas, "pv_pols_if_identical_same_summation": pvif_blas,
                     "net_premium_pp_identical_same_summation": np_blas, "pv_running_sum_max_rel": r_pv,
                     "net_premium_running_sum_max_rel": r_np, "nan_pattern_identical": nan_same, "early_exit_same_bits": exit_same,
                     "rows_with_nan_pv": int(np.isnan(orc["pv"]).any(1).sum())}
    inp, orc = load_set("sample_x10"); p = M.prep(inp["mp"], tb); n = len(inp["mp"])
    assert bool((np.abs(M.reported(run_pv(p)) - orc["pv"]) <= 1e-9 * np.maximum(np.abs(orc["pv"]), 1)).all())
    t_par = best(lambda: run_pv(p), 5); t_full = best(lambda: run_pv(p, False), 5)
    run_pv1, _ = numba_cpu.build(K, parallel=False); run_pv1(p); t_one = best(lambda: run_pv1(p), 3)
    t_one_full = best(lambda: run_pv1(p, False), 3); t_orig = float(orc["seconds_pv"])
    print(f"[speed, {n:,} rows x {p['T']} months] original {t_orig:.2f}s | 1 core, no early exit {t_one_full:.3f}s ({t_orig/t_one_full:.0f}x) | "
          f"1 core {t_one:.3f}s ({t_orig/t_one:.0f}x) | all cores, no early exit {t_full:.4f}s ({t_orig/t_full:.0f}x) | all cores {t_par:.4f}s ({t_orig/t_par:.0f}x)")
    res["speed_100k"] = {"rows": n, "months": p["T"], "original_s": t_orig, "original_s_10k": float(load_set("sample")[1]["seconds_pv"]),
                         "one_core_no_exit_s": t_one_full, "one_core_s": t_one,
                         "all_cores_no_exit_s": t_full, "all_cores_s": t_par, "cores": os.cpu_count()}
    m = M.load_original(); mp = M.portfolio_sample(m, 1000)
    t = time.perf_counter(); p = M.prep(mp, tb); t_prep = time.perf_counter() - t
    t_big = best(lambda: run_pv(p), 3); t_big_full = best(lambda: run_pv(p, False), 2)
    print(f"[scale] {len(mp):,} rows: {t_big:.2f}s on all cores ({t_big_full:.2f}s without early exit; plus {t_prep:.1f}s input preparation); "
          f"original extrapolated {t_orig/n*len(mp)/60:.1f} min")
    res["scale_10M"] = {"rows": len(mp), "seconds": t_big, "seconds_no_exit": t_big_full, "prep_seconds": t_prep, "original_extrapolated_s": t_orig / n * len(mp)}
    save("exact", res)

def harness():
    import pandas as pd
    from exactuary import harness as H
    from exactuary.backends import numpy_vec
    tb = tables(); K = M.KERNEL; res = {}
    # (a) every table lookup, in and out of range, against the original's own lookup cells
    def synth(rows):
        mp = pd.DataFrame(rows, columns=["age_at_entry", "policy_term", "sum_assured", "duration_mth"])
        mp["sex"] = "M"; mp["policy_count"] = 100
        return M._index(mp[M.COLS])
    def table_check(mp, cells):
        m = M.load_original(); P = M.configure(m, mp); p = M.prep(mp, tb); p["T"] = 1
        tr = numpy_vec.run_trace(K, p, names=cells)
        ok = {c: bool(same(tr[j, 0], np.asarray(getattr(P, c)(0), dtype=np.float64)).all()) for j, c in enumerate(cells)}
        ok["premium_pp"] = bool(same(p["prem"], np.asarray(P.premium_pp(), dtype=np.float64)).all())
        return ok, p
    mp = synth([(age - d, 10, 500_000, 12 * d + (age % 12)) for age in range(0, 141) for d in range(-50, 71)])
    ok, p = table_check(mp, ["mort_rate", "mort_rate_mth", "lapse_rate", "pols_lapse", "pols_death", "age", "duration"])
    print(f"(a) mortality and lapse: {len(mp):,} synthetic policies (attained age 0..140 x duration -50..70 years): {ok}")
    res["tables_mort_lapse"] = {"policies": len(mp), "identical": ok}
    from exactuary import gates; gates.record("every table lookup equals the original's, in and out of range (tables_mort_lapse)", ok is True or (isinstance(ok, dict) and all(ok.values())))
    mp = synth([(a, term, sa, 0) for a in range(0, 81) for term in range(0, 41) for sa in (0, 1, 1000, 622_000, 999_999, 10**9 + 7)])
    ok, p = table_check(mp, ["premiums", "commissions", "claims", "expenses", "net_cf"])
    print(f"(a) premium table: {len(mp):,} synthetic policies (entry age 0..80 x term 0..40 x 6 sums assured), "
          f"{int(np.isnan(p['prem']).sum()):,} off-table (NaN): {ok}")
    res["tables_premium"] = {"policies": len(mp), "off_table_nan": int(np.isnan(p["prem"]).sum()), "identical": ok}
    from exactuary import gates; gates.record("every table lookup equals the original's, in and out of range (tables_premium)", ok is True or (isinstance(ok, dict) and all(ok.values())))
    # (b) coverage and (c) seeded defects
    data = {}
    for tag, pick in (("sample", slice(0, None, 11)), ("edge", slice(None)), ("edgenan", slice(None))):
        inp, orc = load_set(tag); p = M.prep(inp["mp"].iloc[pick], tb); pf = M.prep(inp["mp"], tb)
        for k in M.SHARED: p[k] = pf[k]
        if tag != "edgenan":       # NaN rows cannot be used by mutation_scores (its tolerance checks flag NaN = NaN)
            data[tag] = {"p": p, "trace": np.stack([orc["t:" + n][:, pick] for n in K.outputs])}
        labels, cov = H.coverage(K, p); res[f"coverage_{tag}"] = [sum(len(c) for c in cov), 2 * len(cov)]
        missing = [(l, sorted(c)) for l, c in zip(labels, cov) if len(c) < 2]; res[f"coverage_{tag}_missing"] = missing
        print(f"(b) coverage [{tag}, {len(p['age0'])} rows]: {sum(len(c) for c in cov)}/{2*len(cov)} branch outcomes; one-sided (outcomes seen): {missing}")
    t0 = time.time(); ms = H.mutation_scores(K, data, M.SWAPS, workers=6)
    print(f"(c) {len(ms['mutants'])} seeded defects judged in {time.time()-t0:.0f}s")
    for name, k, n in ms["table"]: print(f"    caught {k}/{n} = {k/n:.1%}  <- {name}")
    print("    survivors of the strongest check:"); [print("     ", s) for s in ms["survivors"]]
    res.update({"defects": len(ms["mutants"]), "table": ms["table"], "survivors": ms["survivors"]}); save("harness", res)

def prove(path=None):
    from exactuary import symbolic as S
    from exactuary.mutate import mutants
    ss = M.symbolic_spec(path); res = {}
    for tag, K in (("reference", M.KERNEL), ("rearranged", M.KERNEL_REARRANGED)):
        o = S.prove(K, ss)
        print(f"{tag} kernel: {o['proved']}/{o['total']} obligations proved in {o['seconds']:.1f}s; refuted {[t for t, _ in o['refuted']]}; unknown {o['unknown']}")
        for t, mdl in o["refuted"]: print("   counterexample for", t, ":", S.show_model(mdl))
        res[tag] = {"proved": o["proved"], "total": o["total"], "seconds": o["seconds"], "refuted": [t for t, _ in o["refuted"]], "unknown": o["unknown"]}
    used = set()
    for base in (True, False): used |= {k[0] for k in S.obligations(M.KERNEL, ss, base)[2].memo}
    res["formulas_in_file"], res["axiomatised"] = o["formulas"], o["axiomatised"]
    res["cells_reached"] = sorted(used); res["cells_executed_as_written"] = sorted(used - set(o["axiomatised"]))
    print(f"formulas in the original's source file: {o['formulas']}; cells reached by the obligations: {len(used)}, of which axiomatised: {o['axiomatised']}")
    print("  executed symbolically as written:", sorted(used - set(o["axiomatised"])))
    lem = S.prove_lemmas(M.KERNEL, ss); res["early_exit_lemma"] = all(v == "proved" for _, v in lem); res["early_exit_lemma_parts"] = lem
    print("early-exit lemma:", "proved" if res["early_exit_lemma"] else "NOT proved", lem)
    ms = mutants(M.STEP_SRC, M.SWAPS); t0 = time.time(); tally, rows = S.refute_mutants(M.KERNEL, ss, ms)
    print(f"seeded defects: {tally} of {len(ms)} in {time.time()-t0:.0f}s")
    for kind, desc, v, cex in rows:
        if v != "refuted": print(f"  {v.upper()}: [{kind}] {desc}")
    res["defects"] = {"tally": tally, "rows": rows}; save("prove", res)

def precision():
    from exactuary.backends import numba_cpu, cuda
    from exactuary.errbound import run_pv_bounds
    from exactuary.spec import to_f32_src
    tb = tables(); res = {}
    kernels = {"reference": M.KERNEL, "rearranged": M.KERNEL_REARRANGED}
    def kern(kind, ft): return kernels[kind].with_src(to_f32_src(kernels[kind].step_src)) if ft == "float32" else kernels[kind]
    def six(acc):      # 5 reported present values, pv_pols_if
        return np.column_stack([M.reported(acc), acc[:, 4]])
    inp, orc_ = load_set("sample"); orc = np.column_stack([orc_["pv"], orc_["pv_pols_if"]]); onp = orc_["net_premium_pp"]; n = len(inp["mp"])
    P = {ft: M.prep(inp["mp"], tb, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
    sub = np.arange(0, n, 5); psub = M.prep(inp["mp"].iloc[sub], tb)
    for k in M.SHARED: psub[k] = P["float64"][k]
    scale = np.maximum(np.abs(orc), 1.0); bounds = {}
    for kind in kernels:
        for ft in ("float64", "float32"):
            v, e = run_pv_bounds(kern(kind, ft), psub, ft); e6 = six(e); e6[:, 4] = e[:, :4].sum(1)
            bounds[(kind, ft)] = e6
    e_orig = bounds[("reference", "float64")]
    print(f"accuracy against the original ({n:,} rows x 6 present values; bounds on every 5th row)")
    runs = {}; small = {}
    for dev in ("cpu", "gpu"):
        for kind in kernels:
            for ft in ("float64", "float32"):
                K = kern(kind, ft)
                if dev == "cpu":
                    run_pv, _ = numba_cpu.build(K, ftype=ft); runs[(dev, kind, ft)] = run_pv; acc = run_pv(P[ft]).astype(np.float64)
                else:
                    prepare = cuda.build(K, ft); runs[(dev, kind, ft)] = prepare; go, out = prepare(P[ft]); go(); acc = out.copy_to_host().astype(np.float64)
                pv = six(acc); diff = np.abs(pv - orc); rel = diff / scale; bnd = bounds[(kind, ft)] + e_orig
                r = {"identical_share": float(same(pv, orc).mean()), "worst_per_policy_rel": float(rel.max()),
                     "worst_per_policy_abs": float(diff.max()),
                     "worst_per_policy_rel_by_pv": dict(zip(M.PV_NAMES + ["PV pols_if"], [float(x) for x in rel.max(0)])),
                     "worst_per_policy_abs_by_pv": dict(zip(M.PV_NAMES + ["PV pols_if"], [float(x) for x in diff.max(0)])),
                     "portfolio_totals_rel": float((np.abs(pv.sum(0) - orc.sum(0)) / np.abs(orc.sum(0))).max()),
                     "net_premium_worst_rel": relmax(M.net_premium(acc), onp),
                     "bound_worst_rel": float((bnd / scale[sub]).max()), "bound_median_rel": float(np.median(bnd / scale[sub])),
                     "observed_within_bound": bool((diff[sub] <= (2.0 if dev == "gpu" else 1.0) * bnd + 1e-300).all())}
                res[f"{dev}/{kind}/{ft}"] = r; small[(dev, kind, ft)] = acc
                print(f"  {dev} {kind:10s} {ft}: identical {r['identical_share']:.1%} | worst per-policy rel {r['worst_per_policy_rel']:.1e} (abs {r['worst_per_policy_abs']:.2e}) | "
                      f"portfolio totals {r['portfolio_totals_rel']:.1e} | net premium {r['net_premium_worst_rel']:.1e} | "
                      f"bound worst {r['bound_worst_rel']:.1e}, median {r['bound_median_rel']:.1e} | within bound: {r['observed_within_bound']}")
    m = M.load_original(); mp = M.portfolio_sample(m, 1000)
    P = {ft: M.prep(mp, tb, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
    p = P["float64"]; months = int(np.minimum(np.maximum(12 * p["term"] - p["dm0"] + 1, 0) + 1, p["T"]).sum())
    print(f"speed: {len(mp):,} policies, about {months/1e9:.2f}bn policy-months to maturity")
    for (dev, kind, ft), f in runs.items():
        if dev == "cpu": big = f(P[ft]); t = best(lambda: f(P[ft]))
        else: go, out = f(P[ft]); go(); t = best(go); big = out.copy_to_host()
        big = big.astype(np.float64).reshape(1000, n, -1)       # the 10M rows are the sample repeated 1000 times
        ok = bool(same(big, small[(dev, kind, ft)][None]).all())
        res[f"{dev}/{kind}/{ft}"]["seconds_10M"] = t; res[f"{dev}/{kind}/{ft}"]["results_10M_equal_10k_run"] = ok
        print(f"  {dev} {kind:10s} {ft}: {t:.3f}s; every copy reproduces the 10,000-row result: {ok}")
    res["policy_months_10M"] = months; save("precision", res)

def lean(path=None):
    from exactuary import lean as L
    exe = os.environ.get("LEAN", "lean"); ss = M.symbolic_spec(path); res = {}
    n, failed = L.run_lean(L.GLUE, exe); res["glue"] = {"theorems": n, "failed": failed}
    print(f"induction theorems: {n - len(failed)}/{n} checked")
    for tag, K in (("reference", M.KERNEL), ("rearranged", M.KERNEL_REARRANGED)):
        fn = os.path.join(RES, f"Obligations_{tag}.lean"); L.export(K, ss, fn, tag, composite=M.COMPOSITE)
        t0 = time.time(); n, failed = L.run_lean(fn, exe)
        print(f"{tag} kernel: {n - len(failed)}/{n} obligations checked by Lean in {time.time()-t0:.0f}s; failed: {failed}")
        res[tag] = {"theorems": n, "failed": failed}
    # negative control: a seeded defect must make Lean reject at least one obligation
    from exactuary.mutate import mutants
    kind, desc, src = [x for x in mutants(M.STEP_SRC, M.SWAPS) if "p_decr - p_lapse - p_death" in x[1] and x[0] == "operator"][0]
    fn = os.path.join(DATA, "Obligations_defect.lean"); L.export(M.KERNEL.with_src(src), ss, fn, "defect", composite=M.COMPOSITE)
    n, failed = L.run_lean(fn, exe, gate=False); res["negative_control"] = {"defect": desc, "theorems": n, "failed": failed}
    print(f"negative control ({desc.split('   in:')[0]}): Lean rejects {len(failed)}/{n} obligations: {failed}")
    save("lean", res)

def baseline(n_cmp=200, n_time=1000):
    """BasicTerm_SE: the same model written one policy at a time (plain Python scalars under modelx)."""
    res = {}
    m = M.load_original(); mp = M.portfolio_sample(m); P = M.configure(m, mp.iloc[:n_cmp].copy())
    T = P.max_proj_len(); me = M.run_original(P, True); plen = me["proj_len"]
    s = M.load_unvectorised(); S = s.Projection
    assert S.model_point_table.equals(mp)
    cells = [n for n in M.OUTPUTS]; from exactuary.oracle import parse_cell
    compared = differing = differing_started = 0; worst = 0.0; diff_cells = {}; pv_rel = 0.0; pv_ident = 0
    for i in range(n_cmp):
        S.point_id = i + 1; L = int(S.proj_len()); assert L == plen[i]
        for nme in cells:
            c, *a = parse_cell(nme)
            v = np.array([float(getattr(S, c)(t, *a)) for t in range(L)]); o = me["trace"][nme][:L, i]
            eq = same(v, o); compared += L; k = int((~eq).sum()); differing += k
            differing_started += int((~eq & (me["trace"]["duration_mth"][:L, i] >= 0)).sum())
            if k:
                diff_cells[nme] = diff_cells.get(nme, 0) + k
                worst = max(worst, float((np.abs(v - o)[~eq] / np.maximum(np.abs(o[~eq]), 1e-300)).max()))
        pv = S.result_pv().to_numpy(dtype=np.float64).ravel(); o = me["pv"][i]
        pv_ident += int(same(pv, o).sum()); pv_rel = max(pv_rel, relmax(pv, o))
    print(f"agreement on {n_cmp} sample policies, months before each policy's own projection end: {compared:,} values compared, "
          f"{differing:,} differ, of which in months with duration_mth >= 0: {differing_started}; cells differing: {diff_cells}")
    print(f"  present values: {pv_ident}/{5*n_cmp} identical, worst relative difference {pv_rel:.1e}")
    res["agreement"] = {"policies": n_cmp, "values": compared, "differing": differing, "differing_in_months_with_duration_ge_0": differing_started, "cells_differing": diff_cells,
                        "pv_identical": pv_ident, "pv_values": 5 * n_cmp, "pv_worst_rel": pv_rel}
    s = M.load_unvectorised(); S = s.Projection
    t0 = time.perf_counter()
    for i in range(n_time):
        S.point_id = i + 1; S.result_pv()
    per = (time.perf_counter() - t0) / n_time
    orc = np.load(f"{DATA}/oracle_sample_x10.npz"); t_me = float(orc["seconds_pv"]); n_me = len(orc["pv"])
    print(f"timing: {n_time} policies, {per*1e3:.2f} ms per policy; extrapolated: {per*n_me:.0f}s for the {n_me:,}-row benchmark "
          f"(BasicTerm_ME: {t_me:.2f}s), {per*1e7/3600:.1f} hours for 10,000,000 rows")
    res["timing"] = {"policies_timed": n_time, "seconds_per_policy": per, "extrapolated_s_benchmark": per * n_me, "benchmark_rows": n_me,
                     "vectorised_original_s_benchmark": t_me, "extrapolated_s_10M": per * 1e7}
    save("baseline", res)

def reprice():
    """Two-pass use of the kernel. BasicTerm_ME itself reads premium rates from a table; lifelib created
    that table (create_premium_table.ipynb) as (1 + loading_prem()) * net_premium_pp() for new business
    with sum assured 1, i.e. from present values of a first projection. Here: pass 1 runs the kernel on
    that grid and derives the rates with the original's own expressions; pass 2 projects the sample
    portfolio using the derived rates."""
    import itertools, pandas as pd
    from exactuary.backends import numba_cpu
    tb = tables(); K = M.KERNEL; run_pv, run_trace = numba_cpu.build(K); res = {}
    grid = pd.DataFrame(itertools.product(range(20, 60), [10, 15, 20]), columns=["age_at_entry", "policy_term"])
    grid["sex"] = "M"; grid["policy_count"] = 1; grid["sum_assured"] = 1; grid["duration_mth"] = 0; grid = M._index(grid[M.COLS])
    m = M.load_original(); P = M.configure(m, grid); rate_orig = (1 + P.loading_prem()) * P.net_premium_pp()
    p = M.prep(grid, tb); tr = run_trace(p)
    acc = np.column_stack([tr[K.ix(k)].T @ p["DISC"] for k in K.accumulate])
    rate_k = (1 + 0.5) * M.net_premium(acc); rate_run = (1 + 0.5) * M.net_premium(run_pv(p))
    shipped = np.array([tb["prem"][(a, b)] for a, b in zip(grid["age_at_entry"], grid["policy_term"])])
    res["pass1"] = {"grid": len(grid), "identical_to_original_cells": int(same(rate_k, rate_orig).sum()),
                    "running_sum_max_rel": relmax(rate_run, rate_orig),
                    "identical_to_shipped_table": int(same(rate_k, shipped).sum()), "shipped_table_max_rel": relmax(rate_k, shipped)}
    print(f"pass 1: {len(grid)} rates; identical to the original's (1 + loading_prem()) * net_premium_pp(): {res['pass1']['identical_to_original_cells']}; "
          f"with running-sum present values max relative difference {res['pass1']['running_sum_max_rel']:.1e}; "
          f"identical to the shipped premium_table: {res['pass1']['identical_to_shipped_table']} (max relative difference {res['pass1']['shipped_table_max_rel']:.1e})")
    tb2 = dict(tb, prem={(int(a), int(b)): float(v) for a, b, v in zip(grid["age_at_entry"], grid["policy_term"], rate_k)})
    inp, orc = load_set("sample"); p1, p2 = M.prep(inp["mp"], tb), M.prep(inp["mp"], tb2)
    m2 = M.load_original(); P2 = M.configure(m2, inp["mp"]); pt = P2.premium_table       # the original, given the derived table as input
    P2.premium_table = pd.Series(rate_k, index=pd.MultiIndex.from_arrays([grid["age_at_entry"], grid["policy_term"]], names=pt.index.names), name=pt.name)
    pv_o = P2.result_pv()[M.PV_NAMES].to_numpy(dtype=np.float64); prem_o = np.asarray(P2.premium_pp(), dtype=np.float64)
    tr = run_trace(p2); acc = np.column_stack([tr[K.ix(k)].T @ p2["DISC"] for k in K.accumulate]); pv = M.reported(acc)
    res["pass2"] = {"rows": len(inp["mp"]), "premium_pp_identical_to_original_with_derived_table": int(same(p2["prem"], prem_o).sum()),
                    "pv_identical_to_original_with_derived_table": int(same(pv, pv_o).sum()), "pv_values": int(pv.size),
                    "premium_pp_identical_to_shipped_table_run": int(same(p1["prem"], p2["prem"]).sum()),
                    "pv_max_rel_vs_shipped_table_run": relmax(pv, orc["pv"]),
                    "total_pv_premiums_rel_vs_shipped_table_run": float(abs(pv[:, 0].sum() - orc["pv"][:, 0].sum()) / orc["pv"][:, 0].sum())}
    r = res["pass2"]
    print(f"pass 2: sample portfolio, kernel against the original, both given the derived rates: premium_pp identical {r['premium_pp_identical_to_original_with_derived_table']}/{r['rows']}; "
          f"present values identical {r['pv_identical_to_original_with_derived_table']}/{pv.size}")
    print(f"  against the run with lifelib's shipped table: premium_pp identical for {r['premium_pp_identical_to_shipped_table_run']}/{r['rows']} policies; "
          f"total PV premiums differs by {r['total_pv_premiums_rel_vs_shipped_table_run']:.1e} relative")
    save("reprice", res)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd in ("prove", "lean"): {"prove": prove, "lean": lean}[cmd](arg)
    elif cmd in ("oracle", "exact", "harness", "precision", "baseline", "reprice"): globals()[cmd]()
    else: print(__doc__)
    from exactuary import gates
    gates.finish({"exact": ['exact comparison', 'early exit', 'present values', 'inputs'], "prove": ["proof of", "early-exit obligations"], "lean": ["Lean check"], "harness": ["every table lookup"]}.get(cmd, ()))
