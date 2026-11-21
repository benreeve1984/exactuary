"""Reproduce every result for the lifelib savings example.

    python run.py oracle     run the original model and store every intermediate value
    python run.py exact      compiled CPU kernel: exact comparison with the original, and timings
    python run.py harness    how good is the checking: exhaustive tables, coverage, seeded defects
    python run.py prove      symbolic proof of equivalence for all policies and months (Z3)
    python run.py precision  GPU and single precision: timings, accuracy, rounding bounds
    python run.py lean       re-check the proof obligations with Lean (set LEAN=/path/to/lean)
"""
import sys, os, pickle, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import model as M
from exactuary.compare import same, compare_trace
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)
SETS = [("edge", 500, 4, True), ("sample", 100, 100, True), ("sample", 400, 100, False)]
tag_of = lambda kind, n, s: f"{kind}_{n}x{s}"
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

def oracle():
    m = M.load_original(); pickle.dump(M.extract_tables(m.Projection), open(f"{DATA}/tables.pkl", "wb"))
    for kind, n, s, trace in SETS:
        m = M.load_original()
        mp, scen = M.portfolio_sample(m, n, s) if kind == "sample" else M.portfolio_edge(n, s)
        Z = M.normals(s); res = M.run_original(M.configure(m, mp, scen, Z), trace); tag = tag_of(kind, n, s)
        pickle.dump({"mp": mp, "scen": scen, "Z": Z}, open(f"{DATA}/inputs_{tag}.pkl", "wb"))
        out = {"pv": res["pv"], "T": res["T"], "seconds_pv": res["seconds_pv"]}
        if trace:
            out.update({"t:" + k: v for k, v in res["trace"].items()})
            out["inflation_factor"], out["disc_factors"] = res["inflation_factor"], res["disc_factors"]
        np.savez(f"{DATA}/oracle_{tag}.npz", **out)
        print(f"{tag}: {len(mp):,} rows x {res['T']} months; original took {res['seconds_pv']:.1f}s")

def exact():
    from exactuary.backends import numba_cpu
    tb = tables(); K = M.KERNEL; run_pv, run_trace = numba_cpu.build(K); res = {}
    for kind, n, s, trace in SETS[:2]:
        tag = tag_of(kind, n, s); inp, orc = load_set(tag); p = M.prep(inp["mp"], inp["scen"], inp["Z"], tb)
        assert p["T"] == int(orc["T"]) and (p["infl"] == orc["inflation_factor"]).all() and (p["DISC"] == orc["disc_factors"]).all()
        print(f"[{tag}] {len(inp['mp']):,} rows x {p['T']} months")
        tr = run_trace(p); bad, total = compare_trace(tr, {k: orc["t:" + k] for k in K.outputs}, K.outputs)
        from exactuary import gates; gates.early_exit_agrees(run_pv, p)
        acc = np.column_stack([tr[K.ix(k)].T @ p["DISC"] for k in K.accumulate])
        pv_blas = bool(same(M.net_cf_pv(acc), orc["pv"]).all()); from exactuary import gates; gates.record("present values identical when summed as the original sums them", pv_blas)
        pv = M.net_cf_pv(run_pv(p))
        rel = np.where(same(pv, orc["pv"]), 0, np.abs(pv - orc["pv"]) / np.maximum(np.abs(orc["pv"]), 1e-300))
        print(f"  present values identical when summed as the original sums them: {pv_blas}; as a running sum: max relative difference {np.nanmax(rel):.1e}")
        res[tag] = {"rows": len(inp["mp"]), "months": p["T"], "values": total, "outputs": len(K.outputs),
                    "outputs_differing": len(bad), "pv_identical_same_summation": pv_blas, "pv_running_sum_max_rel": float(np.nanmax(rel))}
    inp, orc = load_set("sample_400x100"); p = M.prep(inp["mp"], inp["scen"], inp["Z"], tb)
    run_pv(p); t_par = best(lambda: run_pv(p), 5); t_full = best(lambda: run_pv(p, False), 5)
    run_pv1, _ = numba_cpu.build(K, parallel=False); run_pv1(p); t_one = best(lambda: run_pv1(p), 2)
    t_one_full = best(lambda: run_pv1(p, False), 2); t_orig = float(orc["seconds_pv"])
    print(f"[speed, 40,000 rows x {p['T']} months] original {t_orig:.2f}s | 1 core, no early exit {t_one_full:.3f}s ({t_orig/t_one_full:.0f}x) | "
          f"1 core {t_one:.3f}s ({t_orig/t_one:.0f}x) | all cores, no early exit {t_full:.4f}s ({t_orig/t_full:.0f}x) | all cores {t_par:.4f}s ({t_orig/t_par:.0f}x)")
    res["speed_40k"] = {"rows": 40000, "months": p["T"], "original_s": t_orig, "one_core_no_exit_s": t_one_full, "one_core_s": t_one,
                        "all_cores_no_exit_s": t_full, "all_cores_s": t_par, "cores": os.cpu_count()}
    m = M.load_original(); mp, scen = M.portfolio_sample(m, 10000, 1000); Z = M.normals(1000)
    t = time.perf_counter(); p = M.prep(mp, scen, Z, tb); t_prep = time.perf_counter() - t
    t_big = best(lambda: run_pv(p), 2)
    print(f"[scale] 10,000,000 rows: {t_big:.1f}s on all cores (plus {t_prep:.1f}s input preparation); original extrapolated {t_orig/40000*1e7/60:.0f} min")
    res["scale_10M"] = {"rows": 10_000_000, "seconds": t_big, "prep_seconds": t_prep, "original_extrapolated_s": t_orig / 40000 * 1e7}
    save("exact", res)

def harness():
    import pandas as pd
    from exactuary import harness as H
    from exactuary.backends import numpy_vec
    tb = tables(); K = M.KERNEL; res = {}
    # (a) every table lookup, in and out of range, against the original's own lookup code
    m = M.load_original(); rows = [(spec, age - d, 12 * d + (age % 12 if d >= 0 else 0)) for spec in "ABCD" for age in range(0, 211) for d in range(-3, 16)]
    mp = pd.DataFrame(rows, columns=["spec_id", "age_at_entry", "duration_mth"])
    for c, v in (("sex", "M"), ("policy_term", 10), ("policy_count", 100), ("sum_assured", 500_000), ("premium_pp", 1000), ("av_pp_init", 0), ("accum_prem_init_pp", 0)): mp[c] = v
    mp.index.name = "row"; scen = np.zeros(len(mp), np.int64); Z = M.normals(1); P = M.configure(m, mp, scen, Z)
    cells = ["mort_rate", "mort_rate_mth", "lapse_rate", "surr_charge_rate", "pols_lapse", "pols_death", "coi_rate"]
    p = M.prep(mp, scen, Z, tb); p["T"] = 1; tr = numpy_vec.run_trace(K, p, names=cells)
    ok = all(bool(same(tr[j, 0], np.asarray(getattr(P, c)(0), dtype=np.float64)).all()) for j, c in enumerate(cells))
    print(f"(a) exhaustive tables: {len(mp):,} synthetic policies, {len(cells)} cells identical: {ok}")
    res["tables"] = {"policies": len(mp), "identical": ok}
    from exactuary import gates; gates.record("every table lookup equals the original's, in and out of range", ok is True or (isinstance(ok, dict) and all(ok.values())))
    # (b) coverage and (c) seeded defects
    data = {}
    for tag, full, pick in (("sample", "sample_100x100", slice(0, None, 11)), ("edge", "edge_500x4", slice(None))):
        inp, orc = load_set(full); p = M.prep(inp["mp"].iloc[pick], inp["scen"][pick], inp["Z"], tb)
        pf = M.prep(inp["mp"], inp["scen"], inp["Z"], tb)
        for k in ("T", "inv_ret", "infl", "DISC", "TAB_LAPSEMTH", "DUR_MIN"): p[k] = pf[k]
        data[tag] = {"p": p, "trace": np.stack([orc["t:" + n][:, pick] for n in K.outputs])}
        labels, cov = H.coverage(K, p); res[f"coverage_{tag}"] = [sum(len(c) for c in cov), 2 * len(cov)]
        print(f"(b) coverage [{tag}, {len(p['age0'])} rows]: {sum(len(c) for c in cov)}/{2*len(cov)} branch outcomes; "
              f"missing: {[l for l, c in zip(labels, cov) if len(c) < 2]}")
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
        res[tag] = {"proved": o["proved"], "total": o["total"], "seconds": o["seconds"], "refuted": [t for t, _ in o["refuted"]], "unknown": o["unknown"]}
    res["formulas"], res["axiomatised"] = o["formulas"], o["axiomatised"]
    print(f"original formulas executed symbolically: {o['formulas']}; axiomatised: {o['axiomatised']}")
    lem = S.prove_lemmas(M.KERNEL, ss); res["early_exit_lemma"] = all(v == "proved" for _, v in lem)
    print("early-exit lemma:", "proved" if res["early_exit_lemma"] else lem)
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
    inp, orc = load_set("sample_400x100"); orc = orc["pv"]
    P = {ft: M.prep(inp["mp"], inp["scen"], inp["Z"], tb, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
    sub = np.arange(0, 40000, 20); psub = M.prep(inp["mp"].iloc[sub], inp["scen"][sub], inp["Z"], tb)
    for k in ("T", "inv_ret", "infl", "DISC", "TAB_LAPSEMTH", "DUR_MIN"): psub[k] = P["float64"][k]
    scale = np.maximum(np.abs(orc), 1.0); bounds = {}
    for kind in kernels:
        for ft in ("float64", "float32"):
            v, e = run_pv_bounds(kern(kind, ft), psub, ft); e9 = M.net_cf_pv(e); e9[:, 8] = e[:, [0, 6, 8, 4, 5, 7]].sum(1)
            bounds[(kind, ft)] = e9
    e_orig = bounds[("reference", "float64")]
    print("accuracy against the original (40,000 rows x 9 present values)")
    runs = {}
    for dev in ("cpu", "gpu"):
        for kind in kernels:
            for ft in ("float64", "float32"):
                K = kern(kind, ft)
                if dev == "cpu":
                    run_pv, _ = numba_cpu.build(K, ftype=ft); runs[(dev, kind, ft)] = run_pv; pv = M.net_cf_pv(run_pv(P[ft]).astype(np.float64))
                else:
                    prepare = cuda.build(K, ft); runs[(dev, kind, ft)] = prepare; go, out = prepare(P[ft]); go(); pv = M.net_cf_pv(out.copy_to_host().astype(np.float64))
                diff = np.abs(pv - orc); rel = diff / scale; bnd = bounds[(kind, ft)] + e_orig
                r = {"identical_share": float(same(pv, orc).mean()), "worst_per_policy_rel": float(rel.max()),
                     "portfolio_totals_rel": float((np.abs(pv.sum(0) - orc.sum(0)) / np.abs(orc.sum(0))).max()),
                     "bound_worst_rel": float((bnd / scale[sub]).max()), "bound_median_rel": float(np.median(bnd / scale[sub])),
                     "observed_within_bound": bool((diff[sub] <= (2.0 if dev == "gpu" else 1.0) * bnd + 1e-300).all())}
                res[f"{dev}/{kind}/{ft}"] = r
                print(f"  {dev} {kind:10s} {ft}: worst per-policy {r['worst_per_policy_rel']:.1e} | portfolio totals {r['portfolio_totals_rel']:.1e} | "
                      f"bound worst {r['bound_worst_rel']:.1e}, median {r['bound_median_rel']:.1e} | within bound: {r['observed_within_bound']}")
    m = M.load_original(); mp, scen = M.portfolio_sample(m, 10000, 1000); Z = M.normals(1000)
    P = {ft: M.prep(mp, scen, Z, tb, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
    p = P["float64"]; months = int(np.minimum(np.maximum(12 * p["term"] - p["dm0"] + 1, 0) + 1, p["T"]).sum())
    print(f"speed: 10,000,000 policy-scenarios, {months/1e9:.2f}bn policy-months")
    for (dev, kind, ft), f in runs.items():
        if dev == "cpu": f(P[ft]); t = best(lambda: f(P[ft]))
        else: go, out = f(P[ft]); go(); t = best(go)
        res[f"{dev}/{kind}/{ft}"]["seconds_10M"] = t; print(f"  {dev} {kind:10s} {ft}: {t:.3f}s")
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
    save("lean", res)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd in ("prove", "lean"): {"prove": prove, "lean": lean}[cmd](arg)
    elif cmd in ("oracle", "exact", "harness", "precision"): globals()[cmd]()
    else: print(__doc__)
    from exactuary import gates
    gates.finish({"exact": ['exact comparison', 'early exit', 'present values'], "prove": ["proof of", "early-exit obligations"], "lean": ["Lean check"], "harness": ["every table lookup"]}.get(cmd, ()))
