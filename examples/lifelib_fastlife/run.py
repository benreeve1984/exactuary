"""Reproduce every result for the lifelib fastlife example.

    python run.py oracle     run the original model and store every intermediate value
    python run.py exact      compiled CPU kernel: exact comparison with the original, and timings
    python run.py harness    how good is the checking: exhaustive tables, coverage, seeded defects
    python run.py prove      symbolic proof of equivalence for all policies and years (Z3)
    python run.py precision  GPU and single precision: timings, accuracy, rounding bounds
    python run.py lean       re-check the proof obligations with Lean (set LEAN=/path/to/lean)
    python run.py baseline   lifelib simplelife (one policy at a time): same values? how slow?
"""
import sys, os, pickle, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import model as M
from exactuary.compare import same, compare_trace
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)
# (tag, kind, size, keep the full trace). "sample_x34" is the benchmark workload: 10,200 policies.
SETS = [("edge_1500", "edge", 1500, True), ("sample_300", "sample", 1, True), ("sample_x34", "sample", 34, False)]
BENCH = "sample_x34"
def save(name, obj): json.dump(obj, open(os.path.join(RES, name + ".json"), "w"), indent=1, default=str)
def load_set(tag):
    inp = pickle.load(open(f"{DATA}/inputs_{tag}.pkl", "rb")); orc = np.load(f"{DATA}/oracle_{tag}.npz")
    return inp, orc
def best(f, k=3):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
def rel_diff(a, b):
    return np.where(same(a, b), 0.0, np.abs(a - b) / np.maximum(np.abs(b), 1.0))
def build_original(kind, size):
    """A fresh original model with its inputs set; returns (model, Projection space, table, extracted inputs)."""
    m = M.load_original()
    if kind == "sample": table = M.portfolio_sample(m, size)
    else:
        M.edge_inputs(m); table = M.portfolio_edge(size) if kind == "edge" else M.portfolio_tables()
    P = M.configure(m, table)
    return m, P, table, M.extract_tables(m)

def oracle():
    for tag, kind, size, trace in SETS:
        m, P, table, tb = build_original(kind, size)
        res = M.run_original(P, trace)
        pickle.dump({"table": table, "tb": tb}, open(f"{DATA}/inputs_{tag}.pkl", "wb"))
        out = {"pv": res["pv"], "T": res["T"], "seconds_pv": res["seconds_pv"]}
        if trace:
            out.update({"t:" + k: v for k, v in res["trace"].items()})
            out.update({"pv:" + k: v for k, v in res["pv_t"].items()})
            out.update({"a:" + k: v for k, v in res["attrs"].items()})
        np.savez(f"{DATA}/oracle_{tag}.npz", **out)
        print(f"{tag}: {len(table):,} policies x {res['T']} years; original PV_NetCashflow(0) took {res['seconds_pv']:.1f}s", flush=True)

def exact():
    from exactuary.backends import numba_cpu
    K = M.KERNEL; run_pv, run_trace = numba_cpu.build(K); res = {}
    trivial = [K.ix(n) for n in M.ZERO_CELLS]
    for tag, kind, size, trace in SETS[:2]:
        inp, orc = load_set(tag); p = M.prep(inp["table"], inp["tb"])
        assert p["T"] == int(orc["T"])
        attrs_ok = all(bool(same(np.asarray(p["_attrs"][k], np.float64), orc["a:" + k].astype(np.float64)).all()) for k in M.ATTR_CELLS)
        print(f"[{tag}] {len(inp['table']):,} policies x {p['T']} years; per-policy inputs derived in prep identical to the original's cells: {attrs_ok}")
        tr = run_trace(p); bad, total = compare_trace(tr, {k: orc["t:" + k] for k in K.outputs}, K.outputs)
        from exactuary import gates; gates.early_exit_agrees(run_pv, p)
        nontrivial = total - len(trivial) * tr.shape[1] * tr.shape[2]
        pvb = M.pv_backward(tr, K, p)
        pv_bad = {c: int((~same(pvb[c], orc["pv:" + c])).sum()) for c in M.PV_CELLS}
        from exactuary import gates; gates.record("inputs prepared for the kernel equal the original's cells", attrs_ok); gates.record("present values identical when summed as the original sums them", not any(pv_bad.values()), str(pv_bad))
        pv_vals = sum(pvb[c].size for c in M.PV_CELLS)
        pv = M.pv_from_acc(run_pv(p)); pv_full = M.pv_from_acc(run_pv(p, False)); rel = rel_diff(pv, orc["pv"])
        print(f"  PV cells for every year, recursion as the original writes it: {pv_vals:,} values, differing {sum(pv_bad.values())}")
        print(f"  PV at time 0 as a running sum: max relative difference {np.nanmax(rel):.1e}; early exit changes nothing: {bool(same(pv, pv_full).all())}")
        res[tag] = {"rows": len(inp["table"]), "years": p["T"], "outputs": len(K.outputs), "values": total, "values_nontrivial": nontrivial,
                    "outputs_differing": len(bad), "differing": {k: v for k, v in bad.items()}, "prep_attributes_identical": attrs_ok,
                    "pv_cells_values": pv_vals, "pv_cells_differing": pv_bad, "pv_running_sum_max_rel": float(np.nanmax(rel)),
                    "early_exit_identical": bool(same(pv, pv_full).all())}
    # speed on the benchmark workload
    inp, orc = load_set(BENCH); n = len(inp["table"])
    t_prep = best(lambda: M.prep(inp["table"], inp["tb"]), 3); p = M.prep(inp["table"], inp["tb"])
    pv = M.pv_from_acc(run_pv(p)); rel = rel_diff(pv, orc["pv"])
    t_par = best(lambda: run_pv(p), 5); t_full = best(lambda: run_pv(p, False), 5)
    run_pv1, _ = numba_cpu.build(K, parallel=False); run_pv1(p); t_one = best(lambda: run_pv1(p), 3)
    t_one_full = best(lambda: run_pv1(p, False), 3); t_orig = float(orc["seconds_pv"])
    print(f"[speed, {n:,} policies x {p['T']} years] original {t_orig:.2f}s | input preparation {t_prep:.3f}s | 1 core, no early exit {t_one_full:.4f}s | "
          f"1 core {t_one:.4f}s | all cores, no early exit {t_full:.5f}s | all cores {t_par:.5f}s")
    print(f"  speed-up including preparation: 1 core {t_orig/(t_prep+t_one):.0f}x, all cores {t_orig/(t_prep+t_par):.0f}x; kernel only: "
          f"1 core {t_orig/t_one:.0f}x, all cores {t_orig/t_par:.0f}x; PV max relative difference {np.nanmax(rel):.1e}")
    res["speed"] = {"rows": n, "years": p["T"], "original_s": t_orig, "prep_s": t_prep, "one_core_no_exit_s": t_one_full, "one_core_s": t_one,
                    "all_cores_no_exit_s": t_full, "all_cores_s": t_par, "cores": os.cpu_count(), "pv_max_rel": float(np.nanmax(rel))}
    m = M.load_original(); big = M.portfolio_sample(m, 33334); tb = M.extract_tables(m)
    t = time.perf_counter(); p = M.prep(big, tb); t_prep = time.perf_counter() - t
    t_big = best(lambda: run_pv(p), 2)
    print(f"[scale] {len(big):,} policies: {t_big:.2f}s on all cores plus {t_prep:.1f}s input preparation; original extrapolated {t_orig/n*len(big)/3600:.1f} h")
    res["scale"] = {"rows": len(big), "seconds": t_big, "prep_seconds": t_prep, "original_extrapolated_s": t_orig / n * len(big)}
    save("exact", res)

def harness():
    from exactuary import harness as H
    from exactuary.backends import numpy_vec
    K = M.KERNEL; res = {}
    # (a) every table lookup against the original's own cells
    m, P, table, tb = build_original("tables", 0); p = M.prep(table, tb); T = p["T"]; n = len(table)
    com_vals, com_bad = 0, 0
    for i, (s, r, tid) in enumerate(p["_klt"]):
        L = m.LifeTable[s, r, tid]
        for tab, cell in ((p["TAB_D"], L.Dx), (p["TAB_M"], L.Mx), (p["TAB_N"], L.Nx)):
            o = np.array([cell(x) for x in range(M.N_AGE)], dtype=np.float64)
            com_vals += o.size; com_bad += int((~same(tab[i], o)).sum())
    print(f"(a) commutation columns D, M, N for {len(p['_klt'])} product keys ({len(set(p['_klt']))} distinct life tables) x {M.N_AGE} ages: {com_vals} values, differing {com_bad}", flush=True)
    cells = ["BaseMortRate", "asmp.MortFactor", "asmp.SurrRate", "pol.ReserveNLP_Rate[PREM]", "pol.SurrCharge", "pol.CashValueRate",
             "PolsDeath", "PolsSurr", "SizeBenefitSurr"]
    o = M.capture(P, cells, T, n); tr = numpy_vec.run_trace(K, p, names=cells)
    lk_bad = {c: int((~same(tr[j], o[c])).sum()) for j, c in enumerate(cells)}
    attrs = {k: np.asarray(getattr(P if sp is None else getattr(P, sp), c)(*a)) for k, (sp, c, *a) in M.ATTR_CELLS.items()}
    at_bad = {k: int((~same(np.asarray(p["_attrs"][k], np.float64), v.astype(np.float64))).sum()) for k, v in attrs.items()}
    print(f"(a) lookups: {n:,} synthetic policies x {T} years, {len(cells)} cells, {tr.size:,} values, differing {sum(lk_bad.values())}; "
          f"{len(attrs)} per-policy inputs from prep, differing {sum(at_bad.values())}", flush=True)
    res["tables"] = {"life_tables": sorted(set(str(k) for k in p["_klt"])), "product_keys": [str(k) for k in p["_keys"]], "commutation_values": com_vals, "commutation_differing": com_bad,
                     "policies": n, "years": T, "lookup_values": int(tr.size), "lookup_differing": lk_bad, "prep_attributes_differing": at_bad}
    from exactuary import gates; gates.record("every table lookup equals the original's, in and out of range", com_bad == 0 and not any(lk_bad.values()) and not any(at_bad.values()))
    # (b) coverage and (c) seeded defects
    data = {}
    for tag, full in (("sample", "sample_300"), ("edge", "edge_1500")):
        inp, orc = load_set(full); p = M.prep(inp["table"], inp["tb"])
        data[tag] = {"p": p, "trace": np.stack([orc["t:" + nm] for nm in K.outputs])}
        labels, cov = H.coverage(K, p); missing = [(l, sorted(c)) for l, c in zip(labels, cov) if len(c) < 2]
        res[f"coverage_{tag}"] = {"seen": sum(len(c) for c in cov), "of": 2 * len(cov), "missing": missing}
        print(f"(b) coverage [{tag}, {len(inp['table'])} rows]: {sum(len(c) for c in cov)}/{2*len(cov)} branch outcomes; missing: {missing}", flush=True)
    t0 = time.time(); ms = H.mutation_scores(K, data, M.SWAPS, workers=6)
    print(f"(c) {len(ms['mutants'])} seeded defects judged in {time.time()-t0:.0f}s")
    for name, k, nn in ms["table"]: print(f"    caught {k}/{nn} = {k/nn:.1%}  <- {name}")
    print("    survivors of the strongest check:"); [print("     ", s) for s in ms["survivors"]]
    res.update({"defects": len(ms["mutants"]), "table": ms["table"], "survivors": ms["survivors"]}); save("harness", res)

def prove(path=None):
    from exactuary import symbolic as S
    from exactuary.mutate import mutants
    ss = M.symbolic_spec(path); res = {}
    for tag, K in (("reference", M.KERNEL), ("rearranged", M.KERNEL_REARRANGED)):
        o = S.prove(K, ss)
        print(f"{tag} kernel: {o['proved']}/{o['total']} obligations proved in {o['seconds']:.1f}s; refuted {[t for t, _ in o['refuted']]}; unknown {o['unknown']}", flush=True)
        res[tag] = {"proved": o["proved"], "total": o["total"], "seconds": o["seconds"],
                    "refuted": [(t, S.show_model(mdl)) for t, mdl in o["refuted"]], "unknown": o["unknown"]}
    res["formulas_projection"], res["axiomatised_projection"] = o["formulas"], o["axiomatised"]; res["axioms"] = M.AXIOMS
    print(f"Projection formulas read: {o['formulas']}; replaced in the Projection space: {o['axiomatised']}")
    lem = S.prove_lemmas(M.KERNEL, ss); res["early_exit_lemmas"] = lem
    print("early-exit lemmas:", lem, flush=True)
    ms = mutants(M.STEP_SRC, M.SWAPS); t0 = time.time(); tally, rows = S.refute_mutants(M.KERNEL, ss, ms)
    print(f"seeded defects: {tally} of {len(ms)} in {time.time()-t0:.0f}s")
    for kind, desc, v, cex in rows:
        if v != "refuted": print(f"  {v.upper()}: [{kind}] {desc}")
    res["defects"] = {"tally": tally, "rows": rows}; save("prove", res)

def precision():
    from exactuary.backends import numba_cpu, cuda
    from exactuary.errbound import run_pv_bounds
    from exactuary.spec import to_f32_src
    res = {}
    kernels = {"reference": M.KERNEL, "rearranged": M.KERNEL_REARRANGED}
    def kern(kind, ft): return kernels[kind].with_src(to_f32_src(kernels[kind].step_src)) if ft == "float32" else kernels[kind]
    inp, orc = load_set(BENCH); orc = orc["pv"]; n = len(orc)
    P = {ft: M.prep(inp["table"], inp["tb"], np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
    sub = np.arange(0, 300); psub = M.prep(inp["table"].iloc[sub], inp["tb"])      # one copy of each sample policy
    assert psub["T"] == P["float64"]["T"]
    scale = np.maximum(np.abs(orc), 1.0); bounds = {}
    for kind in kernels:
        for ft in ("float64", "float32"):
            try:
                v, e = run_pv_bounds(kern(kind, ft), psub, ft); bounds[(kind, ft)] = np.column_stack([e, e.sum(1)])
            except Exception as ex:
                bounds[(kind, ft)] = None; res[f"bound_error/{kind}/{ft}"] = repr(ex)
                print(f"  rounding bound NOT available for {kind} {ft}: {ex!r}")
    e_orig = bounds[("reference", "float64")]
    print(f"accuracy against the original ({n:,} policies x 4 present values)")
    runs = {}
    for dev in ("cpu", "gpu"):
        for kind in kernels:
            for ft in ("float64", "float32"):
                K = kern(kind, ft)
                if dev == "cpu":
                    run_pv, _ = numba_cpu.build(K, ftype=ft); runs[(dev, kind, ft)] = run_pv; pv = M.pv_from_acc(run_pv(P[ft]).astype(np.float64))
                else:
                    prepare = cuda.build(K, ft); runs[(dev, kind, ft)] = prepare; go, out = prepare(P[ft]); go(); pv = M.pv_from_acc(out.copy_to_host().astype(np.float64))
                diff = np.abs(pv - orc); rel = diff / scale
                r = {"identical_share": float(same(pv, orc).mean()), "worst_per_policy_rel": float(rel.max()),
                     "worst_per_policy_abs": float(diff.max()),
                     "portfolio_totals_rel": float((np.abs(pv.sum(0) - orc.sum(0)) / np.abs(orc.sum(0))).max())}
                if bounds[(kind, ft)] is not None and e_orig is not None:
                    bnd = bounds[(kind, ft)] + e_orig
                    r.update({"bound_worst_rel": float((bnd / scale[sub]).max()), "bound_median_rel": float(np.median(bnd / scale[sub])),
                              "observed_within_bound": bool((diff[sub] <= (2.0 if dev == "gpu" else 1.0) * bnd + 1e-300).all())})
                res[f"{dev}/{kind}/{ft}"] = r
                print(f"  {dev} {kind:10s} {ft}: worst per-policy {r['worst_per_policy_rel']:.1e} (abs {r['worst_per_policy_abs']:.3g}) | portfolio totals {r['portfolio_totals_rel']:.1e} | "
                      + (f"bound worst {r['bound_worst_rel']:.1e}, median {r['bound_median_rel']:.1e} | within bound: {r['observed_within_bound']}" if "bound_worst_rel" in r else "no bound"), flush=True)
    m = M.load_original(); big = M.portfolio_sample(m, 33334); tb = M.extract_tables(m)
    P = {ft: M.prep(big, tb, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
    p = P["float64"]; years = int(np.minimum(p["term"] + 2, p["T"]).sum())
    print(f"speed: {len(big):,} policies, about {years/1e9:.2f}bn policy-years with early exit ({len(big)*p['T']/1e9:.2f}bn without)")
    for (dev, kind, ft), f in runs.items():
        if dev == "cpu": f(P[ft]); t = best(lambda: f(P[ft]))
        else: go, out = f(P[ft]); go(); t = best(go)
        res[f"{dev}/{kind}/{ft}"]["seconds_10M"] = t; print(f"  {dev} {kind:10s} {ft}: {t:.3f}s", flush=True)
    res["rows_10M"] = len(big); save("precision", res)

def lean(path=None):
    from exactuary import lean as L
    exe = os.environ.get("LEAN", "lean"); res = {}
    n, failed = L.run_lean(L.GLUE, exe); res["glue"] = {"theorems": n, "failed": failed}
    print(f"induction theorems: {n - len(failed)}/{n} checked", flush=True)
    # "as_exported": the obligations exactly as the library produces them. "zero_folded": the same with the
    # model-side simplification described in model.symbolic_spec (boolean x literal 0 = 0).
    for variant, fold in (("as_exported", False), ("zero_folded", True)):
        ss = M.symbolic_spec(path, fold_zero=fold)
        for tag, K in (("reference", M.KERNEL), ("rearranged", M.KERNEL_REARRANGED)):
            name = f"Obligations_{tag}" + ("" if not fold else "_zero_folded")
            fn = os.path.join(RES, name + ".lean"); L.export(K, ss, fn, tag, composite=M.COMPOSITE)
            t0 = time.time(); n, failed = L.run_lean(fn, exe)
            print(f"[{variant}] {tag} kernel: {n - len(failed)}/{n} obligations checked by Lean in {time.time()-t0:.0f}s; failed: {failed}", flush=True)
            res[f"{variant}/{tag}"] = {"theorems": n, "failed": failed}
            if fold:
                from exactuary import symbolic as S
                o = S.prove(K, ss); res[f"{variant}/{tag}"]["z3"] = f"{o['proved']}/{o['total']}"
                print(f"    Z3 on the same obligations: {o['proved']}/{o['total']} proved, refuted {len(o['refuted'])}, unknown {len(o['unknown'])}", flush=True)
    save("lean", res)

def baseline():
    """lifelib simplelife: the same model written one policy at a time. (1) Same values as fastlife for
    the converted cells, on a small portfolio? (2) Seconds per policy, extrapolated to the benchmark."""
    import modelx as mx
    res = {}
    inp, orc = load_set("sample_300"); T = int(orc["T"])
    s = mx.read_model(M.lib_path("simplelife")); S = s.Projection
    ids = list(range(1, 301, 10))                     # 30 policies: 10 term, 10 whole life, 10 endowment, both sexes
    names = [n for n in M.OUTPUTS if n not in M.DERIVED]
    def sl_value(Sp, name, t):
        if name.startswith("pol."):
            c = M.parse_cell(name[4:]); return getattr(Sp.Policy, c[0])(*c[1:], t)
        if name.startswith("asmp."): return getattr(Sp.Assumptions, name[5:])(t)
        if name == "BaseMortRate": return Sp.Assumptions.BaseMortRate(Sp.AttAge(t))     # simplelife keeps it in Assumptions, by age
        return getattr(Sp, name)(t)
    # simplelife projects each policy to its own last year (last_t); beyond it some of its cells raise
    # (mortality lookup past the end of the table), so each policy is compared for years 0..last_t.
    last = orc["a:last_t"].astype(int); cmp = {}; vals = 0; raised = 0
    for name in names:
        bad = []; nbad = 0
        for i in ids:
            for t in range(last[i - 1] + 1):
                try: a = float(sl_value(S[i], name, t))
                except Exception: raised += 1; continue
                o = float(orc["t:" + name][t, i - 1]); vals += 1
                if not same(np.float64(a), np.float64(o)):
                    nbad += 1; bad.append(abs(a - o) / max(abs(o), 1.0))
        if nbad: cmp[name] = {"differing": nbad, "max_rel": float(max(bad))}
    print(f"simplelife vs fastlife, {len(ids)} policies, years 0..last_t, {len(names)} cells = {vals:,} values ({raised} raised in simplelife): {len(cmp)} cells differ")
    for k, v in cmp.items(): print("   ", k, v)
    pv_s = np.array([[float(getattr(S[i], c)(0)) for c in M.PV_CELLS] for i in ids]); pv_f = orc["pv"][[i - 1 for i in ids]]
    pv_rel = rel_diff(pv_s, pv_f)
    print(f"  PV cells at time 0 ({pv_s.size} values): identical {int(same(pv_s, pv_f).sum())}, max relative difference {pv_rel.max():.2e}")
    res["same_values"] = {"policies": len(ids), "years": "0..last_t of each policy", "cells": len(names), "values": vals, "raised_in_simplelife": raised, "cells_differing": cmp,
                          "pv_values": int(pv_s.size), "pv_identical": int(same(pv_s, pv_f).sum()), "pv_max_rel": float(pv_rel.max()),
                          "pv_differing_policies": [ids[i] for i in np.unique(np.argwhere(~same(pv_s, pv_f))[:, 0])]}
    # timing: a fresh model, the reported result only (PV_NetCashflow at time 0), all 300 sample policies
    s2 = mx.read_model(M.lib_path("simplelife")); S2 = s2.Projection
    t0 = time.perf_counter(); pv = np.array([S2[i].PV_NetCashflow(0) for i in range(1, 301)]); sec = time.perf_counter() - t0
    all_rel = rel_diff(pv, orc["pv"][:, 3])
    nb = len(np.load(f"{DATA}/oracle_{BENCH}.npz")["pv"]); fast = float(np.load(f"{DATA}/oracle_{BENCH}.npz")["seconds_pv"])
    print(f"simplelife: 300 policies in {sec:.2f}s = {sec/300*1000:.2f} ms per policy; extrapolated to {nb:,} policies {sec/300*nb:.0f}s "
          f"(fastlife measured {fast:.1f}s), to 10,000,200 policies {sec/300*10_000_200/3600:.1f} h")
    print(f"  PV_NetCashflow(0), all 300: identical to fastlife {int(same(pv, orc['pv'][:, 3]).sum())}/300, max relative difference {all_rel.max():.2e}")
    res["timing"] = {"policies": 300, "seconds": sec, "seconds_per_policy": sec / 300, "benchmark_rows": int(nb),
                     "extrapolated_benchmark_s": sec / 300 * nb, "fastlife_benchmark_s": fast, "extrapolated_10M_s": sec / 300 * 10_000_200,
                     "pv_net_identical_of_300": int(same(pv, orc["pv"][:, 3]).sum()), "pv_net_max_rel": float(all_rel.max())}
    save("baseline", res)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd in ("prove", "lean"): {"prove": prove, "lean": lean}[cmd](arg)
    elif cmd in ("oracle", "exact", "harness", "precision", "baseline"): globals()[cmd]()
    else: print(__doc__)
    from exactuary import gates
    gates.finish({"exact": ['exact comparison', 'early exit', 'present values', 'inputs'], "prove": ["proof of", "early-exit obligations"], "lean": ["Lean check"], "harness": ["every table lookup"]}.get(cmd, ()))
