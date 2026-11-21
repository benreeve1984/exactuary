"""Reproduce every result for lifelib savings/CashValue_ME_EX2 (dynamic lapse).

    python run.py oracle     run the original model (every sim_id) and store every intermediate value
    python run.py exact      compiled CPU kernels: exact comparison with the original, and timings
    python run.py harness    how good is the checking: exhaustive tables, coverage, seeded defects
    python run.py prove      symbolic proof of equivalence for all policies and months (Z3), per sim_id
    python run.py provemut   which seeded defects the proof refutes
    python run.py precision  GPU and single precision: timings, accuracy, rounding bounds
    python run.py lean       re-check the proof obligations with Lean (set LEAN=/path/to/lean)
"""
import sys, os, pickle, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import model as M
from exactuary.compare import same, compare_trace
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)
# name, scenarios, sim_ids, full trace?
SETS = [("sample1", 10000, M.SIM_IDS, True), ("money", 500, M.SIM_IDS, True), ("edge", 4, M.SIM_IDS, True),
        ("wild", 4, (4, 5), True), ("edgevar", 4, (4, 5), True), ("speed", 10000, (4, 5), False)]
N_SCEN = {s[0]: s[1] for s in SETS}
def save(name, obj): json.dump(obj, open(os.path.join(RES, name + ".json"), "w"), indent=1, default=str)
def tables(name=None): return pickle.load(open(f"{DATA}/tables{'_var' if name == 'edgevar' else ''}.pkl", "rb"))
def vary_tables(P):
    """For the 'edgevar' set only: the edge portfolio under perturbed assumption tables, set through the
    model's own data objects, to distinguish defects that the shipped tables cannot (mortality identical
    across durations and starting at age 0; last surrender-charge row zero)."""
    mt = P.mort_table.loc[18:].copy()
    for j, c in enumerate(mt.columns): mt[c] = np.minimum(mt[c] * (1 + 0.05 * j), 1.0)
    sc = P.surr_charge_table.copy(); sc.loc[10] = [0.005, 0.004, 0.003]
    P.mort_table = mt; P.surr_charge_table = sc
def load_inputs(name): return pickle.load(open(f"{DATA}/inputs_{name}.pkl", "rb"))
def load_oracle(name, sim): return np.load(f"{DATA}/oracle_{name}_sim{sim}.npz")
def best(f, k=3):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
def build_mp(name, m, tb):
    if name == "sample1": return M.portfolio_sample1(m)
    if name in ("money", "speed"): return M.portfolio_money(m)
    if name in ("edge", "edgevar"): return M.portfolio_edge(600, tb["mort_last_age"])
    if name == "wild": return M.portfolio_edge(300, tb["mort_last_age"], wild=True, seed=8)
def rel_diff(a, b):
    """Largest relative difference where two arrays are not identical (NaN against NaN counts as identical);
    a NaN against a number counts as infinite."""
    eq = same(a, b)
    if eq.all(): return 0.0
    with np.errstate(all="ignore"):
        r = np.abs(a - b)[~eq] / np.maximum(np.abs(b[~eq]), 1e-300)
    return float("inf") if np.isnan(r).any() else float(r.max())

def oracle(only=None):
    m = M.load_original(); tb0 = M.extract_tables(m.Projection); pickle.dump(tb0, open(f"{DATA}/tables.pkl", "wb"))
    vary_tables(m.Projection); pickle.dump(M.extract_tables(m.Projection), open(f"{DATA}/tables_var.pkl", "wb")); m.close()
    res = json.load(open(f"{RES}/oracle.json")) if only else {}
    for name, S, sims, trace in SETS:
        if only and name != only: continue
        tb = tables(name)
        for sim in sims:
            m = M.load_original(); mp = build_mp(name, m, tb); P = M.configure(m, mp, S, sim)
            if name == "edgevar": vary_tables(P)
            assert M.read_flags(P) == M.flags_of(tb, sim)
            Z = P.std_norm_rand().to_numpy().reshape(S, M.N_RAND_T)
            assert (Z == M.normals(S)).all()
            idx = P.model_point().index
            assert (idx.get_level_values(0) == np.repeat(mp.index.to_numpy(), S)).all() and (idx.get_level_values(1) == np.tile(np.arange(1, S + 1), len(mp))).all()
            r = M.run_original(P, trace)
            pickle.dump({"mp": mp, "S": S, "Z": Z}, open(f"{DATA}/inputs_{name}.pkl", "wb"))
            out = {"pv": r["pv"], "T": r["T"], "seconds_pv": r["seconds_pv"], "seconds_mc": r["seconds_mc"], "monte_carlo": r["monte_carlo"]}
            info = {"points": len(mp), "scenarios": S, "rows": len(mp) * S, "months": r["T"], "original_result_pv_s": r["seconds_pv"],
                    "original_monte_carlo_s": r["seconds_mc"], "rows_with_nonfinite_pv": int((~np.isfinite(r["pv"])).any(1).sum())}
            if trace:
                out.update({"t:" + k: v for k, v in r["trace"].items()})
                out["inflation_factor"], out["disc_factors"] = r["inflation_factor"], r["disc_factors"]
                info["trace_values"] = int(sum(v.size for v in r["trace"].values()))
                info["trace_nonfinite_values"] = int(sum((~np.isfinite(v)).sum() for v in r["trace"].values()))
            np.savez(f"{DATA}/oracle_{name}_sim{sim}.npz", **out); res[f"{name}/sim{sim}"] = info
            print(f"{name} sim {sim}: {info}", flush=True)
            m.close()
    save("oracle", res)

def exact():
    from exactuary.backends import numba_cpu
    res = {}
    built = {id(K): numba_cpu.build(K) for K in (M.KERNEL_STATIC, M.KERNEL_DYNAMIC)}
    serial = {id(K): numba_cpu.build(K, parallel=False) for K in (M.KERNEL_STATIC, M.KERNEL_DYNAMIC)}
    for name, S, sims, trace in SETS:
        if not trace: continue
        inp = load_inputs(name); tb = tables(name)
        for sim in sims:
            flags = M.flags_of(tb, sim); K = M.kernel_for(flags); run_pv, run_trace = built[id(K)]
            orc = load_oracle(name, sim); p = M.prep(inp["mp"], S, inp["Z"], tb, flags)
            assert p["T"] == int(orc["T"]) and (p["infl"] == orc["inflation_factor"]).all() and (p["DISC"] == orc["disc_factors"]).all()
            n = len(p["age0"]); print(f"[{name} sim {sim}: {'dynamic' if K is M.KERNEL_DYNAMIC else 'static'} kernel] {n:,} rows x {p['T']} months")
            tr = run_trace(p); bad, total = compare_trace(tr, {k: orc["t:" + k] for k in K.outputs}, K.outputs)
            from exactuary import gates; gates.early_exit_agrees(run_pv, p)
            try:        # the single-threaded build of the same kernel (Numba raises on a zero divisor outside prange)
                tr1 = serial[id(K)][1](p); serial_same = bool(all(same(tr1[k], tr[k]).all() for k in range(K.n_out)))
            except Exception as ex: serial_same = "raised " + repr(ex)
            from exactuary import gates; gates.record("one-core build equals the all-cores build", serial_same is True, str(serial_same))
            opv = orc["pv"]
            acc = np.column_stack([tr[K.ix(k)].T @ p["DISC"] for k in K.accumulate]); pv_mat = M.reported(acc)
            pv_run = M.reported(run_pv(p, False)); pv_exit = M.reported(run_pv(p, True))
            mc_o = orc["monte_carlo"]; mc_mat = M.monte_carlo_from(pv_mat, inp["mp"], S); mc_run = M.monte_carlo_from(pv_run, inp["mp"], S)
            nan_rows = (~np.isfinite(opv)).any(1)
            r = {"rows": n, "months": p["T"], "kernel": "dynamic" if K is M.KERNEL_DYNAMIC else "static",
                 "values": total, "outputs": len(K.outputs), "outputs_differing": len(bad),
                 "differing": {k: {"values": v[0], "max_rel": v[1], "first_month_row": v[2]} for k, v in bad.items()},
                 "values_differing": int(sum(v[0] for v in bad.values())),
                 "serial_build_identical_to_parallel": serial_same,
                 "original_rows_with_nonfinite_pv": int(nan_rows.sum()),
                 "pv_identical_same_summation": bool(same(pv_mat, opv).all()), "pv_same_summation_max_rel": rel_diff(pv_mat, opv),
                 "pv_running_sum_max_rel": rel_diff(pv_run, opv),
                 "pv_running_sum_max_rel_finite_rows": rel_diff(pv_run[~nan_rows], opv[~nan_rows]),
                 "pv_early_exit_identical_to_no_exit": bool(same(pv_exit, pv_run).all()),
                 "pv_early_exit_rows_differing_from_no_exit": int((~same(pv_exit, pv_run)).any(1).sum()),
                 "pv_early_exit_max_rel_finite_rows": rel_diff(pv_exit[~nan_rows], opv[~nan_rows]),
                 "monte_carlo_identical_same_summation": bool(same(mc_mat, mc_o).all()), "monte_carlo_running_sum_max_rel": rel_diff(mc_run, mc_o)}
            from exactuary import gates; gates.record("present values identical when summed as the original sums them", r["pv_identical_same_summation"] and r["monte_carlo_identical_same_summation"])
            print(f"  present values identical when summed as the original sums them: {r['pv_identical_same_summation']}; running sum max rel "
                  f"{r['pv_running_sum_max_rel']:.1e} (finite rows {r['pv_running_sum_max_rel_finite_rows']:.1e}); early exit changes "
                  f"{r['pv_early_exit_rows_differing_from_no_exit']} rows; monte_carlo identical: {r['monte_carlo_identical_same_summation']}; "
                  f"original rows with NaN/inf: {r['original_rows_with_nonfinite_pv']}; serial build same: {serial_same}", flush=True)
            res[f"{name}/sim{sim}"] = r
    save("exact", res)

def speed():
    from exactuary.backends import numba_cpu
    tb = tables(); res = json.load(open(f"{RES}/exact.json"))
    built = {id(K): numba_cpu.build(K) for K in (M.KERNEL_STATIC, M.KERNEL_DYNAMIC)}
    serial = {id(K): numba_cpu.build(K, parallel=False) for K in (M.KERNEL_STATIC, M.KERNEL_DYNAMIC)}
    inp = load_inputs("speed"); S = N_SCEN["speed"]
    m = M.load_original(); mps = M.portfolio_scale(1000); m.close(); S_big = 10000; Zb = M.normals(S_big)
    for sim in (4, 5):
        flags = M.flags_of(tb, sim); K = M.kernel_for(flags); run_pv, _ = built[id(K)]; run_pv1, _ = serial[id(K)]
        orc = load_oracle("speed", sim); p = M.prep(inp["mp"], S, inp["Z"], tb, flags); n = len(p["age0"])
        pv = M.reported(run_pv(p)); mc = M.monte_carlo_from(pv, inp["mp"], S)
        t_par = best(lambda: run_pv(p), 5); t_full = best(lambda: run_pv(p, False), 5)
        run_pv1(p); t_one = best(lambda: run_pv1(p), 2); t_one_full = best(lambda: run_pv1(p, False), 2)
        t_orig = float(orc["seconds_pv"]) + float(orc["seconds_mc"])
        t0 = time.perf_counter(); M.prep(inp["mp"], S, inp["Z"], tb, flags); t_prep = time.perf_counter() - t0
        r = {"rows": n, "months": p["T"], "original_s": t_orig, "one_core_no_exit_s": t_one_full, "one_core_s": t_one,
             "all_cores_no_exit_s": t_full, "all_cores_s": t_par, "prep_s": t_prep, "cores": os.cpu_count(),
             "pv_running_sum_max_rel": rel_diff(pv, orc["pv"]), "pv_values_identical_share": float(same(pv, orc["pv"]).mean()),
             "monte_carlo_max_rel": rel_diff(mc, orc["monte_carlo"])}
        print(f"[speed sim {sim}, {n:,} rows x {p['T']} months] original {t_orig:.2f}s | 1 core, no exit {t_one_full:.3f}s ({t_orig/t_one_full:.0f}x) | "
              f"1 core {t_one:.3f}s | all cores, no exit {t_full:.4f}s ({t_orig/t_full:.0f}x) | all cores {t_par:.4f}s | prep {t_prep:.3f}s | "
              f"pv max rel {r['pv_running_sum_max_rel']:.1e}, monte_carlo max rel {r['monte_carlo_max_rel']:.1e}", flush=True)
        res[f"speed/sim{sim}"] = r
        t0 = time.perf_counter(); pb = M.prep(mps, S_big, Zb, tb, flags); t_prep = time.perf_counter() - t0
        run_pv(pb); t_big = best(lambda: run_pv(pb), 2); nb = len(pb["age0"])
        print(f"[scale sim {sim}] {nb:,} rows x {pb['T']} months: {t_big:.2f}s on all cores (plus {t_prep:.1f}s input preparation); "
              f"original extrapolated {t_orig/n*nb/60:.0f} min", flush=True)
        res[f"scale/sim{sim}"] = {"rows": nb, "months": pb["T"], "seconds": t_big, "prep_seconds": t_prep, "original_extrapolated_s": t_orig / n * nb}
        del pb
    save("exact", res)

def harness():
    import pandas as pd
    from exactuary import harness as H
    from exactuary.backends import numpy_vec
    tb = tables(); res = {}
    # (a) every table lookup, in and out of range, against the original's own lookup cells
    rows = [(spec, age - d, 12 * d + (age % 12 if d >= 0 else 0), 200_000 + 997 * age) for spec in "ABCD" for age in range(0, 211) for d in range(-3, 16)]
    mp = pd.DataFrame(rows, columns=["spec_id", "age_at_entry", "duration_mth", "av_pp_init"])
    for c, v in (("sex", "M"), ("policy_term", 10), ("policy_count", 100), ("sum_assured", 500_000), ("premium_pp", 1000), ("accum_prem_init_pp", 0)): mp[c] = v
    mp.index = pd.RangeIndex(1, len(mp) + 1, name="point_id")
    cells = ["mort_rate", "mort_rate_mth", "lapse_rate", "surr_charge_rate", "csv_pp", "pols_lapse", "pols_death", "coi_rate", "maint_fee_pp"]
    for sim in M.SIM_IDS:
        m = M.load_original(); P = M.configure(m, mp, 1, sim); flags = M.flags_of(tb, sim); K = M.kernel_for(flags)
        p = M.prep(mp, 1, M.normals(1), tb, flags, clip_T=True); p["T"] = 1; tr = numpy_vec.run_trace(K, p, names=cells)
        ok = {c: bool(same(tr[j, 0], np.broadcast_to(np.asarray(getattr(P, c)(0), dtype=np.float64), tr[j, 0].shape)).all()) for j, c in enumerate(cells)}
        print(f"(a) exhaustive tables, sim {sim}: {len(mp):,} synthetic policies, {len(cells)} cells identical: {all(ok.values())} {[c for c, v in ok.items() if not v]}", flush=True)
        res[f"tables/sim{sim}"] = {"policies": len(mp), "identical": all(ok.values()), "cells": ok}; m.close()
        from exactuary import gates; gates.record(f"every table lookup equals the original's, in and out of range (sim {sim})", all(ok.values()))
    # (b) coverage and (c) seeded defects, for each kernel
    plan = {"static": (M.KERNEL_STATIC, [("sample", "money", 4, slice(0, None, 7)), ("edge", "edge", 4, slice(None)), ("edgevar", "edgevar", 4, slice(None)),
                                         ("edge_sim1", "edge", 1, slice(None)), ("edge_sim3", "edge", 3, slice(None))]),
            "dynamic": (M.KERNEL_DYNAMIC, [("sample", "money", 5, slice(0, None, 7)), ("edge", "edge", 5, slice(None)), ("edgevar", "edgevar", 5, slice(None))])}
    for kname, (K, sets) in plan.items():
        data = {}; union = None
        for tag, name, sim, pick in sets:
            inp = load_inputs(name); orc = load_oracle(name, sim); tbn = tables(name); pf = M.prep(inp["mp"], N_SCEN[name], inp["Z"], tbn, M.flags_of(tbn, sim))
            idx = np.arange(len(pf["age0"]))[pick]; p = M.subset(pf, idx)
            data[tag] = {"p": p, "trace": np.stack([orc["t:" + n][:, idx] for n in K.outputs])}
            assert np.isfinite(data[tag]["trace"]).all(), "the seeded-defect harness cannot judge portfolios with NaN"
            labels, cov = H.coverage(K, p); union = cov if union is None else [a | b for a, b in zip(union, cov)]
            res[f"coverage/{kname}/{tag}"] = {"seen": sum(len(c) for c in cov), "of": 2 * len(cov), "missing": [l for l, c in zip(labels, cov) if len(c) < 2]}
            print(f"(b) coverage [{kname} kernel, {tag}, {len(idx)} rows]: {sum(len(c) for c in cov)}/{2*len(cov)} branch outcomes; "
                  f"missing: {[l for l, c in zip(labels, cov) if len(c) < 2]}", flush=True)
        res[f"coverage/{kname}/all"] = {"seen": sum(len(c) for c in union), "of": 2 * len(union), "missing": [l for l, c in zip(labels, union) if len(c) < 2]}
        print(f"(b) coverage [{kname} kernel, all portfolios]: {res[f'coverage/{kname}/all']}")
        t0 = time.time(); ms = H.mutation_scores(K, data, M.SWAPS, workers=6)
        print(f"(c) {kname} kernel: {len(ms['mutants'])} seeded defects judged in {time.time()-t0:.0f}s")
        for name, k, n in ms["table"]: print(f"    caught {k}/{n} = {k/n:.1%}  <- {name}")
        print("    survivors of the strongest check:"); [print("     ", s) for s in ms["survivors"]]; sys.stdout.flush()
        res[f"defects/{kname}"] = {"defects": len(ms["mutants"]), "table": ms["table"], "survivors": ms["survivors"]}
        save("harness", res)

def prove(path=None):
    from exactuary import symbolic as S
    tb = tables(); res = {}
    for sim in M.SIM_IDS:
        flags = M.flags_of(tb, sim); K = M.kernel_for(flags); ss = M.symbolic_spec(flags, path)
        o = S.prove(K, ss)
        lem = S.prove_lemmas(K, ss)
        print(f"sim {sim} ({'dynamic' if K is M.KERNEL_DYNAMIC else 'static'} kernel): {o['proved']}/{o['total']} obligations proved in {o['seconds']:.1f}s; "
              f"refuted {[t for t, _ in o['refuted']]}; unknown {o['unknown']}; formulas executed {o['formulas']}; axiomatised {o['axiomatised']}; "
              f"early-exit lemma: {'proved' if all(v == 'proved' for _, v in lem) else [x for x in lem if x[1] != 'proved']}", flush=True)
        res[f"sim{sim}"] = {"flags": flags, "kernel": "dynamic" if K is M.KERNEL_DYNAMIC else "static", "proved": o["proved"], "total": o["total"],
                            "seconds": o["seconds"], "refuted": [(t, S.show_model(mdl)) for t, mdl in o["refuted"]], "unknown": o["unknown"],
                            "formulas": o["formulas"], "axiomatised": o["axiomatised"],
                            "early_exit_lemma": {"goals": len(lem), "proved": sum(v == "proved" for _, v in lem), "not_proved": [x for x in lem if x[1] != "proved"]}}
        save("prove", res)

def provemut(path=None):
    from exactuary import symbolic as S
    from exactuary.mutate import mutants
    tb = tables(); res = {}
    for sim in (5, 4):
        flags = M.flags_of(tb, sim); K = M.kernel_for(flags); ss = M.symbolic_spec(flags, path)
        ms = mutants(K.step_src, M.SWAPS); t0 = time.time(); tally, rows = S.refute_mutants(K, ss, ms)
        print(f"sim {sim}: seeded defects: {tally} of {len(ms)} in {time.time()-t0:.0f}s")
        for kind, desc, v, cex in rows:
            if v != "refuted": print(f"  {v.upper()}: [{kind}] {desc}")
        sys.stdout.flush()
        res[f"sim{sim}"] = {"tally": tally, "defects": len(ms), "seconds": time.time() - t0, "not_refuted": [(k, d, v) for k, d, v, c in rows if v != "refuted"], "rows": rows}
        save("provemut", res)

def precision():
    from exactuary.backends import numba_cpu, cuda
    from exactuary.errbound import run_pv_bounds
    from exactuary.spec import to_f32_src
    tb = tables(); res = {}; inp = load_inputs("speed"); S = N_SCEN["speed"]
    mps = M.portfolio_scale(1000); S_big = 10000; Zb = M.normals(S_big)
    def kern(K, ft): return K.with_src(to_f32_src(K.step_src)) if ft == "float32" else K
    def bound16(e):
        e16 = M.reported(e); e16[:, 8] = e[:, [0, 6, 8, 4, 5, 7]].sum(1); return e16
    for sim in (5, 4):
        flags = M.flags_of(tb, sim); K0 = M.kernel_for(flags); orc = load_oracle("speed", sim); opv = orc["pv"]; omc = orc["monte_carlo"]
        P = {ft: M.prep(inp["mp"], S, inp["Z"], tb, flags, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
        n = len(P["float64"]["age0"]); sub = np.arange(0, n, 18); psub = M.subset(P["float64"], sub)
        scale = np.maximum(np.abs(opv), 1.0); bounds = {}
        for ft in ("float64", "float32"):
            try:
                v, e = run_pv_bounds(kern(K0, ft), psub, ft); bounds[ft] = bound16(e)
            except Exception as ex:
                bounds[ft] = None; res[f"sim{sim}/bound_error/{ft}"] = repr(ex)[:300]; print(f"  rounding bound, {ft}: FAILED {ex!r}"[:300])
        print(f"sim {sim}: accuracy against the original ({n:,} rows x {opv.shape[1]} present values)", flush=True)
        runs = {}
        for dev in ("cpu", "gpu"):
            for ft in ("float64", "float32"):
                K = kern(K0, ft)
                if dev == "cpu":
                    run_pv, _ = numba_cpu.build(K, ftype=ft); runs[(dev, ft)] = run_pv; pv = M.reported(run_pv(P[ft]).astype(np.float64))
                else:
                    prepare = cuda.build(K, ft); runs[(dev, ft)] = prepare; go, out = prepare(P[ft]); go(); pv = M.reported(out.copy_to_host().astype(np.float64))
                diff = np.abs(pv - opv); rel = diff / scale; mc = M.monte_carlo_from(pv, inp["mp"], S)
                if (dev, ft) == ("cpu", "float64"): ref_cpu = pv
                r = {"identical_share": float(same(pv, opv).mean()), "worst_per_policy_rel": float(rel.max()),
                     "worst_per_policy_rel_by_pv": dict(zip(M.PV_ALL, [float(x) for x in rel.max(0)])),
                     "portfolio_totals_rel": float((np.abs(pv.sum(0) - opv.sum(0)) / np.abs(opv.sum(0))).max()),
                     "monte_carlo_max_rel": rel_diff(mc, omc), "monte_carlo_identical": bool(same(mc, omc).all()),
                     "monte_carlo_rel_by_column": dict(zip(["GMDB", "GMAB", "GMxB Total", "PV Fees", "Coverage Ratio"], [float(x) for x in np.abs(mc - omc).max(0) / np.abs(omc).max(0)])),
                     # same summation order as the CPU float64 kernel, so this isolates the platform (pow routine, fused multiply-add) or the precision
                     "vs_cpu_float64_kernel_identical_share": float(same(pv, ref_cpu).mean()), "vs_cpu_float64_kernel_max_rel": float((np.abs(pv - ref_cpu) / scale).max())}
                if bounds[ft] is not None and bounds["float64"] is not None:
                    bnd = bounds[ft] + bounds["float64"]
                    with np.errstate(all="ignore"): ratio = np.where(diff[sub] == 0, 0, diff[sub] / np.maximum(bnd, 1e-300))
                    r.update({"bound_worst_rel": float((bnd / scale[sub]).max()), "bound_median_rel": float(np.median(bnd / scale[sub])),
                              "observed_over_bound_max": float(ratio.max()),
                              "observed_within_bound": bool((diff[sub] <= (2.0 if dev == "gpu" else 1.0) * bnd + 1e-300).all())})
                res[f"sim{sim}/{dev}/{ft}"] = r
                print(f"  {dev} {ft}: identical {r['identical_share']:.1%} | worst per-policy {r['worst_per_policy_rel']:.1e} | portfolio totals {r['portfolio_totals_rel']:.1e} | "
                      f"monte_carlo {r['monte_carlo_max_rel']:.1e} | bound worst {r.get('bound_worst_rel', float('nan')):.1e}, median {r.get('bound_median_rel', float('nan')):.1e} | "
                      f"observed/bound max {r.get('observed_over_bound_max', float('nan')):.2f} | within bound: {r.get('observed_within_bound')}", flush=True)
        del P
        Pb = {ft: M.prep(mps, S_big, Zb, tb, flags, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
        nb = len(Pb["float64"]["age0"]); print(f"  speed: {nb:,} policy-scenarios x {Pb['float64']['T']} months")
        ref = None
        for (dev, ft), f in runs.items():
            if dev == "cpu": pv = f(Pb[ft]); t = best(lambda: f(Pb[ft]))
            else: go, out = f(Pb[ft]); go(); t = best(go); pv = out.copy_to_host()
            pv = M.reported(pv.astype(np.float64)); ref = pv if ref is None else ref
            tot = float((np.abs(pv.sum(0) - ref.sum(0)) / np.abs(ref.sum(0))).max())
            res[f"sim{sim}/{dev}/{ft}"].update({"seconds_10M": t, "totals_rel_vs_cpu_float64_10M": tot})
            print(f"  {dev} {ft}: {t:.3f}s; portfolio totals vs cpu float64: {tot:.1e}", flush=True)
        res[f"sim{sim}/rows_10M"] = nb; del Pb; save("precision", res)

def powcheck():
    """How far do the power routines behind `fpow` differ from NumPy's (which the original uses), on the
    arguments that actually occur: x = 1 - lapse_rate from the original's dynamic-lapse traces, y = 1/12."""
    import math, numba as nb
    from numba import cuda
    res = {}
    x = np.concatenate([load_oracle(n, 5)["t:lapse_rate"].ravel() for n in ("sample1", "money", "edge", "wild")]); x = 1 - x
    @nb.njit
    def cpu(x, y, o):
        for i in range(len(x)): o[i] = x[i] ** y
    @cuda.jit
    def gpu(x, y, o):
        i = cuda.grid(1)
        if i < x.shape[0]: o[i] = math.pow(x[i], y)
    for ft in (np.float64, np.float32):
        xs = x.astype(ft); y = ft(1 / 12)
        with np.errstate(all="ignore"): ref = np.power(xs, y)
        a = np.empty_like(xs); cpu(xs, y, a)
        d = cuda.to_device(np.empty_like(xs)); gpu[(len(xs) + 255) // 256, 256](cuda.to_device(xs), y, d); cuda.synchronize(); g = d.copy_to_host()
        r = {"arguments": int(len(xs)), "nan_results_numpy": int(np.isnan(ref).sum())}
        for tag, v in (("numba_cpu", a), ("cuda", g)):
            ne = ~same(v, ref); fin = ne & np.isfinite(v) & np.isfinite(ref)
            ulps = np.abs(v[fin].view(np.int64 if ft is np.float64 else np.int32).astype(np.int64) - ref[fin].view(np.int64 if ft is np.float64 else np.int32).astype(np.int64))
            r[tag] = {"differing": int(ne.sum()), "differing_share": float(ne.mean()), "max_ulps": int(ulps.max()) if len(ulps) else 0,
                      "nan_or_inf_mismatch": int((ne & ~fin).sum())}
        res[ft.__name__] = r; print(ft.__name__, r, flush=True)
    save("powcheck", res)

def lean(path=None):
    from exactuary import lean as L
    exe = os.environ.get("LEAN", "lean"); tb = tables(); res = {}
    n, failed = L.run_lean(L.GLUE, exe); res["glue"] = {"theorems": n, "failed": failed}
    print(f"induction theorems: {n - len(failed)}/{n} checked", flush=True)
    for sim in (5, 4, 3, 2, 1):
        flags = M.flags_of(tb, sim); K = M.kernel_for(flags); ss = M.symbolic_spec(flags, path)
        fn = os.path.join(RES, f"Obligations_sim{sim}.lean"); L.export(K, ss, fn, f"sim{sim}", composite=M.COMPOSITE)
        t0 = time.time(); n, failed = L.run_lean(fn, exe)
        print(f"sim {sim}: {n - len(failed)}/{n} obligations checked by Lean in {time.time()-t0:.0f}s; failed: {failed}", flush=True)
        res[f"sim{sim}"] = {"theorems": n, "failed": failed, "seconds": time.time() - t0}; save("lean", res)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd in ("prove", "provemut", "lean", "oracle"): globals()[cmd](arg)
    elif cmd in ("exact", "powcheck", "speed", "harness", "precision"): globals()[cmd]()
    else: print(__doc__)
    from exactuary import gates
    gates.finish({"exact": ['exact comparison', 'early exit', 'present values'], "prove": ["proof of", "early-exit obligations"], "lean": ["Lean check"], "harness": ["every table lookup"]}.get(cmd, ()))
