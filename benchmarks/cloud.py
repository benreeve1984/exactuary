"""A self-contained timing of the savings model for any machine, for example a cloud instance.

It builds the same workload as benchmarks/timings.py (10,000 lifelib policies under 1,000 scenarios,
ten million rows), then times the fast model on all processor cores and, where a GPU is present, on
the GPU in double, single and mixed precision. Mixed precision does each period's arithmetic in
single precision and accumulates the discounted totals in double precision. It also reports how far
each GPU result is from the double-precision processor result.

    python benchmarks/cloud.py [--rows-scenarios 1000] [--no-gpu] [--threads 8,16,32]

Prints one line starting with RESULT followed by JSON.
"""
import sys, os, time, json, platform, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples", "lifelib_savings"))

def best(f, k):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)

def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "cpu model")): return line.split(":", 1)[1].strip()
    except OSError: pass
    return platform.processor() or platform.machine()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=1000)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--threads", default="")
    a = ap.parse_args()
    import numba as nb
    import model as M
    from exactuary.backends import numba_cpu
    from exactuary.spec import to_f32_src
    m = M.load_original(); tb = M.extract_tables(m.Projection)
    mp, scen = M.portfolio_sample(m, 10000, a.scenarios); Z = M.normals(a.scenarios)
    t = time.perf_counter(); P = {"float64": M.prep(mp, scen, Z, tb, np.float64)}; t_prep = time.perf_counter() - t
    P["float32"] = M.prep(mp, scen, Z, tb, np.float32)
    K = M.KERNEL; n = len(P["float64"][K.row_args[0]]); mil = n / 1e6
    res = {"machine": {"platform": platform.platform(), "cpu": cpu_name(), "logical_cores": os.cpu_count(),
                       "numba_threads": nb.config.NUMBA_NUM_THREADS},
           "rows": n, "periods": int(P["float64"]["T"]), "prep_seconds": t_prep, "seconds": {}, "seconds_per_million_rows": {},
           "difference_from_cpu_float64": {}}
    def rec(label, sec):
        res["seconds"][label] = sec; res["seconds_per_million_rows"][label] = sec / mil
        print(f"  {label:40s} {sec:9.4f}s = {sec/mil:9.5f}s per million rows", flush=True)
    par, _ = numba_cpu.build(K); ref = par(P["float64"])
    rec("cpu all cores, float64", best(lambda: par(P["float64"]), 3))
    for k in [int(x) for x in a.threads.split(",") if x]:
        if k <= nb.config.NUMBA_NUM_THREADS:
            nb.set_num_threads(k); rec(f"cpu {k} threads, float64", best(lambda: par(P["float64"]), 3))
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
    ref_pv = M.net_cf_pv(ref.astype(np.float64)); scale = np.maximum(np.abs(ref_pv), 1.0)
    def diff(label, acc):
        pv = M.net_cf_pv(acc.astype(np.float64))
        prem = np.maximum(np.abs(ref_pv).max(1, keepdims=True), 1.0)   # each row's largest present value, as a size measure
        share = np.abs(pv - ref_pv) / prem
        d = {"worst_per_policy_rel": float((np.abs(pv - ref_pv) / scale).max()),
             "worst_per_policy_share_of_largest_pv": float(share.max()),
             "p999_per_policy_share_of_largest_pv": float(np.quantile(share.max(1), 0.999)),
             "median_per_policy_share_of_largest_pv": float(np.median(share.max(1))),
             "portfolio_totals_rel": float((np.abs(pv.sum(0) - ref_pv.sum(0)) / np.abs(ref_pv.sum(0))).max())}
        res["difference_from_cpu_float64"][label] = d
        print(f"  {label:40s} worst per policy {d['worst_per_policy_rel']:.2e} (share of the row's largest present value: worst {d['worst_per_policy_share_of_largest_pv']:.2e}, "
              f"99.9% {d['p999_per_policy_share_of_largest_pv']:.2e}, median {d['median_per_policy_share_of_largest_pv']:.2e}), totals {d['portfolio_totals_rel']:.2e}", flush=True)
    if not a.no_gpu:
        try:
            from numba import cuda as ncuda
            from exactuary.backends import cuda
            res["machine"]["gpu"] = ncuda.get_current_device().name.decode() if isinstance(ncuda.get_current_device().name, bytes) else str(ncuda.get_current_device().name)
            K32 = K.with_src(to_f32_src(K.step_src))
            mixed = dict(P["float32"]); mixed[K.discount] = P["float64"][K.discount]
            for label, spec, ft, p in (("gpu, float64", K, "float64", P["float64"]), ("gpu, float32", K32, "float32", P["float32"]),
                                       ("gpu, mixed", K32, "mixed", mixed)):
                go, out = cuda.build(spec, ft)(p); go()
                rec(label, best(go, 5)); diff(label, out.copy_to_host())
                del go, out
        except Exception as e:
            res["gpu_error"] = repr(e)[:400]; print("  GPU not run:", repr(e)[:300], flush=True)
    print("RESULT " + json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
