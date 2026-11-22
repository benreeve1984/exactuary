"""Vectorised code taken straight to a GPU, for comparison with the compiled fast model.

The savings model's step runs as whole-array arithmetic: one month for all rows, then the next
month. With NumPy this is the style of lifelib's own model, without modelx and without storing
every month. With CuPy, a GPU library that mirrors NumPy, the same code runs on a GPU with no other
change. This is the usual way to move vectorised Python to a GPU.

The script times both, and compares each result with the compiled fast model on processor cores.

    python benchmarks/vectorised_gpu.py [--scenarios 1000] [--numpy-scenarios 20] [--no-gpu]

Prints one line starting with RESULT followed by JSON.
"""
import sys, os, time, json, platform, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "examples", "lifelib_savings"))

def whole_array_pv(xp, spec, p, ftype=np.float64):
    """Present values [row, accumulated output] by whole-array arithmetic over every period, with array library xp."""
    from exactuary.backends import numpy_vec
    from exactuary.spec import make_step
    one = ftype(1)
    h = numpy_vec.helpers(xp)
    where = xp.where
    h.update(sel=lambda c, a, b: (a if c else b) if isinstance(c, (bool, np.bool_)) else where(c, a, b),
             tof=lambda a: a.astype(ftype), C=ftype, ONE=one, ZERO=ftype(0))
    step = make_step(h, spec.step_src)
    # CuPy's compiled operators import their own modules on first use and look for __import__ in the
    # caller's builtins. The step has none, so give it that one name. The step's text cannot refer to it.
    step.__globals__["__builtins__"]["__import__"] = __import__
    q = {k: (xp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    n, T = len(p[spec.row_args[0]]), int(p["T"])
    state = [xp.zeros(n, dtype=ftype) for _ in spec.state]
    acc = [xp.zeros(n, dtype=ftype) for _ in spec.accumulate]
    ix = [spec.ix(name) for name in spec.accumulate]
    disc = p[spec.discount]
    for t in range(T):
        r = step(t, t == 0, *numpy_vec.step_args(spec, q, t, state))
        state = list(r[spec.n_out:])
        d = disc[t]
        for k, j in enumerate(ix):
            acc[k] = acc[k] + r[j] * d
    out = xp.stack(acc, axis=1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=1000)
    ap.add_argument("--numpy-scenarios", type=int, default=20)
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    import model as M
    from exactuary.backends import numba_cpu
    from exactuary.spec import to_f32_src
    m = M.load_original(); tb = M.extract_tables(m.Projection)
    K = M.KERNEL; K32 = K.with_src(to_f32_src(K.step_src))
    def inputs(n_scen):
        mp, scen = M.portfolio_sample(m, 10000, n_scen); Z = M.normals(n_scen)
        return {ft: M.prep(mp, scen, Z, tb, np.float32 if ft == "float32" else np.float64) for ft in ("float64", "float32")}
    res = {"machine": {"platform": platform.platform(), "logical_cores": os.cpu_count()}, "runs": {}}
    par, _ = numba_cpu.build(K)
    def record(label, n, sec, acc, ref):
        pv, rpv = M.net_cf_pv(np.asarray(acc, dtype=np.float64)), M.net_cf_pv(ref.astype(np.float64))
        size = np.maximum(np.abs(rpv).max(1, keepdims=True), 1.0)
        ok = np.isfinite(pv).all(1)           # rows with an undefined or infinite value are counted, then left out
        d = {"rows": n, "seconds": sec, "seconds_per_million_rows": sec / (n / 1e6),
             "share_of_rows_not_finite": float(1 - ok.mean()),
             "worst_row_share_of_largest_pv": float((np.abs(pv[ok] - rpv[ok]) / size[ok]).max()),
             "portfolio_totals_rel": float((np.abs(pv[ok].sum(0) - rpv[ok].sum(0)) / np.abs(rpv[ok].sum(0))).max())}
        res["runs"][label] = d
        print(f"  {label:44s} {sec:9.3f}s for {n:,} rows = {d['seconds_per_million_rows']:9.4f}s per million | "
              f"rows not finite {d['share_of_rows_not_finite']:.2%}, worst row {d['worst_row_share_of_largest_pv']:.1e}, "
              f"totals {d['portfolio_totals_rel']:.1e}", flush=True)
    # whole-array NumPy on the processor, on a smaller workload
    P = inputs(a.numpy_scenarios); n = len(P["float64"][K.row_args[0]]); ref = par(P["float64"], False)
    with np.errstate(all="ignore"):
        t = time.perf_counter(); acc = whole_array_pv(np, K, P["float64"]); sec = time.perf_counter() - t
    record("whole-array NumPy, processor, float64", n, sec, acc, ref)
    if not a.no_gpu:
        try:
            import cupy as cp
            res["machine"]["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            res["machine"]["cupy"] = cp.__version__
            P = inputs(a.scenarios); n = len(P["float64"][K.row_args[0]]); ref = par(P["float64"], False)
            t = time.perf_counter(); par(P["float64"]); res["compiled_cpu_seconds"] = time.perf_counter() - t
            for label, spec, ft in (("whole-array CuPy, GPU, float64", K, "float64"), ("whole-array CuPy, GPU, float32", K32, "float32")):
                fty = np.float32 if ft == "float32" else np.float64
                best = None
                for _ in range(2):          # the first run includes compiling CuPy's kernels
                    t = time.perf_counter(); acc = whole_array_pv(cp, spec, P[ft], fty); cp.cuda.Stream.null.synchronize()
                    sec = time.perf_counter() - t; best = sec if best is None else min(best, sec)
                host = cp.asnumpy(acc); del acc; cp.get_default_memory_pool().free_all_blocks()
                record(label, n, best, host, ref)
        except Exception as e:
            res["gpu_error"] = repr(e)[:400]; print("  GPU not run:", repr(e)[:300], flush=True)
    print("RESULT " + json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
