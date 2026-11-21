"""One consistent set of timings for every example model, taken one after another on an idle machine.

For each model:
  * the original (already vectorised) model is run once, fresh, on a workload it can complete;
  * the kernel is run on about ten million rows in each configuration: one core, all cores, with and
    without early exit, on the GPU in double and in single precision.
All figures are then expressed per million rows. The original's figure is scaled linearly from the
size it was measured at, which is stated. Input preparation and transfer to the GPU are excluded
from kernel times and reported separately.

    python benchmarks/timings.py [model ...]        # default: all
"""
import sys, os, time, json, platform
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from models import MODELS
from exactuary.backends import numba_cpu
from exactuary.spec import to_f32_src

def best(f, k):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)

def run(name):
    m = MODELS[name](big=True, original=True)
    K, p, t_prep, period, (o_rows, o_sec, o_desc) = m["kernel"], m["p"], m["prep"], m["period"], m["original"]
    n = len(p["float64"][K.row_args[0]]); mil = n / 1e6
    res = {"rows": n, "periods_per_row": int(p["float64"]["T"]), "period": period, "prep_seconds": t_prep,
           "original": {"rows": o_rows, "seconds": o_sec, "workload": o_desc, "seconds_per_million_rows": o_sec / o_rows * 1e6},
           "kernel_seconds": {}, "kernel_seconds_per_million_rows": {}}
    print(f"[{name}] original: {o_sec:.2f}s for {o_rows:,} rows ({o_desc}) = {o_sec/o_rows*1e6:.1f}s per million", flush=True)
    def rec(label, sec):
        res["kernel_seconds"][label] = sec; res["kernel_seconds_per_million_rows"][label] = sec / mil
        print(f"  {label:38s} {sec:9.4f}s for {n:,} rows = {sec/mil:9.5f}s per million  ({o_sec/o_rows*1e6/(sec/mil):,.0f}x the original)", flush=True)
    P64 = p["float64"]
    one, _ = numba_cpu.build(K, parallel=False); one(P64)
    rec("cpu 1 core, float64, no early exit", best(lambda: one(P64, False), 2))
    rec("cpu 1 core, float64", best(lambda: one(P64), 2))
    par, _ = numba_cpu.build(K); par(P64)
    rec("cpu all cores, float64, no early exit", best(lambda: par(P64, False), 5))
    rec("cpu all cores, float64", best(lambda: par(P64), 5))
    try:
        from exactuary.backends import cuda
        for ft in ("float64", "float32"):
            Kf = K.with_src(to_f32_src(K.step_src)) if ft == "float32" else K
            t = time.perf_counter(); go, out = cuda.build(Kf, ft)(p[ft]); t_dev = time.perf_counter() - t
            go(); rec(f"gpu, {ft}", best(go, 5)); res[f"gpu_transfer_and_compile_seconds_{ft}"] = t_dev
            del go, out
    except Exception as e:
        res["gpu_error"] = repr(e)[:300]; print("  GPU not run:", repr(e)[:200], flush=True)
    return res

if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    path = os.path.join(ROOT, "benchmarks", "results", "timings.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["machine"] = {"platform": platform.platform(), "processor": platform.machine(), "cores": os.cpu_count(),
                      "load_average_at_start": os.getloadavg()[0]}
    try:
        from numba import cuda as _c; out["machine"]["gpu"] = _c.get_current_device().name.decode() if hasattr(_c.get_current_device().name, "decode") else str(_c.get_current_device().name)
    except Exception: pass
    for nm in names:
        out[nm] = run(nm); json.dump(out, open(path, "w"), indent=1)
