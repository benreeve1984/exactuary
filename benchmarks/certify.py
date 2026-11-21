"""How far is each computed result from the exact answer? A rigorous, per-row answer for every example.

For a sample of rows of each model, `exactuary.interval.enclose` gives an interval that contains the
result the kernel would produce in exact real arithmetic on the prepared inputs. Each computed variant
is then placed against that interval: its distance from the exact result is at most the distance to the
interval's far end. The variants are

    original's summation   the kernel's float64 trace summed as the original sums it (a matrix product);
                           the exact comparison shows this is the original's own reported result
    cpu float64            the compiled kernel, running sum, early exit
    cpu float32, gpu float64, gpu float32

Distances are relative to max(|exact|, 1) per value, where |exact| is taken at the middle of the interval.

    python benchmarks/certify.py [model ...] [--rows N]
"""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from models import MODELS, subset
from exactuary import gates
from exactuary.interval import enclose, distance, combine, NotCertifiable
from exactuary.backends import numba_cpu
from exactuary.spec import to_f32_src

def summarise(d, scale, ok):
    r = (d / scale)[ok]
    return {"worst": float(r.max()), "median": float(np.median(r))} if r.size else {"worst": None, "median": None}

def run(name, n_rows):
    m = MODELS[name](big=False, original=False); K = m["kernel"]; p = m["p"]; n = len(p["float64"][K.row_args[0]])
    idx = np.unique(np.linspace(0, n - 1, min(n, n_rows)).astype(np.int64))
    s64, s32 = subset(K, p["float64"], idx), subset(K, p["float32"], idx)
    t = time.perf_counter()
    try: enc = enclose(K, s64)
    except NotCertifiable as e:
        print(f"[{name}] no enclosure: {e}"); gates.record(f"{name}: enclosure computed", False, str(e)); return {"error": str(e)}
    secs = time.perf_counter() - t; ok = enc["certified"]
    mid = 0.5 * (enc["lo"] + enc["hi"]); scale = np.maximum(np.abs(np.where(ok[:, None], mid, 1.0)), 1.0)
    with np.errstate(all="ignore"): width = np.where(ok[:, None], enc["hi"] - enc["lo"], np.inf)
    nlo, nhi = combine(enc, m["net"]); nmid = 0.5 * (nlo + nhi); nscale = np.maximum(np.abs(np.where(ok, nmid, 1.0)), 1.0)
    res = {"rows": int(len(idx)), "rows_certified": int(ok.sum()), "periods": int(s64["T"]), "outputs": len(K.accumulate),
           "undecided_comparisons": enc["undecided"], "seconds": secs,
           "enclosure_width": summarise(width, scale, ok[:, None] & np.ones_like(width, bool)),
           "net_cashflow_enclosure_width": summarise((nhi - nlo)[:, None], nscale[:, None], ok[:, None]), "variants": {}}
    print(f"[{name}] {len(idx):,} rows x {s64['T']} {m['period']}s, {len(K.accumulate)} accumulated outputs: enclosure in {secs:.1f}s; "
          f"{int(ok.sum()):,} rows certified; {enc['undecided']} undecided comparisons")
    print(f"  width of the enclosure itself: worst {res['enclosure_width']['worst']:.1e}, median {res['enclosure_width']['median']:.1e}; "
          f"for net cashflow: worst {res['net_cashflow_enclosure_width']['worst']:.1e}")
    gates.record(f"{name}: rows with a finite enclosure", ok.any(), f"{int(ok.sum()):,} of {len(idx):,}")
    variants = {}
    run_pv, run_trace = numba_cpu.build(K)
    tr = run_trace(s64); variants["original's summation (float64)"] = np.column_stack([tr[K.ix(a)].T @ s64[K.discount] for a in K.accumulate]); del tr
    variants["cpu float64"] = run_pv(s64)
    K32 = K.with_src(to_f32_src(K.step_src)); variants["cpu float32"] = numba_cpu.build(K32, "float32")[0](s32)
    try:
        from exactuary.backends import cuda
        for ft, KK, pp in (("float64", K, s64), ("float32", K32, s32)):
            go, out = cuda.build(KK, ft)(pp); go(); variants[f"gpu {ft}"] = out.copy_to_host()
    except Exception as e: print("  GPU not run:", repr(e)[:160])
    w = np.array(m["net"], dtype=np.float64)
    for label, v in variants.items():
        v = np.asarray(v, np.float64); d = distance(v, enc)
        net = v @ w
        with np.errstate(all="ignore"): dn = np.maximum(np.abs(net - nlo), np.abs(nhi - net)) * (1 + 1e-15) + np.abs(net) * 1e-15
        okv = ok[:, None] & np.isfinite(v)
        a, b = summarise(d, scale, okv), summarise(dn[:, None], nscale[:, None], ok[:, None] & np.isfinite(net)[:, None])
        res["variants"][label] = {"distance_from_exact": a, "net_cashflow_distance_from_exact": b}
        print(f"  {label:32s} at most {a['worst']:.1e} from exact (median {a['median']:.1e}); net cashflow at most {b['worst']:.1e} (median {b['median']:.1e})")
    return res

if __name__ == "__main__":
    args = sys.argv[1:]; n_rows = 2000
    if "--rows" in args: i = args.index("--rows"); n_rows = int(args[i + 1]); del args[i:i + 2]
    path = os.path.join(ROOT, "benchmarks", "results", "certify.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    out = json.load(open(path)) if os.path.exists(path) else {}
    for nm in args or list(MODELS):
        out[nm] = run(nm, n_rows); json.dump(out, open(path, "w"), indent=1)
    gates.finish()
