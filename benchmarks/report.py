"""Print the timing and certification results as Markdown tables (from benchmarks/results/*.json)."""
import os, json
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NAMES = {"lifelib_savings": "Savings", "lifelib_savings_dynamic": "Savings, dynamic lapse",
         "lifelib_basicterm": "Term life", "lifelib_fastlife": "Annual (fastlife)"}
def sig(x): return "n/a" if x is None else f"{x:,.0f}" if x >= 100 else f"{x:.3g}"

def timings():
    t = json.load(open(os.path.join(R, "timings.json"))); ms = [m for m in NAMES if m in t]
    print("Seconds per million rows" + f" ({t['machine'].get('cores')} cores, {t['machine'].get('gpu', 'GPU')})\n")
    print("| Step | " + " | ".join(NAMES[m] for m in ms) + " |\n|---|" + "---|" * len(ms))
    print("| Original, vectorised | " + " | ".join(sig(t[m]["original"]["seconds_per_million_rows"]) for m in ms) + " |")
    for k in next(iter(t[m]["kernel_seconds_per_million_rows"] for m in ms)):
        print(f"| Kernel: {k} | " + " | ".join(sig(t[m]["kernel_seconds_per_million_rows"].get(k)) for m in ms) + " |")
    print("\nThe original was measured on: " + "; ".join(f"{NAMES[m]}: {t[m]['original']['rows']:,} rows ({t[m]['original']['workload']})" for m in ms) + ".")

def certify():
    c = json.load(open(os.path.join(R, "certify.json"))); ms = [m for m in NAMES if m in c and "variants" in c[m]]
    vs = list(c[ms[0]]["variants"])
    print("\nGreatest distance from the exact result, relative to the size of the value (every row certified unless stated)\n")
    print("| Model | Rows | Certified | Enclosure width, median | " + " | ".join(vs) + " |\n|---|---|---|---|" + "---|" * len(vs))
    for m in ms:
        r = c[m]
        print(f"| {NAMES[m]} | {r['rows']:,} | {r['rows_certified']:,} | {r['enclosure_width']['median']:.1e} | "
              + " | ".join(f"{r['variants'][v]['distance_from_exact']['worst']:.1e}" if v in r["variants"] else "n/a" for v in vs) + " |")
    print("\nThe same for the net cashflow present value (a difference of large numbers)\n")
    print("| Model | " + " | ".join(vs) + " |\n|---|" + "---|" * len(vs))
    for m in ms:
        r = c[m]
        print(f"| {NAMES[m]} | " + " | ".join(f"{r['variants'][v]['net_cashflow_distance_from_exact']['worst']:.1e}" if v in r["variants"] else "n/a" for v in vs) + " |")

if __name__ == "__main__":
    for f in (timings, certify):
        try: f()
        except FileNotFoundError as e: print(f"(no results yet: {e.filename})")
