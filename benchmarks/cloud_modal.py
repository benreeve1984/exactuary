"""Run benchmarks/cloud.py on Modal (modal.com), on a chosen GPU or on processor cores only.

    modal run benchmarks/cloud_modal.py --gpu H100
    modal run benchmarks/cloud_modal.py --cores 64

Each run prints the RESULT line from cloud.py and appends it to benchmarks/results/cloud.jsonl.
"""
import json, os, subprocess, modal

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.4.4", "pandas==3.1.1", "numba==0.65.0", "numba-cuda[cu12]==0.30.4",
                      "lifelib==0.17.3", "modelx==0.33.0", "openpyxl==3.1.5")
         .add_local_dir(os.path.join(ROOT, "exactuary"), "/repo/exactuary", ignore=["**/__pycache__"])
         .add_local_dir(os.path.join(ROOT, "benchmarks"), "/repo/benchmarks", ignore=["results", "**/__pycache__"])
         .add_local_dir(os.path.join(ROOT, "examples", "lifelib_savings"), "/repo/examples/lifelib_savings",
                        ignore=["results", "data", "**/__pycache__"]))
app = modal.App("exactuary-cloud-timing", image=image)

@app.function(timeout=1800)
def bench(args: list[str]) -> str:
    out = subprocess.run(["python", "/repo/benchmarks/cloud.py", *args], capture_output=True, text=True)
    print(out.stdout[-6000:]); print(out.stderr[-3000:])
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    return lines[-1][7:] if lines else json.dumps({"error": out.stderr[-2000:]})

@app.local_entrypoint()
def main(gpu: str = "", cores: int = 8, threads: str = ""):
    f = bench.with_options(gpu=gpu, cpu=cores, memory=32768) if gpu else bench.with_options(cpu=cores, memory=32768)
    args = ([] if gpu else ["--no-gpu"]) + (["--threads", threads] if threads else [])
    res = json.loads(f.remote(args)); res["where"] = f"Modal, {gpu or 'no GPU'}, {cores} cores requested"
    print(json.dumps(res, indent=1))
    path = os.path.join(ROOT, "benchmarks", "results", "cloud.jsonl")
    with open(path, "a") as fh: fh.write(json.dumps(res) + "\n")
