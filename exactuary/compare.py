"""Exact comparison of a kernel's trace with the original model's trace."""
import numpy as np

def same(a, b):
    """True where two values are equal as numbers, or both NaN. +0 and -0 count as equal, and so do NaNs
    with different payloads; `bits_differ` counts those cases so that they can be reported."""
    return (a == b) | (np.isnan(a) & np.isnan(b))

def bits_differ(a, b):
    """True where two float64 arrays differ in their bit patterns."""
    a, b = np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)
    return a.view(np.uint64) != b.view(np.uint64)

def compare_trace(trace, oracle, names, verbose=True, gate=True):
    """trace: array [output, period, row]; oracle: mapping name -> array [period, row].
    Returns ({name: (count, max relative difference, first (period, row))}, values compared).
    Raises if the two are not the same shape, if there is nothing to compare, or if a name repeats:
    a comparison that silently broadcasts or compares nothing would pass vacuously."""
    names = list(names)
    if not names or len(set(names)) != len(names): raise ValueError("output names must be non-empty and unique")
    if np.ndim(trace) != 3: raise ValueError("the kernel trace must be an array [output, period, row]")
    if len(trace) != len(names): raise ValueError(f"trace has {len(trace)} outputs; {len(names)} names were given")
    bad, total, bits = {}, 0, 0
    for k, n in enumerate(names):
        o, a = np.asarray(oracle[n]), np.asarray(trace[k])
        if o.ndim != 2: raise ValueError(f"{n}: the original's trace must be an array [period, row]")
        if a.shape != o.shape: raise ValueError(f"{n}: kernel trace has shape {a.shape}, the original's has {o.shape}")
        if o.size == 0: raise ValueError(f"{n}: nothing to compare")
        if a.dtype != np.float64 or o.dtype != np.float64: raise ValueError(f"{n}: both traces must be float64")
        eq = same(a, o); total += eq.size
        bits += int((eq & bits_differ(a, o)).sum())
        if not eq.all():
            with np.errstate(all="ignore"):
                rel = np.nanmax(np.abs(a - o)[~eq] / np.maximum(np.abs(o[~eq]), 1e-300))
            bad[n] = (int((~eq).sum()), float(rel), tuple(int(x) for x in np.argwhere(~eq)[0]))
    if verbose:
        print(f"  compared {len(names)} outputs, {total:,} values: {len(bad)} outputs differ; "
              f"equal as numbers but not bit for bit (sign of zero, NaN payload): {bits}")
        for n, v in list(bad.items())[:12]:
            print(f"    DIFFERS {n}: {v[0]:,} values, max relative {v[1]:.3g}, first at (period, row) = {v[2]}")
    if gate:
        from . import gates
        gates.record(f"exact comparison of {len(names)} outputs, {total:,} values", not bad, f"{len(bad)} outputs differ")
    return bad, total
