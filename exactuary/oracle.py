"""Capture every intermediate value of an original modelx/lifelib model, to serve as the oracle."""
import numpy as np

def cell_name(cell, arg="-"):
    return cell if arg == "-" else f"{cell}[{arg if arg is not None else 'ALL'}]"

def parse_cell(name):
    """'claims[DEATH]' -> ('claims', 'DEATH');  'claims[ALL]' -> ('claims', None);  'age' -> ('age',)"""
    if "[" in name:
        c, k = name[:-1].split("["); return (c, None if k == "ALL" else k)
    return (name,)

def capture(space, names, T, n=None, cells=None):
    """Evaluate each named cell of a modelx space for periods 0..T-1.
    Returns {name: float64 array [period, row]}.

    n:     number of rows. Give it when some cells return a single number rather than one value
           per row; such values are repeated for every row.
    cells: optional {name: function of t}, for cells that live in another space or take their
           arguments in a different order from `cell(t, *args)`."""
    out, cells = {}, cells or {}
    for name in names:
        if name in cells: f = cells[name]
        else:
            cell, *args = parse_cell(name)
            f = (lambda g, a: lambda t: g(t, *a))(getattr(space, cell), args)
        vals = []
        for t in range(T):
            v = np.asarray(f(t))
            if v.dtype.kind in "iu" and v.size and np.abs(v).max() > 2 ** 53:
                raise ValueError(f"{name} at period {t}: integer beyond 2^53 cannot be compared as float64")
            vals.append(v.astype(np.float64))
        rows = n if n is not None else max(v.size for v in vals)
        out[name] = np.array([np.broadcast_to(v, (rows,)) for v in vals])
    return out
