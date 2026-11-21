"""Export every symbolic obligation as a Lean 4 theorem, so that each is checked by Lean's proof
kernel as well as by Z3. Shared sub-terms become variables with defining hypotheses. Requires core
Lean only (no Mathlib).

What this does and does not remove from the trusted base. Lean checks the proposition it is given.
The proposition is still built by `symbolic.py` from the model's `SymbolicSpec`, and translated by
this module: both remain trusted. The translation keeps Z3's meaning as follows.
  * Numbers are `Rat`, where Z3 uses the reals. Every floating-point input is a rational number and
    + - * / keep results rational, so the rationals are the domain that matters for actual inputs.
  * Division by anything other than a non-zero numeric constant goes through a function `divU` that is
    known to equal division only where the divisor is non-zero, as in Z3, where x/0 is unspecified.
    (Lean's own x/0 is 0, which would make some false statements provable.)
  * Each uninterpreted function keeps a distinct name."""
import os, subprocess, z3
from . import symbolic as S

class Printer:
    def __init__(self):
        self.names, self.binders, self.hyps, self.memo, self.props, self.funcs = {}, [], [], {}, [], {}
        self.fn_of, self.need_div, self.binder_ids, self.hyp_ids = {}, False, [], []
    def var(self, e, hint):
        k = e.get_id()
        if k not in self.names: self.names[k] = f"{hint}{len(self.names)}"
        return self.names[k]
    def ty(self, s):
        if s == z3.IntSort(): return "Int"
        if s == z3.RealSort(): return "Rat"
        if s == z3.BoolSort(): return "Prop"
        raise NotImplementedError(f"sort {s} cannot be exported: only integers, reals and truth values are")
    def num(self, e):
        if z3.is_int_value(e): return f"({e.as_long()} : Int)"
        fr = e.as_fraction(); n, d = fr.numerator, fr.denominator
        return f"({n} : Rat)" if d == 1 else f"(({n} : Rat) / {d})"
    def term(self, e):
        k = e.get_id()
        if k not in self.memo: self.memo[k] = self._term(e)
        return self.memo[k]
    def _term(self, e):
        self.ty(e.sort())                                       # refuse any sort outside Int, Real, Bool
        if z3.is_int_value(e) or z3.is_rational_value(e): return self.num(e)
        if z3.is_true(e): return "True"
        if z3.is_false(e): return "False"
        d = e.decl(); kind = d.kind(); ch = e.children()
        if kind == z3.Z3_OP_UNINTERPRETED:
            if not ch:
                n = self.var(e, "c")
                if n not in [b[0] for b in self.binders]:
                    self.binders.append((n, self.ty(e.sort()), d.name())); self.binder_ids.append(e.get_id())
                    if z3.is_bool(e): self.props.append(n)
                return n
            sig = " → ".join([self.ty(d.domain(i)) for i in range(d.arity())] + [self.ty(d.range())])
            key = (d.name(), sig)                  # distinct functions must stay distinct after renaming
            if key not in self.fn_of:
                base = "F_" + "".join(c if c.isalnum() else "_" for c in d.name()); fn, k = base, 1
                while fn in self.funcs: k += 1; fn = f"{base}_{k}"
                self.fn_of[key] = fn; self.funcs[fn] = sig
            fn = self.fn_of[key]
            return self.bind(e, f"({fn} {' '.join(self.term(c) for c in ch)})")
        a = [self.term(c) for c in ch]
        if z3.is_bool(e):
            ops = {z3.Z3_OP_EQ: "=", z3.Z3_OP_LT: "<", z3.Z3_OP_GT: ">", z3.Z3_OP_LE: "≤", z3.Z3_OP_GE: "≥"}
            if kind == z3.Z3_OP_DISTINCT:                       # n-ary: every pair differs
                return "(" + " ∧ ".join(f"({a[i]} ≠ {a[j]})" for i in range(len(a)) for j in range(i + 1, len(a))) + ")" if len(a) > 1 else "True"
            if kind in ops:
                if len(a) != 2: raise NotImplementedError(f"{d.name()} with {len(a)} operands")
                return f"({a[0]} {ops[kind]} {a[1]})"
            if kind in (z3.Z3_OP_IMPLIES, z3.Z3_OP_XOR, z3.Z3_OP_IFF): raise NotImplementedError(d.name())
            if kind == z3.Z3_OP_AND: return "(" + " ∧ ".join(a) + ")"
            if kind == z3.Z3_OP_OR: return "(" + " ∨ ".join(a) + ")"
            if kind == z3.Z3_OP_NOT: return f"(¬ {a[0]})"
            if kind == z3.Z3_OP_ITE: return f"(if {a[0]} then {a[1]} else {a[2]})"
            raise NotImplementedError(d.name())
        if kind == z3.Z3_OP_ADD: s = "(" + " + ".join(a) + ")"
        elif kind == z3.Z3_OP_SUB: s = "(" + " - ".join(a) + ")"
        elif kind == z3.Z3_OP_MUL: s = "(" + " * ".join(a) + ")"
        elif kind == z3.Z3_OP_IDIV:
            if not (z3.is_int_value(ch[1]) and ch[1].as_long() > 0): raise NotImplementedError("integer division by a non-constant")
            s = f"({a[0]} / {a[1]})"
        elif kind == z3.Z3_OP_DIV:
            const = z3.is_rational_value(ch[1]) or z3.is_int_value(ch[1])
            if const and z3.is_true(z3.simplify(ch[1] != 0)): s = f"({a[0]} / {a[1]})"
            else: self.need_div = True; s = f"(divU {a[0]} {a[1]})"
        elif kind == z3.Z3_OP_UMINUS: s = f"(- {a[0]})"
        elif kind == z3.Z3_OP_TO_REAL: s = f"(({a[0]} : Int) : Rat)"
        elif kind == z3.Z3_OP_ITE: s = f"(if {a[0]} then {a[1]} else {a[2]})"
        else: raise NotImplementedError(d.name())
        return self.bind(e, s)
    def bind(self, e, s):
        n = self.var(e, "x"); self.binders.append((n, self.ty(e.sort()), None)); self.binder_ids.append(e.get_id())
        self.hyps.append(f"{n} = {s}"); self.hyp_ids.append(e.get_id())
        return n

TACTIC = "first | grind | (simp_all <;> grind)"
def _render(name, p, A, L, R, comment, body, private=False):
    fs = " ".join(f"({n} : {t})" for n, t in p.funcs.items())
    if p.need_div: fs += " (divU : Rat → Rat → Rat) (hdiv : ∀ a b : Rat, b ≠ 0 → divU a b = a / b)"
    bs = " ".join(f"({n} : {t})" for n, t, _ in p.binders)
    dec = " ".join(f"[Decidable {n}]" for n in p.props)
    hy = "".join(f"\n    (ha{i} : {h})" for i, h in enumerate(A))
    hy += "".join(f"\n    (hd_{p.names[z]} : {h})" for z, h in zip(p.hyp_ids, p.hyps))
    legend = ", ".join(f"{n}={orig}" for n, _, orig in p.binders if orig)
    kw = "private theorem" if private else "theorem"
    return f"/-- {comment}\n{legend} -/\n{kw} {name} {fs}\n    {bs} {dec}{hy} :\n    {L} = {R} := by {body}\n"

def theorem(name, assume, lhs, rhs, comment, parts=(), private=False):
    """One obligation as a Lean theorem. For an aggregate output, `parts` are its components, each a tuple
    (kernel value, original value[, that component's own parts]): each is first proved as a lemma of its own,
    and the aggregate's proof cites those lemmas."""
    if z3.is_int(lhs) != z3.is_int(rhs): lhs, rhs = S.real(lhs), S.real(rhs)
    p = Printer(); A = [p.term(a) for a in assume]; L, R = p.term(lhs), p.term(rhs)
    lemmas, haves = [], ""
    for k, part in enumerate(parts):
        pl, pr, sub_parts = part[0], part[1], (part[2] if len(part) > 2 else ())
        if z3.is_int(pl) != z3.is_int(pr): continue
        if not (pl.get_id() in p.memo and pr.get_id() in p.memo) or p.memo[pl.get_id()] == p.memo[pr.get_id()]: continue
        q = Printer(); qA = [q.term(a) for a in assume]; qL, qR = q.term(pl), q.term(pr)
        lemmas.append(theorem(f"{name}_part{k}", assume, pl, pr, comment, sub_parts, private=True))
        args = [p.fn_of[key] for key in q.fn_of] + (["divU", "hdiv"] if q.need_div else [])
        args += [p.names[z] for z in q.binder_ids] + [f"ha{i}" for i in range(len(qA))] + [f"hd_{p.names[z]}" for z in q.hyp_ids]
        haves += f"\n  have := {name}_part{k} {' '.join(args)}"
    # where division is guarded, first try without the guard's hypothesis: it is rarely needed and slows the search
    g = "(first | (clear hdiv; grind) | grind)" if p.need_div else "grind"
    if haves:       # once the components are established, only the definitions above them are relevant: drop the rest
        stop = {x.get_id() for part in parts for x in part[:2]}; need, todo = set(), [lhs, rhs] + list(assume)
        while todo:
            e = todo.pop(); k = e.get_id()
            if k in need: continue
            need.add(k)
            if k not in stop: todo.extend(e.children())
        drop = [f"hd_{p.names[z]}" for z in p.hyp_ids if z not in need or z in stop]
        short = (f"clear {' '.join(drop)}; " if drop else "") + g
        body = haves + f"\n  first | ({short}) | {g}"
    else: body = TACTIC.replace("grind", g)
    if L == R: body = "rfl"
    return "\n".join(lemmas + [_render(name, p, A, L, R, comment, body, private)])

def export(kspec, sspec, path, tag="kernel", src=None, composite=None):
    """Write all obligations to a Lean file. `composite` maps an aggregate output to the outputs it sums."""
    composite = composite or {}
    lines = ["set_option maxRecDepth 200000", "set_option maxHeartbeats 2000000",
             "set_option linter.unusedVariables false", "namespace Exactuary", ""]
    names = []
    for i, (label, assume, lhs, rhs) in enumerate(S.exit_obligations(kspec, sspec, src)):
        lines.append(theorem(f"{tag}_exit_{i:03d}", assume, lhs, rhs, label)); names.append((f"{tag}_exit_{i:03d}", label))
    for base in (True, False):
        assume, obs, _ = S.obligations(kspec, sspec, base, src)
        byl = {l: (a, b) for l, a, b in obs}
        def parts_of(label): return [byl[c] + (parts_of(c),) for c in composite.get(label, []) if c in byl]
        for i, (label, lhs, rhs) in enumerate(obs):
            nm = f"{tag}_{'month0' if base else 'monthT'}_{i:03d}"
            parts = parts_of(label)
            lines.append(theorem(nm, assume, lhs, rhs, ("month 0: " if base else "month t>=1: ") + label, parts))
            names.append((nm, ("month 0: " if base else "month t>=1: ") + label))
    lines.append("end Exactuary")
    open(path, "w").write("\n".join(lines))
    return names

ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}       # Lean's own standard axioms

def run_lean(path, lean="lean", gate=True):
    """Check a Lean file. Returns (number of theorems, [labels of theorems that failed]).

    Beyond Lean's own verdict, a theorem counts as failed if Lean reports any error or admission against it,
    or if the axioms its proof depends on (as reported by `#print axioms`, which follows dependencies) include
    anything other than Lean's three standard axioms: in particular `sorryAx`. A file that imports other
    modules, or that produces an error or admission that cannot be attributed to a theorem, fails as a whole."""
    import re, tempfile
    text = open(path).read(); src = text.split("\n"); failed = set()
    if re.search(r"^\s*import\b", text, flags=re.M): failed.add("(the file imports other modules)")
    # Only generated forms are accepted: `theorem` / `private theorem` starting a line, and nothing else that
    # declares or admits anything. A declaration in any other form would escape the axiom audit below.
    code = re.sub(r"/-.*?-/", "", text, flags=re.S); code = re.sub(r"--.*", "", code)
    decl = r"(theorem|lemma|axiom|def|instance|opaque|abbrev|example|macro|syntax|elab|structure|inductive|class|constant|unsafe|partial|noncomputable|attribute|initialize|mutual|declare_syntax_cat|notation|infix|prefix|postfix)"
    start = re.compile(r"^\s*(@\[[^\]]*\]\s*)*((private|protected|noncomputable|unsafe|partial|nonrec)\s+)*" + decl + r"\b")
    # A line that starts at the left margin must begin with one of the forms the generated files and the
    # glue file use; anything else fails, whether or not it is recognised as a declaration.
    margin = re.compile(r"((private )?theorem\s+[A-Za-z_]|def\s+[A-Za-z_]|namespace\s|end\s|set_option\s|variable\s|open\s)")
    for line in code.split("\n"):
        if not line.strip(): continue
        if not line[0].isspace():
            if not margin.match(line): failed.add("(a line the checker does not accept: " + line.strip()[:60] + ")")
            continue
        if start.match(line) or re.match(r"\s*@\[", line) or re.search(r"\b(nonrec|theorem|lemma|axiom|instance)\b", line):
            failed.add("(declaration in a form the checker does not audit: " + line.strip()[:60] + ")")
    names = re.findall(r"^(?:private )?theorem\s+([A-Za-z_][A-Za-z0-9_'.]*)", text, flags=re.M)
    label_of = {}
    for k, line in enumerate(src):
        m = re.match(r"(?:private )?theorem\s+([A-Za-z_][A-Za-z0-9_'.]*)", line)
        if m:
            lab = next((src[j][4:] for j in range(k - 1, -1, -1) if src[j].startswith("/-- ")), None)
            label_of[m.group(1)] = lab if lab is not None and all(not re.match(r"(?:private )?theorem\s", src[i]) for i in range(src.index("/-- " + lab) + 1, k)) else m.group(1)
    probes = "\n".join(f"#print axioms {n}" for n in names)
    ends = [k for k, line in enumerate(src) if re.match(r"end\s+\S+\s*$", line)]
    body = src[:ends[-1]] + [probes] + src[ends[-1]:] if ends else src + [probes]
    with tempfile.TemporaryDirectory() as d:
        probe_path = os.path.join(d, os.path.basename(path)); open(probe_path, "w").write("\n".join(body))
        proc = subprocess.run([lean, probe_path], capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    for line in out.split("\n"):
        if ": error" in line or "declaration uses" in line:
            try: ln = int(line.split(".lean:")[1].split(":")[0])
            except (IndexError, ValueError): failed.add("(unattributed: " + line[:120] + ")"); continue
            lab = next((src[k][4:] for k in range(min(ln, len(src)) - 1, -1, -1) if src[k].startswith("/-- ")), None)
            failed.add(lab if lab is not None else "(unattributed: " + line[:120] + ")")
    seen = set()
    for m in re.finditer(r"'([^']+)' (depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", out):
        short = m.group(1).split(".")[-1]; seen.add(short)
        axioms = {x.strip() for x in (m.group(3) or "").replace("\n", " ").split(",") if x.strip()}
        if axioms - ALLOWED_AXIOMS and label_of.get(short, short) not in failed:
            failed.add(label_of.get(short, short) + f" (depends on {sorted(axioms - ALLOWED_AXIOMS)})")
    for n in names:
        if n.split(".")[-1] not in seen and label_of.get(n, n) not in failed:
            failed.add(label_of.get(n, n) + " (axioms could not be listed)")
    if proc.returncode != 0 and not failed: failed.add(f"(lean exited with code {proc.returncode})")
    n = sum(l.startswith("theorem ") for l in src)
    if gate:
        from . import gates
        gates.record(f"Lean check of {os.path.basename(path)} ({n} theorems)", not failed and n > 0, f"{len(failed)} failed")
    return n, sorted(failed)

GLUE = os.path.join(os.path.dirname(__file__), "lean", "Glue.lean")
