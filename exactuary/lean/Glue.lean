/-!
Lean, part 1: the induction that turns the per-month obligations (checked by Z3 and re-checked
in the exported obligation files) into a statement about the whole projection, for every month and any
projection length. Core Lean only, no Mathlib.
-/
namespace Exactuary

variable {σ β : Type}

/-- State the kernel carries into month `t` when started from `s0`. -/
def runState (step : Nat → σ → β × σ) (s0 : σ) : Nat → σ
  | 0 => s0
  | n + 1 => (step n (runState step s0 n)).2

/-- What the kernel reports for month `t`. -/
def kernelOut (step : Nat → σ → β × σ) (s0 : σ) (t : Nat) : β :=
  (step t (runState step s0 t)).1

/-- **Simulation theorem.** `obs t` are the original model's cells in month `t`, `carry t` is the
state the original implies the kernel should be carrying into month `t`, `inv` is an invariant of
the original alone. If month 0 matches from any start state, and each later month matches whenever
the carried state matches, then the kernel reproduces the original in every month. -/
theorem simulation (step : Nat → σ → β × σ) (s0 : σ) (obs : Nat → β) (carry : Nat → σ)
    (inv : Nat → Prop)
    (base : ∀ s, step 0 s = (obs 0, carry 1) ∧ inv 0)
    (ind : ∀ t, 1 ≤ t → inv (t - 1) → step t (carry t) = (obs t, carry (t + 1)) ∧ inv t) :
    ∀ t, kernelOut step s0 t = obs t := by
  have key : ∀ t, runState step s0 (t + 1) = carry (t + 1) ∧ inv t := by
    intro t
    induction t with
    | zero => simp [runState, (base s0).1, (base s0).2]
    | succ n ih =>
      have h := ind (n + 1) (by omega) (by simpa using ih.2)
      simp [runState] at ih ⊢
      rw [ih.1, h.1]; exact ⟨rfl, h.2⟩
  intro t
  cases t with
  | zero => simp [kernelOut, runState, (base s0).1]
  | succ n =>
    have h := ind (n + 1) (by omega) (by simpa using (key n).2)
    simp [kernelOut, (key n).1, h.1]

/-- Present value as the kernel computes it: a running sum over months. -/
def runningPV (flow disc : Nat → Rat) : Nat → Rat
  | 0 => 0
  | n + 1 => runningPV flow disc n + flow n * disc n

/-- If every month's cashflow matches, every present value matches, for any horizon. -/
theorem pv_matches (flowK flowO disc : Nat → Rat) (h : ∀ t, flowK t = flowO t) :
    ∀ n, runningPV flowK disc n = runningPV flowO disc n := by
  intro n; induction n with
  | zero => rfl
  | succ n ih => simp [runningPV, ih, h]

/-- Early exit, part 1: once the exit condition `P` holds it holds forever and cashflows stay zero.
The hypothesis is exactly the one-month lemma proved by Z3. -/
theorem zero_after (P : Nat → Prop) (flow : Nat → Rat)
    (lemma1 : ∀ t, P t → flow (t + 1) = 0 ∧ P (t + 1)) (k : Nat) (hk : P k) :
    ∀ d, flow (k + d + 1) = 0 ∧ P (k + d + 1) := by
  intro d; induction d with
  | zero => exact lemma1 k hk
  | succ d ih => exact lemma1 (k + d + 1) ih.2

/-- Early exit, part 2: stopping the loop after month `k` loses nothing. -/
theorem early_exit_sound (P : Nat → Prop) (flow disc : Nat → Rat)
    (lemma1 : ∀ t, P t → flow (t + 1) = 0 ∧ P (t + 1)) (k : Nat) (hk : P k) :
    ∀ d, runningPV flow disc (k + 1 + d) = runningPV flow disc (k + 1) := by
  intro d; induction d with
  | zero => rfl
  | succ d ih =>
    have hz := (zero_after P flow lemma1 k hk d).1
    have e : k + 1 + d = k + d + 1 := by omega
    show runningPV flow disc (k + 1 + d) + flow (k + 1 + d) * disc (k + 1 + d) = _
    rw [ih, e, hz]; grind

end Exactuary
