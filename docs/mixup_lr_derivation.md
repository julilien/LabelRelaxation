# Mixup-compatible label relaxation — closed-form projection

Companion note for `MixupLabelRelaxationLoss`. This is the formal contribution of Paper A,
Phase 3a: label relaxation for a mixed (mixup/CutMix) target. It is stated so the proposition
and its proof can be lifted into the paper appendix, and it is what the numerical verification
script (`scripts/verify_mixup_lr.py`) checks against a convex-solver oracle.

## Setup

Plain label relaxation (Lienen & Hüllermeier, AAAI 2021) trains against the credal set
`Q_{α,y} = { p ∈ Δ : p_y ≥ 1 − α }` via the optimistic superset loss
`L(p̂, y) = 0` if `p̂ ∈ Q_{α,y}`, else `KL(q* ‖ p̂)` with `q* = argmin_{q ∈ Q_{α,y}} KL(q ‖ p̂)`.
(Moment / reverse-I projection: the *first* KL argument varies. Detaching `q*` leaves the
gradient unchanged by Danskin's theorem, exactly as in the single-label case.)

Mixup replaces the one-hot target by `λ e_i + (1 − λ) e_j`, `λ ∈ [0, 1]`, classes `i ≠ j`.

## The mixed credal set

Relax each mixed component by the same factor `(1 − α)`:

    S = { p ∈ Δ : p_i ≥ a, p_j ≥ b },   a := λ(1 − α),  b := (1 − λ)(1 − α).

The total required mass `a + b = 1 − α` matches the single-label case, so `S` is the natural
one-parameter generalisation. `MixupLabelRelaxationLoss` projects onto `S`.

**Relation to the Minkowski combination.** Let `M = λ Q_{α,i} ⊕ (1 − λ) Q_{α,j}`
(`= { λq + (1−λ)r : q ∈ Q_{α,i}, r ∈ Q_{α,j} }`). Then `M ⊆ S` always
(each component bound survives the convex combination). Equality `M = S` holds on the simplex;
the loss is defined by projection onto `S` and does not depend on the equality, but the
equality is the clean "credal-set combination" narrative and is checked numerically. *(Proof
of `S ⊆ M`: given `p ∈ S`, construct `q, r` by a two-coordinate mass transport; deferred to the
appendix — the loss below is agnostic to it.)*

## Closed-form projection onto S

Minimise `KL(q ‖ p̂)` over `q ∈ Δ` with `q_i ≥ a`, `q_j ≥ b`. KKT stationarity gives, with a
common normaliser `C`,

    q_k = C · p̂_k                      (k ∉ {i, j}),
    q_i = C · p̂_i · e^{μ_i},   q_j = C · p̂_j · e^{μ_j},   μ_i, μ_j ≥ 0,

so a constrained coordinate is either at its natural value `C p̂` (multiplier 0) or pinned to
its bound. Writing `rest = 1 − p̂_i − p̂_j`, four cases (an exact partition, no boundary fuzz):

1. **none active** — `p̂_i ≥ a` and `p̂_j ≥ b`: `q = p̂`, loss `0` (prediction already in `S`).
2. **i only** — raise `i` to `a`, keep the shape elsewhere:
   `C = (1 − a)/(1 − p̂_i)`, `q_i = a`, `q_k = C p̂_k` (`k ≠ i`).
   Valid iff `p̂_i < a` and the resulting `q_j = C p̂_j ≥ b`.
3. **j only** — symmetric: `C = (1 − b)/(1 − p̂_j)`, `q_j = b`, `q_k = C p̂_k` (`k ≠ j`).
   Valid iff `p̂_j < b` and `q_i = C p̂_i ≥ a`.
4. **both active** — `q_i = a`, `q_j = b`, remaining mass `α` spread by shape:
   `C = α / rest`, `q_k = C p̂_k` (`k ∉ {i, j}`).

Selection is exact: if both `p̂_i < a` and `p̂_j < b`, then raising `i` alone gives
`C = (1−a)/(1−p̂_i) < 1`, so `q_j = C p̂_j < p̂_j < b` — i.e. raising one deficient coordinate
pulls the other further below its bound, forcing case 4. Hence:
none if neither deficient; if exactly one is deficient, its single-coordinate case unless it
knocks the other below its bound (then both); if both deficient, both.

The loss is `KL(q ‖ p̂) = Σ_k q_k (log q_k − log p̂_k)`, using `log p̂_k = logsoftmax(z)_k`.

## Limits (sanity constraints; also asserted in the tests)

- **λ → 1** ⇒ `a = 1 − α`, `b = 0`: constraint `j` is vacuous, cases collapse to plain LR for
  class `i` (`q_i = 1 − α`, `q_k = α p̂_k/(1 − p̂_i)`).
- **λ → 0** ⇒ plain LR for class `j`.
- **α → 0** ⇒ `a = λ`, `b = 1 − λ`, `a + b = 1`: `S` shrinks to the single point
  `λ e_i + (1 − λ) e_j`, and the loss reduces to standard mixup cross-entropy (up to the
  target's constant entropy; the logit gradient is identical).
- **i = j** ⇒ single constraint `q_i ≥ a + b = 1 − α`: plain LR on class `i` (handled
  explicitly, since `rest = 1 − 2 p̂_i` would otherwise be wrong).
- **loss = 0 iff `p̂ ∈ S`** (case 1), matching the credal semantics.
