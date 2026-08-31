# Round 13: Growing the Ensemble Member Pool — Pre-registration

**Document:** 31-round13-preregistration.md · **Status: FROZEN before any
candidate or ensemble run** · **Date frozen:** 2026-08-31
**Origin:** round 11's consequence (docs/28): the ensemble axis rests
"unless the pool gains new architectures." This round grows the pool with
genuinely different architectures and re-runs the diverse-strong ensemble
under a qualification rule frozen *before any candidate result exists*
(including round 12's AT/AU, still unreported at this freeze).

## 1 Hypothesis and rationale

Round 11's complements law: an ensemble buys the most from members that are
strong AND fail differently, with the weakest member setting a soft floor
(AJ at 0.8504 diluted a strong pool). New members must therefore clear a
strength bar *and* differ architecturally. Two candidates the library can
already express, plus round 12's arms as free candidates:

- **AV** — a different *neuron model*: heterogeneous adaptive LIF (k = 3,
  per-neuron random τ, spike-triggered adaptation) under the modern recipe
  (two recurrent 256 layers, aug, attention, 5 ms). This is also
  adaptation's registered retest inside the combination era — round 4
  tested it only on the count-readout/10 ms recipe, where every other
  modern feature was likewise null-or-negative alone.
- **AW** — a different *shape*: one wide 512 layer at full modern spec
  (attention, learned τ, 5 ms, recurrent).
- **AT, AU** (round 12, docs/29) — different *dynamics class* (no reset);
  their bump-0 draws are candidates under the same rule at no extra cost.

**U-H1:** at least one new architecture qualifies, and the enlarged
diverse-strong ensemble beats AR.

## 2 Protocol (frozen)

Harness at the commit carrying this document; full-test-set evaluation;
`--threads 16`.

**Stage 1 — candidates:** `AV AW --seeds 3` (bumps 0/100/200). Rules vs
the reused AK mean 0.8888, ±0.015 bands (informational verdicts; for AV
this scores adaptation-in-the-modern-recipe as a registered finding in its
own right).

**Stage 2 — qualification (frozen decision function):** a candidate tag
joins the ensemble pool iff its **bump-0 draw ≥ 0.8700** (above AJ's
diluting 0.8504, below the weakest current member X's 0.8884). Applies to
AV, AW, AT, AU. No other selection freedom exists: the ensemble arm is

```
AX = { AK, AP, X } ∪ { every qualifying candidate }
```

evaluated once (summed logits, bump-0 members — the AR machinery). If no
candidate qualifies, AX is not run and the round records that the pool
cannot currently be grown.

**Stage 3 — ensemble rule:** AX vs AR = 0.9366: ≥ +0.010 → **POSITIVE**;
≤ −0.010 → **NEGATIVE**; else **NULL**. **Milestone flag: AX ≥ 0.9400** —
the top of the published band.

Commands:

```
cargo run --release --features datasets --example shd_sweep -- AV AW --seeds 3 --threads 16
cargo run --release --features datasets --example shd_sweep -- AX --threads 16   # if any candidate qualifies
```

(The AX member list is inserted into the harness's DIVERSE_ARMS table by
the frozen rule above once stage-2 qualification is mechanical; the code
change carries no discretion.)

## 3 Predictions (falsifiable, before running)

- **U1:** AW qualifies (≥ 0.87 at bump 0): width is a mild variation of a
  recipe known to work. Point estimate for mean(AW): 0.87–0.89.
- **U2:** AV lands NULL-or-better vs AK — the combination-era retest
  rescues adaptation the way it rescued learned τ (point estimate
  mean(AV) ≈ 0.87); qualification is a coin flip.
- **U3:** AT does *not* qualify (docs/29 predicts it lands well below
  0.87); AU qualification uncertain.
- **U4:** if ≥ 1 candidate qualifies, AX ≥ AR (more strong-and-different
  members help); milestone (≥ 0.94) plausible only if two or more qualify.
- **U5:** no ensemble regression below AR − 0.010 (the bar exists
  precisely to prevent dilution).

## 4 Disclosures

- Frozen before ANY candidate result, including round 12's still-running
  arms. AV/AW have never been run; AV's neuron model ran only its layer
  gates.
- Expected wall-clock: stage 1 ≈ 6 × 12–16 min; stage 3 ≈ 35–60 min
  depending on pool size.
- Amendment policy: tighten-only; deviations recorded in docs/32.

## 5 Consequence

- AX POSITIVE → new campaign headline; if the 0.94 milestone fires, the
  campaign closes at the top of the published band.
- AX NULL/NEGATIVE or no qualifiers → the pool-growth avenue closes at
  this architecture generation; further gains require model classes the
  library does not yet express (each behind its own registration).
- AV's verdict is recorded either way as adaptation's
  combination-era retest — completing the arc begun in round 4.
