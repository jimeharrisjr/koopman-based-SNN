# Round 10: Depth's Enabler and the Diverse Ensemble — Pre-registration

**Document:** 22-round10-preregistration.md · **Status: FROZEN before any
gated run** · **Date frozen:** 2026-08-28
**Origin:** the last two unexplored accuracy items on improvements.md P1
(depth's enabling ingredient; diverse ensembles). Protocol committed and
pushed before results, per the P0.1 discipline.

## 1 Hypotheses and rationale

**Depth.** Three data points say each depth increment needs a new enabling
ingredient: round 1's second layer failed feedforward, paid under
recurrence + augmentation (round 4); round 5's third layer lost 2 points on
the then-best recipe. The candidate enabler, chosen for its clean identity
story over per-layer learning rates and normalization analogs: **skip
connections grown from zero** — layer *l* ≥ 2 also reads layer *l−2*'s
same-step spikes through `W_skip`, zero-initialized, so the network *is* the
plain chain at step 0 and training grows the bypass only if it pays (the
recurrence/τ/attention discipline). Two arms separate depth from enabler.

- **X1:** a third layer on the AK recipe without an enabler does not pay
  (the AE result replicates on the new recipe).
- **X2:** zero-init skips are the enabler: the skip arm beats the plain
  three-layer arm.

**Diversity.** AN's homogeneous ensemble (three AKs) hit 0.9000. Diverse
members decorrelate errors but the available diverse members are weaker
(AJ mean 0.8594, X mean 0.8563 vs AK 0.8888).

- **X3:** at these member strengths, diversity roughly offsets weakness —
  the diverse ensemble lands within noise of AN.

## 2 Implementation under test (merged and gated)

`KoopmanLayer::with_skip` + network routing + a new BPTT credit path
(layer *l*'s spikes now also reach layer *l+2* within the step) and skip
weight gradients, all under the shared optimizer/clip and the threaded
path. Gates: `zero_skip_matches_plain_chain_exactly_at_init` (bitwise
logits and first-loss equality) and `skip_connections_grow_from_zero_and_train`
(loss decreases, `W_skip` moves, threaded). The diverse-ensemble runner
trains one member per config and sums logits; members bin the shared test
samples each at their own time resolution. Logit scales across readout
modes are comparable by construction (all are readout maps of mean spike
activity); this is a disclosed assumption, not a gate.

## 3 Protocol (frozen)

Harness at the commit carrying this document; full-test-set evaluation
(2,240 samples); `--threads 16`.

| run | config | seeds | role |
|---|---|---|---|
| `AO --seeds 3` | AK recipe + third recurrent layer (256-256-256) | 0/100/200 | depth, no enabler |
| `AP --seeds 3` | same + zero-init skips on layer 2 | 0/100/200 | depth + enabler |
| `AQ` | diverse ensemble {AK, AJ, X}, one member each at seed_bump 0, summed logits | one evaluation | diversity arm |
| AK | **reused** (round 8): mean **0.8888** | — | control |
| AN | **reused** (round 9): **0.9000** | — | homogeneous-ensemble comparator |

Commands:

```
cargo run --release --features datasets --example shd_sweep -- AO AP --seeds 3 --threads 16
cargo run --release --features datasets --example shd_sweep -- AQ --threads 16
```

**Rules (frozen):**
- **AO** and **AP**, each: mean over 3 seeds − 0.8888: ≥ +0.015 →
  **POSITIVE**; ≤ −0.015 → **NEGATIVE**; else **NULL**.
- **Enabler comparison:** mean(AP) − mean(AO) ≥ +0.015 → **ENABLER
  CONFIRMED** (skips specifically help depth, regardless of whether depth
  itself beats two layers); ≤ −0.015 → enabler harmful; else inconclusive.
- **AQ:** value − 0.9000 (AN): ≥ +0.010 → **DIVERSITY WINS**; ≤ −0.010 →
  **HOMOGENEITY WINS**; else **NULL**.
- **Skip mechanism gate** (required for any ENABLER/POSITIVE claim on AP):
  the printed `max |W_skip|` ≥ 0.01 in at least 2 of 3 AP members — the
  bypass must actually have grown.

**Power caveat** (standing): ±1.5 points on 3-seed means is a claim
throttle, not significance; single-evaluation ensemble comparisons carry
their own (reduced but nonzero) draw noise.

## 4 Predictions (falsifiable, before running)

- **W1:** AO NULL-to-NEGATIVE (point estimate ≈ −1): depth alone still
  doesn't pay — the AE pattern replicates.
- **W2:** mean(AP) ≥ mean(AO) + 0.010: the skip is a real enabler.
- **W3:** AP vs AK lands NULL — even enabled, a third layer does not beat
  two at this data/model scale.
- **W4:** AQ NULL vs AN (within ±0.010): diversity gain ≈ member-strength
  loss.
- **W5:** no trained run below 0.84; the skip gate passes in AP.

## 5 Disclosures

- AO, AP, and AQ have never been run on SHD before this freeze; the skip
  machinery ran only its synthetic-task gates.
- AQ retrains its members from scratch (no model persistence): its AK/AJ/X
  members are threads-16 bump-0 runs — deterministic re-realizations of
  previously recorded runs (AK 0.8812, AJ 0.8504, X 0.8884), disclosed so
  the member draws are known in advance. This makes AQ's member quality a
  KNOWN quantity: the diverse ensemble combines members at 0.8812 / 0.8504
  / 0.8884, mean 0.8733 — weaker on average than AN's members.
- Expected wall-clock: AO/AP ≈ 6 × 15 min; AQ ≈ 25 min; ~2 h all-in.
- Amendment policy: tighten-only; deviations recorded in the results
  document.

## 6 Consequence

- AP POSITIVE vs AK → skips + depth join the default recipe; a registered
  four-layer follow-up becomes eligible.
- ENABLER CONFIRMED but AP ≤ AK → skips work, depth is exhausted at this
  scale; the depth axis closes with a mechanism in hand for any future
  larger-data attempt.
- AO ≈ AP (enabler inconclusive/harmful) → the depth axis closes entirely;
  the harness keeps the machinery.
- AQ ≥ AN + 0.010 → the diverse ensemble becomes the campaign headline and
  a registered follow-up may combine diversity WITH member strength (three
  AK-family variants).
- Otherwise the homogeneous AN 0.9000 stands as the headline. With this
  round, every P1 item carries a registered outcome and the accuracy
  program concludes; remaining open work is P2/P3 (engineering and
  identification science) and the paper revision.
