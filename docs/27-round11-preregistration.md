# Round 11: Diversity WITH Member Strength — Pre-registration

**Document:** 27-round11-preregistration.md · **Status: FROZEN before any
gated run** · **Date frozen:** 2026-08-28
**Origin:** the registered follow-up unlocked by round 10 (docs/23
§consequence): the diverse ensemble AQ won at 0.9179 *despite* weak members
(mean member strength 0.873); this round tests whether diversity and member
strength stack.

## 1 Hypothesis and rationale

AQ's members were AK (0.8812), AJ (0.8504), and X (0.8884) at seed_bump 0.
Since round 10, the strongest single recipe on record is AP (three layers
with zero-init skips; bump-0 draw 0.9018). Because threads-16 runs are
deterministic re-realizations, **every member draw in this round is known in
advance** — the only unknowns are the ensemble sums.

**Y1 (strength):** replacing AQ's weakest member (AJ) with the strongest
recipe (AP) improves the diverse ensemble.
**Y2 (more diversity at strength):** keeping AJ as a fourth member alongside
the three strong ones does not hurt, and plausibly helps.

## 2 Arms (frozen)

| arm | members (bump-0 draws, all known) | member mean |
|---|---|---|
| `AR` | AK (0.8812), **AP (0.9018)**, X (0.8884) | 0.890 |
| `AS` | AK (0.8812), AP (0.9018), AJ (0.8504), X (0.8884) | 0.880 |
| AQ | **reused** (round 10): 0.9179 | 0.873 |

Architectural diversity retained: AR spans 2-layer-combination /
3-layer-skip / vanilla-count-10ms (readouts attention/attention/count; bins
5/5/10 ms; τ learned/learned/fixed); AS adds AJ's 10 ms attention. Members
train at their own configs, logits summed at evaluation (the AQ machinery,
now parameterized by arm).

Command:

```
cargo run --release --features datasets --example shd_sweep -- AR AS --threads 16
```

**Rules (frozen), each arm vs AQ = 0.9179:**
- ≥ +0.010 → **POSITIVE** · ≤ −0.010 → **NEGATIVE** · else **NULL**.
- Secondary comparison AS − AR reported descriptively (four vs three
  members at strength).
- **Milestone flag:** any arm ≥ **0.9200** closes the loop on round 5's
  registered-and-missed 0.92 target, five rounds and one library era late.

## 3 Predictions (falsifiable, before running)

- **Z1:** AR ≥ 0.920 (POSITIVE): member strength adds on top of retained
  architectural diversity. Tempered by the known risk: AP shares AK's
  readout and bin width, so AR's members correlate more than AQ's did.
- **Z2:** AS ≥ AR: at matched strength, a fourth decorrelated member helps
  or is neutral (standard ensemble behavior).
- **Z3:** neither arm below 0.908 (AQ − 0.010): adding strength to a
  working diverse ensemble should not break it.
- **Z4:** the milestone fires (≥ 0.9200) in at least one arm.

## 4 Disclosures

- Neither AR nor AS has ever been evaluated; member draws are known from
  rounds 8/10 by determinism (disclosed above), the sums are not.
- Expected wall-clock: AR ≈ 31 min, AS ≈ 38 min at 16 threads (members
  retrained; no model persistence).
- Amendment policy: tighten-only; deviations recorded in the results
  document.

## 5 Consequence

- Any POSITIVE → that arm becomes the campaign headline; if the milestone
  fires, round 5's 0.92 target is formally reached and recorded as such.
- Both NULL → diversity saturates near 0.918 at this member pool; the
  ensemble axis closes with AQ standing.
- Any NEGATIVE → member correlation (AK/AP shared design) outweighs
  strength — itself the informative outcome, sharpening the round-10
  finding to "diversity of *failure modes*, not accuracy, is what an
  ensemble buys."
