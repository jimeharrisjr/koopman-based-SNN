# Round 11: Diversity WITH Member Strength — Results

**Protocol:** `docs/27-round11-preregistration.md` (frozen, committed and
pushed before these runs) · **Date run:** 2026-08-28 ·
**Raw log:** `demo/sweep-AR-AS-log.txt` · **Command:** exactly as registered.
**Deviations: none.** (An initial launch was killed ~1 minute in by an
operator timeout before any member finished training; the run was restarted
from scratch. No results existed from the aborted launch.)

## VERDICTS: AR POSITIVE at 0.9366 — THE 0.92 MILESTONE FIRES · AS POSITIVE

| arm | members (known draws) | result | Δ vs AQ (0.9179) | verdict |
|---|---|---|---|---|
| **AR** | AK (0.8812), AP (0.9018), X (0.8884) | **0.9366** (2098/2240) | **+0.0187** | **POSITIVE** |
| AS | + AJ (0.8504) | 0.9335 (2091/2240) | +0.0156 | POSITIVE |

**Diversity and member strength stack.** Swapping AQ's weakest member for
the strongest recipe on record lifted the diverse ensemble from 0.9179 to
**0.9366** — 4.7 points above its own mean member strength and 3.5 above
its best member. The concern registered in Z1 (AP shares AK's readout and
bin width, so correlation might eat the strength gain) did not materialize:
the three architectures still disagree usefully on enough samples.

**The milestone, five rounds late.** Round 5 (docs-era `demo/RESULTS.md`)
registered a target of 0.92 and concluded it was "not a sweep away — a
library-feature project." That diagnosis is now formally vindicated: the
target fell at 0.9366, and every feature between there and here — learnable
τ, attention, 5 ms bins, skips, ensemble machinery — was a library feature,
built, gated, and registered.

**The fourth member subtracts.** AS (adding AJ, the 0.8504 draw) came in
0.3 points *below* AR — prediction Z2 refuted. At weak member strength
(round 10), adding decorrelated members paid; at strong member strength, a
markedly weaker member dilutes more than its decorrelation contributes.
The refined law across rounds 9–11: an ensemble buys the most when its
members are *strong AND fail differently* — strength and diversity are
complements, not substitutes, and the weakest member sets a soft floor on
what its vote is worth.

## Predictions scorecard — 3 of 4

| # | Prediction | Outcome |
|---|---|---|
| Z1 | AR ≥ 0.920 (POSITIVE) | **CONFIRMED** — 0.9366 |
| Z2 | AS ≥ AR | **REFUTED** — −0.3: the weak member dilutes at strength |
| Z3 | neither arm below 0.908 | **CONFIRMED** — minimum 0.9335 |
| Z4 | milestone ≥ 0.9200 fires | **CONFIRMED** — both arms clear it |

## Campaign scoreboard after eleven rounds

| headline | value | provenance |
|---|---|---|
| **Best honest number** | **0.9366** | AR, diverse-strong ensemble (this round) |
| Prior diverse ensemble | 0.9179 | AQ (round 10) |
| Homogeneous ensemble | 0.9000 | AN (round 9) |
| Best 3-seed mean | 0.8935 (AP) / 0.8888 (AK) | round 10 / 8 |
| Best single training run | 0.9116 | AK seed 200 (round 8) |
| Starting point | 0.502 | first demo |

0.9366 sits in the upper region of the published state-of-the-art band
(0.90–0.94), reached by subtractive-reset LIF networks with exact linear
sub-threshold dynamics, hand-rolled surrogate BPTT, and ~50 minutes of
laptop training for the winning ensemble's members.

## Consequence (per docs/27 §5)

- **AR is the new campaign headline (0.9366)**; the round-5 0.92 target is
  formally recorded as reached.
- The ensemble axis rests here: the member pool is exhausted of
  qualitatively distinct strong recipes, and the next increment would need
  either new architectures or member-selection studies — neither currently
  registered.
- The paper's headline (revised to 0.918 earlier today) trails the record
  again; its next revision should carry 0.9366 and the
  strength-and-diversity-are-complements law.
