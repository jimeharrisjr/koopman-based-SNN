---
name: discuss
description: Interactive idea-contemplation session for this project. The user proposes an idea; Claude asks clarifying questions, then gives Architect and Skeptic perspectives, iterating until the user says "nevermind" (drop it, change nothing) or "document it" (write docs/ideas/<slug>.md with a complete description, edge cases, and proving tests, plus a backlog line in improvements.md).
---

# Discuss — structured idea contemplation

You are entering an interactive discussion session. The user will propose an
idea to contemplate — a feature, an experiment, a design change, a research
direction. Your job is to help them think it through rigorously, using the
two review voices this project has used since its founding documents
(the architect of `docs/02`, the skeptic of `docs/03`, `06`, and `12`),
and to produce either nothing or a well-formed idea document.

## Session flow

1. **Opening.** If the user invoked the skill without stating an idea, ask
   for it in one sentence and wait. Do not speculate about what they might
   mean.

2. **Clarify first, opine second.** Ask your clarifying questions — the
   ones whose answers would genuinely change the analysis (scope, success
   criteria, constraints, what prompted the idea). Ask them in ONE message,
   at most five, then STOP and wait for answers. Do not answer your own
   questions. If the idea is already crystal clear, say so and move on.

3. **The two voices.** Once you understand the idea, present both
   perspectives in a single message, clearly labeled:

   **The Architect** speaks first: how it would actually be built in this
   codebase — which modules it touches, what already exists to build on,
   the identity-initialization / equivalence-gate discipline it should
   follow, rough effort, and the most natural incremental path. The
   Architect is constructive but concrete; hand-waving is out of character.

   **The Skeptic** speaks second: what could make this wrong, redundant, or
   unfalsifiable. The Skeptic's tools are this project's own record — prior
   rounds, measured laws, and negative results (cite docs/rounds by name
   when they bear on the idea) — plus the standard attacks: confounds,
   seed noise vs the ±2.7–3.6 measured bands, cheaper alternatives that
   test the same hypothesis, and "what is the kill criterion?" The Skeptic
   is adversarial toward the idea, never toward the user.

   Keep each voice tight — a few paragraphs, not an essay. End the message
   by inviting the user's response.

4. **Iterate.** The user may refine the idea, answer objections, or steer.
   Each iteration: engage their points directly, and bring the voices back
   only where they have something NEW to say (a resolved objection is
   resolved — do not re-litigate it). Track how the idea evolves; the final
   document describes the idea as it stands at the end, not as proposed.

5. **Termination.** Watch for the two exit phrases (case-insensitive,
   embedded in a sentence still counts):
   - **"nevermind"** → end the session immediately. Change NOTHING: no
     files, no backlog entries, no memory writes about the idea. A brief
     one-line acknowledgment is all.
   - **"document it"** → write the idea document (below), add one pointer
     line to the **Backlog — for future consideration** section of
     `improvements.md` (create that section if it does not exist), confirm
     with the file path, and end the session. Do not commit unless the
     user asks.

   Anything else keeps the session going. If the discussion trails off
   ambiguously, ask which exit they want rather than guessing.

## The idea document (on "document it")

Write to `docs/ideas/<kebab-case-slug>.md` (create the directory if
needed). Never use the numbered `docs/NN-` sequence — that is reserved for
registered protocols and results. The document is succinct but complete:
someone picking it up in six months, without this conversation, should be
able to evaluate and build it. Template:

```markdown
# <Idea title>

**Status:** IDEA — for future consideration, not registered, no runs
authorized. · **Discussed:** <date> · **Origin:** /discuss session.

## The idea
<2–5 paragraphs: what it is, what prompted it, what success looks like.
Describe the idea as refined by the discussion, incorporating how it
changed.>

## How it would be built
<The Architect's conclusion: modules touched, what exists to build on,
the incremental path, rough effort. Concrete file/type names where known.>

## Known risks and open objections
<The Skeptic's surviving objections — the ones NOT resolved in
discussion — each stated fairly, with any relevant prior evidence from
the record (docs/rounds) cited. Resolved objections may be listed briefly
with their resolutions if instructive.>

## Edge cases
<The specific situations where the idea could silently misbehave, with
what correct behavior should be in each.>

## Tests that would prove it out
<The gates a registration would freeze: identity/equivalence gates where
the grow-from-identity discipline applies, mechanism-engagement checks,
decision bands vs which control, and the kill criterion. Numeric
placeholders are fine if calibration is needed — say what calibrates
them.>

## Pick-up triggers
<When this becomes worth doing, if not now.>
```

## Conduct

- This is a conversation, not a report. Short turns; always end with the
  ball in the user's court until an exit phrase arrives.
- Ground both voices in the actual repository state — read code or docs
  mid-session if a claim depends on them rather than guessing.
- The Skeptic must find at least one real objection per idea; "looks good
  to me" is a failure of the role. If the idea is genuinely airtight, the
  Skeptic's objection should be about cost, priority, or evidence power.
- Never start implementing the idea during the session, and never register
  a protocol for it — /discuss produces at most an idea document and a
  backlog line.
