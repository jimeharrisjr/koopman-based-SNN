# docs/ — reading order

The numbered documents tell the project story end to end: premise → adversarial
re-scoping → two pre-registered experiments (both negative) → the benchmark
gate that fixed the final scope → close-out reviews. Raw experiment output is
kept next to each results document.

| Doc | One-line description |
|---|---|
| [01-scientific-foundations.md](01-scientific-foundations.md) | Literature survey and mathematical basis (Koopman/DMD ↔ LIF linearity), with verified citations and the novelty position. |
| [02-architecture.md](02-architecture.md) | System design: crate layout, data structures, the linear-advance + threshold split, DMDc integration, faer conventions. |
| [03-skeptic-review.md](03-skeptic-review.md) | Adversarial review of the premise; the concerns (C1–C…) that re-scoped the project onto value cases V1–V4. |
| [04-v2-preregistration.md](04-v2-preregistration.md) | Frozen protocol for V2: EDMD lifted surrogates of the Izhikevich neuron (tasks, baselines, metrics, kill criteria). |
| [05-v2-results.md](05-v2-results.md) | V2 verdict: **FAIL** — cumulative phase drift breaks ±2 ms spike timing. Raw output: [05-v2-results-raw.txt](05-v2-results-raw.txt). |
| [06-skeptic-v2-review.md](06-skeptic-v2-review.md) | Skeptic audit of the V2 failure itself — was the negative result real? (Yes; findings F1–F8.) |
| [07-v2b-preregistration.md](07-v2b-preregistration.md) | Frozen protocol for V2b, the owner-approved bounded rescue: ISI/Poincaré return-map surrogate. |
| [08-v2b-results.md](08-v2b-results.md) | V2b verdict: **FAIL** — exact timing inside the training envelope, breaks at its edges. Raw output: [08-v2b-results-raw.txt](08-v2b-results-raw.txt). |
| [09-phase5-benchmarks.md](09-phase5-benchmarks.md) | Phase 5 benchmark gate: structured fast path at 1.02× the reference simulator (PASS); dense fitted-operator inference demoted (FAIL). |
| 10–12 | Phase 6 close-out review reports (code quality, findings, and adequacy — being filed by the final review passes). |
| [13-documentation-audit.md](13-documentation-audit.md) | Documentation audit: what was checked and fixed, and the release loose ends left for the owner. |

Companion documents at the repository root: `SNN-project.md` (the original
premise) and `IMPLEMENTATION_PLAN.md` (the phased plan, owner decisions Q1–Q8,
and the pre-registered kill criteria).
