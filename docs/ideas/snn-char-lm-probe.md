# SNN character-level language-model probe (the "snn-BERT" rung 1)

**Status:** IDEA — for future consideration, not registered, no runs
authorized. · **Discussed:** 2026-09-01 · **Origin:** /discuss session.

## The idea

The proposal arrived as "snn-BERT": build a BERT-class language model with
spiking neurons, in a new repository consuming the `kdmd-snn` crate,
motivated by work-context experience with BERT models plus the observation
that the brain runs on spikes and this project already taught a small SNN
to understand spoken digits (SHD campaign, 0.88 single / 0.9379 ensemble).

The discussion reframed it into a ladder whose first rung is deliberately
small and decisive: a **causal next-character SNN language model** on a
5–10 MB subset of text8, measured in bits-per-character against matched
baselines, with a pre-committed kill criterion. Rung 2 (masked prediction
with a bidirectional two-stack architecture, small BPE vocab) and rung 3
(a distilled mini-BERT) exist only if rung 1 wins. The probe is valuable
because it is asymmetric: a win licenses exactly one more rung, nothing
grander; a loss kills the entire ladder cheaply.

"Practical" was defined during discussion as a two-ledger question, frozen
before any run: **(1) task quality** — BPC versus a Kneser-Ney 5-gram, a
context-matched linear model, and a parameter-matched GRU; **(2) energy
proxy** — synaptic operations × spike rate versus MACs, the standard
neuromorphic accounting, with the explicit caveat that this ledger only
cashes out on neuromorphic hardware (Loihi-2 class); on CPU/GPU
simulation an SNN is strictly slower, and any efficiency number measured
in simulation is a fact about the simulator, not about spikes.

Settled during discussion: causal (not masked) at rung 1, since an SNN is
causal by construction and the baseline comparison is cleanest; compute
budget is "days of churn" on the Mac; baselines run in PyTorch (the
framework asymmetry is conservative — a maximally strong baseline makes
the kill decision more trustworthy).

## How it would be built

A new repo depending on `kdmd-snn`. Reused from the crate: the PSN layer
(`layer.rs` — the one component that makes text-length sequences
thinkable), surrogate-gradient training (`train/surrogate.rs`,
`train/optim.rs`), and the metrics scaffolding. New pieces, all small:

- **Corpus pipeline**: text8 subset, frozen split.
- **Input**: embedding lookup per character, injected as input *current*
  (spike-encoding of text is explicitly deferred — a separate research
  question).
- **Readout**: per-timestep 256-way softmax off membrane potential — the
  only genuinely new mechanism; existing readouts classify whole
  sequences.
- **Spike-rate logging from day one**: spikes-per-synapse-per-character,
  the raw material of the SOPs ledger, so even the probe answers the
  practicality question.

Model: 2–3 PSN layers, hidden widths in the few hundreds. Key novel
hyperparameter the SHD work never faced: **simulation steps per
character** — T internal steps per char multiplies sequence length by T.
Order of work: an unregistered wall-clock feasibility spike first (one
afternoon: seconds-per-1000-chars at candidate steps-per-char), then
freeze corpus/split/baselines/kill criterion, then run. Rough effort:
weeks for rung 1 including the PyTorch baseline jobs. If it is too slow,
`docs/24` (parallel-in-time assessment) and `docs/33` (PSN time-parallel
training plan) are the relevant accelerants.

## Known risks and open objections

- **Domain transfer.** The SHD result is a success in spiking's *native*
  domain — sound is a sparse temporal signal; the spiking cochleagram is
  text's opposite. Thirteen rounds (`docs/04`–`docs/32`) earned 0.88 on
  20 classes and 8k samples, with ±2.7–3.6 seed bands, and round 13
  (`docs/31`–`32`) found ensemble saturation even there. Nothing in the
  record says the machinery survives a jump to a 256-way decision at
  every one of millions of timesteps.
- **Prior art already answers the headline.** SpikeBERT and SpikingBERT
  exist: GPU-trained via distillation from ANN BERTs, lagging their
  teachers, efficiency claims hardware-hypothetical. The only novel
  question this project can ask is whether *its* machinery (PSN +
  surrogate training; Koopman-DMD analysis) adds anything.
- **Koopman relevance is unpromising for language.** Koopman theory wants
  smooth dynamics with a decent linear latent approximation; language is
  discrete, combinatorial, long-range. Rung 1 does not use the DMD
  readout at all — the crate contributes infrastructure, not its
  signature analysis. (Post-hoc DMD analysis of the trained net is a
  possible side study, not a claim.)
- **Inferential reach.** A rung-1 win says "spiking dynamics can carry
  linguistic information at character scale." It says nothing about
  attention-scale semantics, transfer, or BERT. The failure mode of probe
  experiments is a win narrated as more than it is; this paragraph is the
  antidote, written in advance.
- **Scope creep.** A days-scale budget breeds mid-run temptations (second
  corpus, fourth layer, a masking variant "while we're here"). The
  registration freezes one configuration ladder and the seed study,
  nothing else; rung 2 is where sanctioned curiosity goes.

Resolved in discussion: masked-vs-causal (causal; masking deferred to
rung 2), baseline framework purity (PyTorch fine — the asymmetry favors
the baseline, which is conservative), "beats a 5-gram" as the floor
(too low alone — replaced by the linear gate below).

## Edge cases

- **Spike-rate collapse or saturation.** Rates near 0 make the network a
  dead wire; rates near 1 make the SOPs ledger meaningless (no sparsity
  advantage) and the model effectively non-spiking. Correct behavior:
  report rates alongside BPC always; either extreme voids the efficiency
  ledger even if BPC is good.
- **Readout does all the work.** A per-timestep softmax off membrane
  potential could learn the task while the spiking dynamics contribute
  nothing. The context-matched linear gate exists to detect exactly this;
  correct behavior is to report the gate margin, not just the headline
  BPC.
- **Truncation mismatch.** BPTT truncation length must be identical for
  the SNN and the GRU baseline, and the linear baseline must see the same
  context window, or the comparison is void.
- **Published-number leakage.** Text8 BPC figures in the literature are
  for the full 100 MB. Every baseline must be recomputed on the identical
  subset and split; quoting a published number is a protocol violation.
- **Steps-per-char too costly.** If the feasibility spike shows the
  needed T makes days-scale runs impossible, the correct behavior is to
  stop and wait for `docs/33`-class speedups — not to shrink the corpus
  below where baselines are meaningful.
- **Reset-vs-no-reset class.** Round 12 (`docs/29`–`30`) priced the
  time-parallelizable (no-reset) class at −2.5 pts on SHD. Whether that
  tax transfers to char-LM is unknown; the configuration choice must be
  frozen before runs, not tuned after.

## Tests that would prove it out

- **Linear gate (grow-from-identity analog):** the SNN must first *match*
  a degenerate linear/embedding configuration, then beat a
  context-matched logistic regression on the same window. This is the
  **kill criterion**: if the SNN char-LM cannot beat the context-matched
  linear model in BPC by more than the seed band at rung-1 scale, the
  ladder dies.
- **Interesting comparison:** parameter-matched GRU at matched truncation,
  tuned honestly (Adam, standard init — its strength is conservative).
  Decision band: placeholder ±0.0X BPC, calibrated by the seed study.
- **Seed study:** 3–5 seeds on the final configuration to calibrate the
  BPC noise band — this task's equivalent of SHD's ±2.7–3.6. No
  comparison is interpretable before this exists.
- **Mechanism-engagement check:** spike-rate distribution per layer within
  sane sparsity bounds (placeholder: mean rate in [0.01, 0.5] — calibrate
  in the feasibility spike), demonstrating the network is neither dead
  nor saturated.
- **Energy ledger:** SOPs/char vs MACs/char for the GRU at matched
  quality, reported with the neuromorphic-hardware caveat attached.
- **Feasibility spike numbers** (seconds-per-1000-chars vs T) frozen into
  the registration before any training run.

## Compute placement

**Measured 2026-09-02** on the two machines available: this Mac (Apple
M5 Max, 18 cores, 128 GB) and the batcomputer (Intel i9-7940X, 14 cores /
28 threads, AVX-512, 62 GB, one TITAN Xp 12 GB, Ubuntu 22.04; dedicated,
nothing else runs on it). Same `shd_sweep` binary, same data, training
seconds from the run summary. Experiment A reproduced the Mac's accuracy
and loss to four decimals across architectures; AK did **not** (Mac
0.8812 / loss 0.1212, batcomputer 0.8884 / 0.1709 — inside the seed band,
but a different trajectory), so the cross-architecture divergence appears
once the matmuls are large enough to hit faer's arch-specific SIMD
kernels, whose rounding differs.

| job | Mac | batcomputer | ratio |
|---|---|---|---|
| A (1×128, 1500 mb), 1 thread | 22.9 s | 87.9 s (81.3 s with `-C target-cpu=native`) | 3.8× (3.5×) |
| A, 16 threads | 6.8 s | 43.2 s (32 threads: 38.0 s) | 6.4× |
| AK (2×256, 6000 mb, 200 steps), 16 threads | 607.6 s | 2059.9 s | 3.4× |

What the numbers say:

- **Per core, the Mac is ~3.8× faster** on the crate's kernels, and
  **3.4× on the real 16-thread workload**. AVX-512 buys 8 %: the kernels
  are not vectorization-bound, consistent with the cache-hostile per-step
  loops `docs/33` already flagged.
- **Thread scaling on the batcomputer is poor, and the cause is the
  SHD pipeline, not the trainer.** A gains 1.3× / 1.8× / 1.9× / 2.1× at
  4 / 8 / 16 / 32 threads (the Mac gains 3.4× at 16 on the same too-small
  job); the AK run shows ~650 % CPU on 16 threads. jemalloc via
  `LD_PRELOAD` changed nothing (81.8 s serial, 43.8 s at 16 threads), so
  the allocator is exonerated. The residual is the serial section every
  minibatch in `shd_sweep` (`train_member`): zero-fill ~18 MB of dense
  per-step input matrices, bin and scatter 32 event streams, then after
  the parallel gradient chunks a serial reduction and Adam update. That
  work is memory-bound, and DDR4 pays for it far more than the M5 Max's
  unified memory does. Lesson for the char-LM repo, whose pipeline is
  new anyway: assemble batches once (or inside the parallel section),
  and measure thread scaling on both machines in the feasibility spike
  rather than assuming the Mac's curve.
- **RAM is not a differentiator** (128 GB here vs 62 GB there); neither
  machine is memory-bound on this workload.
- **The GPU is not a target for the crate as written.** The crate is f64
  throughout; Pascal runs f64 at 1/32 rate (≈0.4 TFLOPS). The "GPU port"
  pick-up trigger below implies an f32 port first, which is its own
  numerical question.

Placement rules that follow (freeze into the registration; state machine,
thread count, and build flags per arm, since threaded runs are
deterministic per thread count but not bit-identical across counts):

1. **SNN training goes on the Mac by default.** A job that takes two days
   here takes about a week on the batcomputer at the measured 3.4×.
   The batcomputer earns SNN work only when the Mac is needed for
   something else — its one advantage is that it costs nothing from the
   Mac — and only for jobs nobody is waiting on: seed-study members, the
   final registered run. Interactive work (the feasibility spike, the
   configuration ladder) stays on the Mac.
2. **PyTorch baselines (linear gate, parameter-matched GRU) go on the
   batcomputer's GPU**, concurrently with SNN training on the Mac. A
   TITAN Xp is ample for a GRU on 5–10 MB of text (use the cu126 wheels;
   newer wheels dropped sm_61). This removes the baseline jobs from the
   serial "weeks for rung 1" estimate, which is the biggest schedule win
   available. The Kneser-Ney 5-gram runs anywhere.
3. **Split the seed study across both machines** rather than serializing
   it: while the Mac runs three seeds back to back, the batcomputer
   finishes about one. Cross-machine members are **not** bit-reproducible
   (AK above), so the registration must name the machine per seed and
   treat it as part of that seed's identity — a free hardware-robustness
   check, provided nobody later re-runs a batcomputer seed on the Mac and
   calls the difference a result.
4. **Checkpoint/resume is a precondition for any multi-day placement**,
   and doubly so remote: the crate has none today (no save/restore of
   weights, optimizer moments, or RNG state), the batcomputer has no
   tmux (jobs run under `nohup`, polled by `tail`), and a reboot or
   sleep loses everything. Save every N minibatches with a `--resume`
   flag before the first registered run, on either machine.
5. **Build wrinkle, recorded so it is not rediscovered:** the crate's
   static HDF5 (`datasets` feature) needs cmake ≥ 3.26 and the
   batcomputer ships 3.22 with no sudo; a `pip install cmake` into the
   per-project conda env (`./.conda/bin/cmake` on `PATH`) fixes it. The
   char-LM repo itself reads plain text and never needs HDF5.

Consequence for the pick-up triggers: "idle days-scale compute" is now
partly satisfied on the hardware side — the batcomputer is idle by
construction — but at 3.8× it changes what the days buy, not whether the
kill criterion must be frozen first.

## Pick-up triggers

- The `docs/33` PSN time-parallel training plan lands, making
  text-length sequences several-fold cheaper.
- A GPU/SIMD port of the crate's forward pass.
- Work context makes the SNN-for-language question concretely valuable
  (e.g., an energy-constrained inference target with a neuromorphic
  hardware path).
- Idle days-scale compute and the itch — the probe is deliberately cheap
  enough that curiosity alone is a legitimate trigger, *provided* the
  kill criterion is frozen first.
