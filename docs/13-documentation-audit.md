# Documentation Audit — Phase 6 Close-Out

**Document:** 13-documentation-audit.md · **Author:** Documentarian subagent · **Date:** 2026-08-15
**Mandate:** verify the repository's documentation is accurate against the code and sufficient for a newcomer to build, run, and understand the project — including its negative results.

## What was checked

1. **README.md vs the code.** Every API name in the quick-start snippets was
   verified against the source: `Lif::new(LifParams)`, `KoopmanLayer::lif`,
   `Network::new`/`step`, `SpikeVec::from_indices`/`count`, `Trainer::new`/
   `train_step`, `StepStats.loss`/`.accuracy`, `fit_controlled(&snapshots,
   Some(b), None, &IdentifyConfig::default())`, `lif_structural_b`. All
   signatures and shapes (w: n_out×n_in) match. The claims table was checked
   against docs/09 (1.02× gate result, spike-for-spike equivalence) and the
   V2/V2b negative-result docs (docs/05, docs/08) — phrasing matches the
   recorded verdicts; no overclaiming found.
2. **Rustdoc.** `cargo doc --no-deps` — zero warnings, before and after this
   pass (no broken intra-doc links). Every pub module carries an orienting
   module-level doc comment (spot-checked all 14); no source edits were needed.
3. **docs/ coherence.** The numbered docs 01–09 tell a complete story
   (premise → skeptic re-scope → two pre-registered failures → benchmark
   gate); an index was missing.
4. **IMPLEMENTATION_PLAN.md** status header still said "Approved", predating
   completion.
5. **SHD demo.** Instructions executed as written from the repository root:
   with `data/shd/shd_train.h5` + `shd_test.h5` present the download is
   skipped, 8156 train / 2264 test samples load, and training proceeds
   (verified through minibatch 100, then stopped). The demo resolves
   `data/shd/` relative to the cwd — now documented.
6. **Build reproducibility for others.** The critical gap: `koopman-dmd =
   "0.2"` is unpublished and `[patch.crates-io]` points at a sibling
   `../rust-dmd` checkout whose `dmdc` branch is local-only, so **a fresh
   clone cannot build**. This was not stated anywhere user-facing.

## What was fixed

- **README.md**
  - New prominent **"Building — read this first"** section: the unpublished
    `koopman-dmd` 0.2 dependency, the required sibling-checkout layout, the
    fact that the `dmdc` branch is unpushed (so third parties are blocked
    until the owner pushes/publishes), the CI implication, MSRV 1.85, and the
    cmake-for-`datasets` prerequisite.
  - Quick-start snippet rewritten as a `fn main() -> Result<(), SnnError>`
    block, replacing the doctest-only `# Ok::<(), …>(())` hidden line, which
    renders as literal text on GitHub (README blocks are not doctested here).
  - SHD demo section: run-from-repo-root requirement, `curl`/`gunzip`
    dependency, and download-skip behavior documented.
  - Repository map now points at the new `docs/README.md` index and lists the
    Phase 6 review reports (docs/10+).
- **docs/README.md** (new): reading-order index of docs 01–13 with one-line
  descriptions; 10–12 listed as close-out review reports (in progress at
  audit time).
- **IMPLEMENTATION_PLAN.md**: Status header updated to COMPLETE (Phases 0–6),
  recording the V2/V2b negative results, the shipped scope, and the remaining
  release loose ends. Only the Status paragraph was touched; the original
  approval note is preserved inside it.
- **No code, tests, or numbered docs 01–09 were modified.** No rustdoc
  changes were needed (already warning-free).

## Verification after edits

- `cargo test --release` — all suites pass (see repository CI/test log).
- `cargo doc --no-deps` — warning-free.
- SHD demo smoke run — loads and trains with the pre-downloaded files.

## Remaining for the owner (release checklist)

1. **Push the `rust-dmd` `dmdc` branch** and open the PR (plan decision Q1);
   until then nobody else can build this repository at all.
2. **Publish `koopman-dmd` 0.2.0** to crates.io, then delete the
   `[patch.crates-io]` section from the root `Cargo.toml` (its own comment
   says to remove it before publishing `kdmd-snn`).
3. **CI** strips the patch and expects `koopman-dmd = "0.2"` from crates.io —
   it will fail until (2) lands; the interim options (checkout-alongside or a
   git dependency) are already noted in `.github/workflows/ci.yml`.
4. Optionally publish `kdmd-snn` 0.1.0 after (2); `Cargo.toml` metadata
   (description, license, keywords, categories, MSRV) is already in place.
