# Scientific Foundations: Koopman-DMD for Spiking Neural Networks

**Document:** 01-scientific-foundations.md
**Author:** Scientist subagent (literature survey conducted 2026-08-14)
**Status:** All citations below were located and verified via web search on the survey date. Where a claim rests on secondhand reporting rather than a directly inspected source, this is stated explicitly.

---

## 1. Mathematical Foundation

### 1.1 The Koopman operator

For a discrete-time dynamical system $x_{t+1} = F(x_t)$, $x \in \mathcal{M} \subseteq \mathbb{R}^n$, the Koopman operator $\mathcal{K}$ acts on scalar observable functions $g : \mathcal{M} \to \mathbb{C}$ by composition:

$$(\mathcal{K} g)(x) = g(F(x)), \qquad \text{i.e.} \qquad g(x_{t+1}) = (\mathcal{K} g)(x_t).$$

$\mathcal{K}$ is **linear** on the (generally infinite-dimensional) function space even when $F$ is nonlinear — this is the foundational trade: nonlinear finite-dimensional dynamics for linear infinite-dimensional dynamics (Koopman 1931; modern treatment in Mezić 2005; comprehensive review in Brunton, Budišić, Kaiser & Kutz 2022). If $\mathcal{K}$ has eigenfunctions $\varphi_j$ with eigenvalues $\mu_j$ ($\mathcal{K}\varphi_j = \mu_j \varphi_j$), any observable in their span evolves as a superposition of geometric/exponential modes:

$$g(x_t) = \sum_j v_j \varphi_j(x_0)\, \mu_j^t,$$

where $v_j$ are Koopman modes. The eigenvalue moduli $|\mu_j|$ are decay/growth rates; their arguments are oscillation frequencies.

**Caveats the project must respect** (Brunton et al. 2022; Korda & Mezić 2018a):

- A finite-dimensional *invariant* subspace containing the state observables exists only for special systems. Systems with multiple isolated fixed points, limit cycles, or chaos cannot be represented globally by a finite linear system in any smooth lifting that includes the state itself.
- Systems with **discontinuous** maps — which is what an LIF network with hard reset is (a piecewise-affine, discontinuous map) — fall outside the smooth-dynamics theory where most convergence results live. The Koopman operator still exists, but finite-rank approximations across the discontinuity behave like linear regressions through a jump: they average over it rather than represent it. This is precisely why the project isolates the threshold as an explicit nonlinearity rather than asking DMD to absorb it. That design choice is sound and is the same one made in Korda & Mezić (2018b) for control ("linear dynamics + isolated nonlinearity").

### 1.2 DMD

Given snapshot matrices $X = [x_0, \dots, x_{m-1}]$ and $Y = [x_1, \dots, x_m]$, DMD finds the best-fit linear operator

$$A = \arg\min_A \|Y - AX\|_F = Y X^{\dagger},$$

computed via the (truncated, rank-$r$) SVD $X \approx U_r \Sigma_r V_r^{*}$ (Kutz, Brunton, Brunton & Proctor 2016). Two operators must be kept distinct — the project premise document conflates them slightly:

- **Projected operator** (what you compute): $\tilde{A} = U_r^{*} Y V_r \Sigma_r^{-1} \in \mathbb{R}^{r \times r}$.
- **Full-state approximation**: $A \approx Y V_r \Sigma_r^{-1} U_r^{*} \in \mathbb{R}^{n \times n}$.

The premise's formula $\tilde{A} = U^{T} Y V \Sigma^{-1}$ is the projected $r \times r$ operator, not the full $A$; a state update in the original coordinates requires either lifting back ($x_{t+1} \approx U_r \tilde{A} U_r^{*} x_t$) or working entirely in the reduced coordinate $z_t = U_r^{*} x_t$. The reduced coordinate is the computationally interesting one (see §3.2). DMD eigenvalues/eigenvectors approximate Koopman eigenvalues/modes when the observables (here, the raw state) sit near a Koopman-invariant subspace (Rowley et al. 2009, cited via Mezić 2005 lineage; Korda & Mezić 2018a for convergence).

### 1.3 EDMD

Extended DMD (Williams, Kevrekidis & Rowley 2015) generalizes this by lifting the state through a dictionary $\Psi(x) = [\psi_1(x), \dots, \psi_N(x)]^{T}$ and regressing

$$\Psi(x_{t+1}) \approx K\, \Psi(x_t), \qquad K = \arg\min_K \sum_t \|\Psi(x_{t+1}) - K \Psi(x_t)\|^2 .$$

EDMD converges to the $L^2(\rho)$-orthogonal projection of $\mathcal{K}$ onto the dictionary span as data grows, and to $\mathcal{K}$ itself as the dictionary becomes complete (Korda & Mezić 2018a). With the dictionary $\Psi(x) = x$, EDMD reduces exactly to DMD. **Dictionary choice is the entire game** for nonlinear systems; for spiking models, candidate dictionaries include polynomials in $(V, w)$ (AdEx/Izhikevich state), exponential/softplus functions of $V - V_{th}$, delay embeddings of the membrane potential, and per-neuron radial basis functions.

### 1.4 DMD with control (DMDc)

Proctor, Brunton & Kutz (2016) extend DMD to actuated systems — exactly the premise's equation:

$$x_{t+1} = A x_t + B u_t.$$

Stack $\Omega = \begin{bmatrix} X \\ \Upsilon \end{bmatrix}$ with $\Upsilon = [u_0, \dots, u_{m-1}]$ and solve $[A\;\; B] = Y\, \Omega^{\dagger}$ via SVD of $\Omega$ (with a second SVD of $Y$ for order reduction). Two variants matter for this project:

- **Known-B DMDc**: when the input matrix is known (here: the synaptic weight matrix $W$ and the reset magnitude are *known by construction*), one solves only $A = (Y - B\Upsilon) X^{\dagger}$. This is the variant the SNN library should default to — see §3.5 on identifiability.
- **Unknown-[A,B] DMDc**: the joint regression, needed only if the effective input coupling is itself being discovered.

The controlled-Koopman generalization (Koopman with inputs and control, KIC; Proctor, Brunton & Kutz 2018) and the lifted linear-predictor framework of Korda & Mezić (2018b) put EDMD + control on firmer footing: lift the state, keep the input linear, i.e. $z_{t+1} = A z_t + B u_t$, $z = \Psi(x)$, $\hat{x} = C z$. Korda & Mezić prove this class of predictors and use it for MPC. **This — EDMDc on lifted observables — is the mathematically correct statement of the project's core object.** Note that for control-affine continuous-time systems the exact lifted dynamics are generally **bilinear** ($z_{t+1} = A z_t + \sum_i u_{t,i} N_i z_t$), not linear in $u$; the linear-in-$u$ model is an additional approximation that is exact for LIF with current-based synapses (see §2) and approximate for conductance-based synapses.

### 1.5 Surrogate gradients

The spike generation $s_t = \Theta(V_t - V_{th})$ has derivative zero almost everywhere and undefined at threshold, so BPTT through an SNN produces zero gradients. The surrogate gradient (SG) method replaces $\Theta'$ only in the backward pass with a smooth surrogate $\sigma'(V - V_{th})$ — fast sigmoid $\left(1/(1+\beta|v|)\right)^2$ (SuperSpike; Zenke & Ganguli 2018), rectangular windows, exponential kernels (SLAYER; Shrestha & Orchard 2018), etc. The canonical reference and empirical treatment is Neftci, Mostafa & Zenke (2019). Key facts from that literature relevant here:

- SG-BPTT gradients for a recurrent SNN over $T$ steps involve products $\prod_{k} J_k$ where each step Jacobian $J_k = \frac{\partial x_{k+1}}{\partial x_k}$ contains (i) the linear sub-threshold propagator and (ii) surrogate-derivative factors at threshold crossings. Both factors shape gradient magnitude.
- Alternatives exist and should be benchmarks: **e-prop** (Bellec et al. 2020) factorizes the gradient into forward-computable eligibility traces (no backward unroll — biologically plausible, hardware friendly, approximate); **EventProp** (Wunderlich & Pehle 2021) computes *exact* gradients of spike-time-dependent losses via the adjoint method with jump conditions at spike events. EventProp is intellectually close to this project: it also treats the dynamics between spikes as exactly integrable (linear) and handles spikes as discrete events with derivative jump rules.

### 1.6 Gradient propagation and the spectrum of A — a correction to the premise

The premise claims the Koopman matrix $A$ "provides a stable, mathematically rigorous Jacobian … preventing exploding or vanishing gradients over long time windows." **The literature does not support "preventing."** The classical analysis (Pascanu, Mikolov & Bengio 2013) shows that for a recurrence with Jacobian $J$, long products $\prod_{k=1}^{T} J_k$ vanish when the largest singular value / spectral radius is $< 1$ and explode when $> 1$; the condition is spectral, not structural. Applied here:

- The backward pass through the linear part of the Koopman-SNN layer multiplies by $A^{T}$ (transpose powers). Gradient norms scale as $\|A^{\top T}\| \sim \rho(A)^T$ up to non-normality effects.
- Any leaky (dissipative) neural system has $\rho(A) < 1$ — that is what "leak" means. So gradients through the linear part **still vanish geometrically** over long horizons, exactly as in a standard LIF simulation with decay factor $\alpha = e^{-\Delta t/\tau_m}$. A DMD-fit $A$ of a leaky network cannot have $\rho(A) \geq 1$ unless the data contain sustained/growing modes.
- What Koopman/DMD *actually* offers is **diagnosis and design, not automatic prevention**: the DMD eigenvalues $\{\mu_j\}$ are directly readable. One can (a) report per-mode gradient decay timescales $\tau_j = -\Delta t / \ln|\mu_j|$, (b) detect near-unit-circle modes that carry long-range credit, and (c) *constrain* the fitted spectrum (e.g., project eigenvalues toward the unit circle, as in unitary/orthogonal RNN literature) as an explicit regularization. This is a real and defensible contribution — but it must be stated as spectral control, not as an intrinsic property of the Koopman formalism.
- The surrogate-derivative factors at threshold crossings multiply into the same product and are typically the *dominant* source of gradient attenuation/instability in deep SNNs (Neftci et al. 2019 discuss surrogate steepness trade-offs). Linearizing the sub-threshold part does not touch this.

**Recommendation:** rewrite the premise claim as: "the DMD spectrum makes the gradient propagation properties of the linear part explicit and controllable ($\|\partial x_{t+T}/\partial x_t\| \sim \rho(A)^T$), enabling principled spectral regularization; the threshold surrogate factors remain the separate, dominant nonlinear influence."

---

## 2. Neuron Model Landscape — Where Linearity Already Holds

### 2.1 LIF sub-threshold dynamics are already exactly linear

This is the single most important fact for honest positioning of the project. The LIF membrane equation

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R I(t)$$

is a linear ODE. Its exact discretization (zero-order hold on input) is

$$V_{t+1} = \alpha V_t + (1-\alpha)(V_{rest} + R I_t), \qquad \alpha = e^{-\Delta t / \tau_m},$$

and with current-based exponential synapses ($\tau_s \dot{I} = -I + W s^{(l-1)}(t)$) the joint state $x_t = (V_t, I_t)$ of an $N$-neuron layer evolves **exactly** as

$$x_{t+1} = A_{\text{exact}}\, x_t + B_{\text{exact}}\, u_t, \qquad
A_{\text{exact}} = \begin{bmatrix} \alpha I & \beta' I \\ 0 & \beta I \end{bmatrix},\;
u_t = \begin{bmatrix} s^{(l-1)}_t \\ s^{(l)}_t \end{bmatrix},$$

with $\beta = e^{-\Delta t/\tau_s}$, feed-forward spikes entering through $W$ and the neuron's own output spikes entering through the reset term. $A_{\text{exact}}$ is known in closed form, diagonal per neuron, and requires no data, no SVD, and no regression. **DMD applied to a standard current-based LIF layer will simply re-estimate these known decay constants, with sampling error.** Any claim that DMD "linearizes" plain LIF is claiming credit for a linearity that is already there analytically.

### 2.2 The reset as an impulse input: the Spike Response Model already did this

The Spike Response Model (SRM; Gerstner & Kistler 2002, Ch. 4; Scholarpedia "Spike-response model") writes the membrane potential as a sum of linear filter responses:

$$V_i(t) = \sum_{\hat{t}_i} \eta(t - \hat{t}_i) + \sum_j w_{ij} \sum_{\hat{t}_j} \kappa(t - \hat{t}_j) + V_{rest},$$

where $\eta$ is the reset/after-potential kernel triggered by the neuron's own spikes and $\kappa$ the postsynaptic potential kernel. This is exactly "resets folded in as impulse control inputs": the SRM *is* the convolution form of the premise's $x_{t+1} = A x_t + B u_t$ with $u_t$ containing the neuron's own spike train. In discrete time, the **soft reset** (subtract $V_{th}$ at spike) is precisely $V_{t+1} = \alpha V_t + \dots - V_{th}\, s_t$ — linear in $(V_t, u_t)$ with a *known* coefficient. The **hard reset** (set to $V_{reset}$) is $V_{t+1} = (1-s_t)(\alpha V_t + \dots) + s_t V_{reset}$, which is bilinear in $(s_t, V_t)$ — *not* representable exactly as $A x_t + B u_t$ with state-independent $B$. Design consequence: **prefer soft reset** in the library's canonical model, since the premise's equation is then exact rather than approximate; note this is also what most SG-training frameworks default to for gradient quality.

### 2.3 The field already exploits this linearity (the strongest challenge to novelty)

A substantial 2023–2026 literature builds SNN training directly on the linear-state-space form of LIF, without invoking Koopman/DMD:

- **PSN — Parallel Spiking Neuron** (Fang et al., NeurIPS 2023; arXiv:2304.12760): removes the reset so the LIF recursion becomes a pure linear filter, computable in parallel across time.
- **SpikingSSMs** (AAAI; proceedings PDF located): merges S4-style state space models with spiking dynamics.
- **P-SpikeSSM** (arXiv:2406.02923), **SPikE-SSM** (arXiv:2410.17268), **SiLIF** (arXiv:2506.06374): spiking SSMs with parallel training; SiLIF explicitly reparametrizes an adaptive-LIF state transition matrix following S4. (Stan & Rhodes 2023, "linearizing LIF neurons," is reported within these papers; I did not locate the primary source and cite it only secondhand.)

These works constitute the "already-linear" baseline the project must beat or differentiate from. They also demonstrate the payoff of the linear form: parallel-in-time training, which the Koopman-SNN architecture should also inherit and which the implementation plan should exploit.

### 2.4 Genuinely nonlinear neuron models (where lifting has something to do)

| Model | State | Nonlinearity | Citation (verified) |
|---|---|---|---|
| LIF (current synapses) | $V, I$ | none sub-threshold | Gerstner & Kistler 2002 |
| LIF (conductance synapses) | $V, g_e, g_i$ | bilinear $g\,(V - E_{syn})$ | Gerstner & Kistler 2002 |
| Adaptive LIF / ALIF | $V, a$ | linear + spike-triggered adaptation | Bellec et al. 2020 (uses ALIF) |
| AdEx | $V, w$ | exponential $\Delta_T e^{(V-V_T)/\Delta_T}$ | Brette & Gerstner 2005 |
| Izhikevich | $V, u$ | quadratic $0.04V^2$ | Izhikevich 2003 |
| Hodgkin–Huxley | $V, m, h, n$ | cubic/quartic gating products | Gerstner & Kistler 2002, Ch. 2 (original: Hodgkin & Huxley 1952) |

For these, a finite-dimensional linear surrogate is *not* available analytically, and EDMD lifting is a legitimate, literature-supported tool. Koopman theory has an established track record on neuron-type dynamics: Mauroy, Mezić & Moehlis (2013) define **isostables** as level sets of the slowest Koopman eigenfunction and apply the framework to the FitzHugh–Nagumo excitable neuron model; subsequent phase-amplitude reduction work builds on this. Adaptive LIF is a particularly attractive first nonlinear target: it is the model of record in e-prop (Bellec et al. 2020), it materially improves task performance over plain LIF, and its spike-triggered adaptation makes the *effective closed-loop* dynamics (sub-threshold + firing statistics) nonlinear even though each piece is simple.

---

## 3. Where Koopman-DMD Genuinely Adds Value — and Where It Does Not

### 3.1 Does NOT add value: linearizing plain current-based LIF

Covered in §2.1–2.3. For plain LIF with soft reset and current-based synapses, $A$ and $B$ are known exactly in closed form, and DMD can only recover them noisily. The forward-pass pseudocode in the premise (dense $N \times N$ `koopman_A` times potentials) is, for plain LIF, a dense re-implementation of what is exactly a *diagonal* update — strictly more expensive ($O(N^2)$ vs $O(N)$) with no accuracy gain. **The library should treat "DMD recovers the analytic LIF propagator to machine precision" as a validation test, not a feature.**

### 3.2 ADDS value: reduced-order modeling of recurrent layers (the strongest case)

For a *recurrent* layer of $N$ neurons with recurrent weights $W_{rec}$, the sub-threshold-plus-average-spiking dynamics couple all neurons and a step costs $O(N^2)$. Rank-$r$ DMD/DMDc yields a reduced state $z_t = U_r^{*} x_t \in \mathbb{R}^r$ with

$$z_{t+1} = \tilde{A} z_t + \tilde{B} u_t, \qquad x_t \approx U_r z_t,$$

costing $O(Nr + r^2)$ per step — a genuine compression when the layer's activity lives on a low-dimensional manifold, which large trained/biological networks empirically do. Precedents: DMD extracting low-rank spatiotemporal structure from large-scale neural recordings (Brunton, Johnson, Ojemann & Kutz 2016 — note: applied to *continuous* ECoG voltages with delay embedding, not to binary spike trains; the library should likewise apply DMD to membrane potentials/currents/filtered traces, never raw binary spikes); survey of model order reduction in neuroscience (arXiv:2003.05133). This is the clearest performance story for the Rust library: **surrogate inference models and fast approximate rollouts of large recurrent SNN layers.**

### 3.3 ADDS value: lifted linear surrogates for nonlinear neuron models

EDMD/EDMDc with a chosen dictionary gives a principled linear surrogate for AdEx/Izhikevich/conductance-synapse/HH layers (§2.4), enabling (i) the same fast linear stepping and parallel-in-time training as LIF enjoys natively, and (ii) gradient computation through the surrogate where the true model's stiffness (e.g., AdEx's exponential blow-up at spike onset) makes BPTT numerically nasty. This is where the premise's architecture is actually novel rather than redundant. Caveat from theory (§1.1): the lifting is local/approximate — validity regions must be quantified empirically (prediction horizon vs. error), and the threshold/reset must stay outside the lifted linear model.

### 3.4 ADDS value: spectral analysis of trained networks

DMD eigenvalues of a trained layer's dynamics give, essentially for free: intrinsic timescales, oscillatory modes (argument of $\mu_j$), stability margins ($1 - |\mu_j|$), and — per §1.6 — per-mode gradient propagation depth. Precedents for operator-theoretic analysis of trained networks: Naiman & Azencot's operator-theoretic analysis of sequence models (arXiv:2102.07824, AAAI); Dogra & Redman (NeurIPS 2020) applying Koopman theory to *weight* training dynamics; Redman et al. (ICLR 2022, arXiv:2110.14856) Koopman-based pruning. A `spectrum()` API on trained layers is cheap to build atop the koopman-dmd crate and scientifically well-grounded.

### 3.5 Systemic weaknesses the project must confront (candid assessment)

1. **Closed-loop identifiability.** DMDc assumes $u_t$ is exogenous. Here $u_t$ includes the layer's *own* spikes, $u_t = \Theta(C x_t)$ — deterministic state feedback. Regressing $[A\;B]$ jointly on such data is ill-posed: the regressors $x_t$ and $u_t$ are strongly collinear (Proctor et al. 2016 flag exactly this closed-loop caution). **Mitigation (and recommended design):** the reset contribution is known analytically (soft reset: coefficient $-V_{th}$), and feed-forward $B$ is the known $W$. Use known-B DMDc and fit only what is genuinely unknown. Fit $A$ freely only for the nonlinear-model surrogates of §3.3, ideally with input-excitation (frozen-noise probe currents) during data collection.
2. **Chicken-and-egg with training.** A DMD-fit $A$ models the network *at fixed weights*. During training the weights change; the fitted $A$ for recurrent layers (where $A$ absorbs $W_{rec}$) goes stale. Options: (a) restrict DMD-$A$ to weight-independent neuron dynamics (then it never goes stale — but for LIF that is the trivial diagonal $A$); (b) periodic re-fitting (an EM-like alternation: train weights, re-fit $A$); (c) online/streaming DMD updates. The architecture document must choose; the literature contains no ready-made answer for the SNN case.
3. **DMD on spikes is ill-posed.** Binary spike trains are not smooth observables. All successful neural-data DMD work uses continuous signals (Brunton et al. 2016). The observables must be membrane potentials, synaptic currents, or low-pass-filtered spike traces.
4. **The gradient claim** must be restated as spectral diagnosis/regularization, not prevention (§1.6).
5. **Hard reset is not linear-in-control** (§2.2); use soft reset or accept a bilinear correction.

---

## 4. Prior and Related Work (verified citations)

### Koopman / DMD foundations
- Mezić, I. (2005). Spectral properties of dynamical systems, model reduction and decompositions. *Nonlinear Dynamics* 41(1–3), 309–325. https://link.springer.com/article/10.1007/s11071-005-2824-x
- Williams, M. O., Kevrekidis, I. G., & Rowley, C. W. (2015). A data-driven approximation of the Koopman operator: extending dynamic mode decomposition. *Journal of Nonlinear Science* 25(6), 1307–1346. arXiv: https://arxiv.org/abs/1408.4408
- Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.* SIAM. ISBN 978-1-61197-449-2. http://dmdbook.com
- Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016). Dynamic mode decomposition with control. *SIAM J. Applied Dynamical Systems* 15(1), 142–161. arXiv: https://arxiv.org/abs/1409.6358
- Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2018). Generalizing Koopman theory to allow for inputs and control. *SIAM J. Applied Dynamical Systems* 17(1), 909–930. arXiv: https://arxiv.org/abs/1602.07647
- Korda, M., & Mezić, I. (2018a). On convergence of extended dynamic mode decomposition to the Koopman operator. *Journal of Nonlinear Science* 28, 687–710. https://link.springer.com/article/10.1007/s00332-017-9423-0
- Korda, M., & Mezić, I. (2018b). Linear predictors for nonlinear dynamical systems: Koopman operator meets model predictive control. *Automatica* 93, 149–160. arXiv: https://arxiv.org/abs/1611.03537
- Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). Deep learning for universal linear embeddings of nonlinear dynamics. *Nature Communications* 9, 4950. https://doi.org/10.1038/s41467-018-07210-0 · arXiv: https://arxiv.org/abs/1712.09707
- Brunton, S. L., Budišić, M., Kaiser, E., & Kutz, J. N. (2022). Modern Koopman theory for dynamical systems. *SIAM Review* 64(2), 229–340. https://doi.org/10.1137/21M1401243

### Surrogate gradients and SNN training
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine* 36(6), 51–63. https://ieeexplore.ieee.org/document/8891809 · arXiv: https://arxiv.org/abs/1901.09948
- Zenke, F., & Ganguli, S. (2018). SuperSpike: Supervised learning in multilayer spiking neural networks. *Neural Computation* 30(6), 1514–1541. https://direct.mit.edu/neco/article/30/6/1514/8378 · arXiv: https://arxiv.org/abs/1705.11146
- Shrestha, S. B., & Orchard, G. (2018). SLAYER: Spike layer error reassignment in time. *NeurIPS 31.* https://proceedings.neurips.cc/paper_files/paper/2018/hash/82f2b308c3b01637c607ce05f52a2fed-Abstract.html
- Bellec, G., Scherr, F., Subramoney, A., Hajek, E., Salaj, D., Legenstein, R., & Maass, W. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop). *Nature Communications* 11, 3625. https://www.nature.com/articles/s41467-020-17236-y
- Wunderlich, T. C., & Pehle, C. (2021). Event-based backpropagation can compute exact gradients for spiking neural networks (EventProp). *Scientific Reports* 11, 12829. https://www.nature.com/articles/s41598-021-91786-z · arXiv: https://arxiv.org/abs/2009.08378
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML 2013.* arXiv: https://arxiv.org/abs/1211.5063

### Neuron models
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models: Single Neurons, Populations, Plasticity.* Cambridge University Press. https://dl.acm.org/citation.cfm?id=583784 · SRM summary: http://www.scholarpedia.org/article/Spike-response_model
- Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Trans. Neural Networks* 14(6), 1569–1572. https://www.izhikevich.org/publications/spikes.pdf
- Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. *J. Neurophysiology* 94, 3637–3642. https://cenl.ucsd.edu/CompNeuro/Readings/week13/Brette-Gerstner+Adaptive-exponential-integrate-fire-effective+JNP+2005.pdf · http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model
- (Hodgkin & Huxley 1952 is cited via Gerstner & Kistler 2002, Ch. 2; primary source not separately retrieved in this survey.)

### DMD/Koopman applied to neural systems and neural networks
- Brunton, B. W., Johnson, L. A., Ojemann, J. G., & Kutz, J. N. (2016). Extracting spatial–temporal coherent patterns in large-scale neural recordings using dynamic mode decomposition. *J. Neuroscience Methods* 258, 1–15. https://www.sciencedirect.com/science/article/abs/pii/S0165027015003829 · arXiv: https://arxiv.org/abs/1409.5496
- Mauroy, A., Mezić, I., & Moehlis, J. (2013). Isostables, isochrons, and Koopman spectrum for the action–angle representation of stable fixed point dynamics. *Physica D* 261, 19–30. https://www.sciencedirect.com/science/article/abs/pii/S0167278913001620
- Dogra, A. S., & Redman, W. T. (2020). Optimizing neural networks via Koopman operator theory. *NeurIPS 33.* https://arxiv.org/abs/2006.02361
- Redman, W. T., et al. (2022). An operator theoretic view on pruning deep neural networks. *ICLR 2022.* https://arxiv.org/abs/2110.14856 · https://openreview.net/forum?id=pWBNOgdeURp
- Naiman, I., & Azencot, O. An operator theoretic approach for analyzing sequence neural networks. *AAAI.* arXiv: https://arxiv.org/abs/2102.07824
- Extraction of nonlinearity in neural networks with Koopman operator. arXiv: https://arxiv.org/abs/2402.11740 (located in search; abstract-level relevance only)
- Model order reduction in neuroscience (survey). arXiv: https://arxiv.org/abs/2003.05133

### Linear-state-space SNNs (the no-Koopman baseline; 2023–2026)
- Fang, W., et al. (2023). Parallel spiking neurons with high efficiency and ability to learn long-term dependencies (PSN). *NeurIPS 2023.* https://arxiv.org/abs/2304.12760
- SpikingSSMs: Learning long sequences with sparse and parallel spiking state space models. *AAAI.* https://ojs.aaai.org/index.php/AAAI/article/download/34245/36400
- P-SpikeSSM: Harnessing probabilistic spiking state space models for long-range dependency tasks. https://arxiv.org/abs/2406.02923
- SPikE-SSM: A sparse, precise, and efficient spiking state space model for long sequences learning. https://arxiv.org/abs/2410.17268
- SiLIF: Structured state space model dynamics and parametrization for spiking neural networks. https://arxiv.org/abs/2506.06374

### Prior art specifically combining Koopman/DMD with SNNs
Multiple targeted searches (August 2026: "Koopman spiking neural network," "DMD spiking neural network," "Koopman linearize LIF surrogate gradient," arXiv-restricted queries) found **no published work that uses Koopman/DMD to model SNN layer dynamics as a linear-plus-control system for inference or surrogate-gradient training.** The closest items are: Koopman analyses of *trained-network weight dynamics* (Dogra & Redman 2020; Redman et al. 2022), operator-theoretic analysis of RNN hidden dynamics (Naiman & Azencot), DMD on neural recordings (Brunton et al. 2016), one paper mentioning spiking neurons only as motivation (Sun, Chen & Baillieul, arXiv:2405.00627 — verified by direct fetch that it contains no SNN-specific content), and a "spiking mode-based neural networks" paper using a different (connectivity-mode) notion of decomposition (arXiv:2310.14621). **The specific combination proposed by this project appears to be open territory** — with the caveat that the spiking-SSM literature (above) already occupies the neighboring "linear state + threshold nonlinearity" design point without Koopman machinery, so novelty claims must rest on §3.2–3.4, not on linearizing LIF.

---

## 5. Open Scientific Questions for This Project

1. **Value-over-baseline question (the central one).** For plain LIF, the analytic propagator is exact and diagonal. What measurable advantage does a DMD-fit $A$ deliver over (i) the analytic LIF propagator and (ii) spiking-SSM parallel training? Proposed answer to test: advantage exists only for (a) rank-$r$ compression of recurrent layers, (b) nonlinear neuron models via EDMD lifting, (c) spectral diagnostics. Design experiments that can falsify each.
2. **Dictionary design for spiking models.** Which observables let EDMD capture AdEx/Izhikevich sub-threshold dynamics to a useful horizon? Candidates: monomials in $(V, w)$, $e^{(V - V_T)/\Delta_T}$ itself (for AdEx this makes the dynamics nearly bilinear), delay embeddings. No literature answers this for spiking models specifically.
3. **Closed-loop fitting protocol.** How to collect snapshot data so that $[A, B]$ identification is well-posed given that spikes are state feedback (§3.5.1)? Known-B fitting, probe-input excitation, and regularized regression need empirical comparison.
4. **Staleness under training.** How fast does a fitted $\tilde{A}$ for a recurrent layer degrade as $W_{rec}$ is updated by SG descent, and what re-fit schedule (or online DMD) keeps surrogate-gradient bias acceptable? No prior art.
5. **Gradient quality through the reduced model.** Does backpropagating through $z_{t+1} = \tilde{A} z_t + \tilde{B} u_t$ (rank $r < N$) yield weight updates that converge comparably to full BPTT-with-SG? Relation to e-prop (which also truncates credit assignment) is worth a formal comparison.
6. **Spectral regularization.** Can constraining DMD eigenvalues (e.g., $|\mu_j| \in [1-\epsilon, 1]$ for selected modes) measurably extend usable credit-assignment horizons in SNN training, analogous to unitary-RNN results? This would convert the premise's (currently unsupported) gradient claim into a testable, publishable hypothesis.
7. **Error bounds across thresholds.** The linear surrogate is valid sub-threshold; each spike/reset re-injects state into a region where the fit may be poor. How does surrogate rollout error grow with firing rate? (Prediction: accuracy degrades at high rates; quantify.)
8. **Continuous vs. binary observables.** Confirm empirically that DMD on filtered spike traces/membrane variables works where DMD on raw binary spikes fails, consistent with Brunton et al. 2016's use of continuous ECoG.
