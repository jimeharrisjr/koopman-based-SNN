//! # kdmd-snn
//!
//! Spiking Neural Networks whose sub-threshold dynamics advance through a
//! linear-plus-control step
//!
//! ```text
//! x_{t+1} = A·x_t + B·u_t
//! ```
//!
//! with the spike threshold `Θ` as the single isolated nonlinearity. The state
//! operator `A` is identified with DMD/DMDc (via the [`koopman-dmd`] crate) for
//! neuron models where it is not known in closed form, while the input matrix
//! `B = [Eᵢ·W, −θ·Eᵥ]` is known by construction (synaptic weights plus the
//! subtractive-reset coefficient) and is never estimated.
//!
//! For plain LIF neurons the sub-threshold propagator is known analytically
//! (diagonal per neuron); this crate treats that case as its validation oracle
//! and performance baseline. The fitted-operator machinery earns its place on
//! **reduced-order modeling of recurrent layers** and **spectral diagnostics**
//! of identified operators. Two further candidates — per-step EDMD lifted
//! surrogates and spike-to-spike return-map surrogates of nonlinear neurons
//! (Izhikevich) — were tested under pre-registered protocols and **failed
//! their gates** (see the repository's docs/05 and docs/08); the [`surrogate`]
//! and [`return_map`] modules remain as the experimental record.
//!
//! Design documents live in the repository's `docs/` directory; the phased
//! delivery plan is `IMPLEMENTATION_PLAN.md`.
//!
//! [`koopman-dmd`]: https://crates.io/crates/koopman-dmd

pub mod data;
pub mod encoding;
pub mod error;
pub mod identify;
pub mod layer;
pub mod metrics;
pub mod network;
pub mod neuron;
pub mod operator;
pub mod return_map;
pub mod spikes;
pub mod state;
pub mod surrogate;
pub mod train;
pub(crate) mod util;

pub use error::SnnError;
pub use identify::{
    fit_autonomous, fit_controlled, lif_structural_b, IdentifiedControlled, IdentifiedOperator,
    IdentifyConfig, RankPolicy,
};
pub use layer::KoopmanLayer;
pub use network::Network;
pub use neuron::{
    AdLif, AdLifParams, Izhikevich, IzhikevichParams, Lif, LifParams, NeuronModel, ResetMode,
};
pub use operator::Operator;
pub use return_map::{DictionaryFamily, FitDiagnostics, IzhikevichReturnMap, ReturnMapConfig};
pub use spikes::{SpikeBatch, SpikeVec};
pub use state::LayerState;
pub use surrogate::{IzhikevichSurrogate, SurrogateConfig, SurrogateRollout};
pub use train::{OptimConfig, StepStats, SurrogateKind, TrainConfig, Trainer};
