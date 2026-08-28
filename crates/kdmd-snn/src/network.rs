//! Feedforward network of [`KoopmanLayer`]s.
//!
//! Owns the per-layer states and the inter-layer spike buffers so a network
//! step is allocation-free. Layers are stepped sequentially within one time
//! step (layer ℓ's spikes drive layer ℓ+1 in the same step). The readout
//! head arrives with training (Phase 6); until then the last layer's spikes
//! are the network output.

use crate::error::SnnError;
use crate::layer::KoopmanLayer;
use crate::spikes::{SpikeBatch, SpikeVec};
use crate::state::LayerState;

/// A feedforward stack of layers with owned state.
#[derive(Debug, Clone)]
pub struct Network {
    layers: Vec<KoopmanLayer>,
    states: Vec<LayerState>,
    /// Inter-layer sparse buffers: `spike_bufs[l]` holds layer l's output.
    spike_bufs: Vec<SpikeVec>,
    /// Inter-layer dense buffers for the batched path.
    batch_bufs: Vec<SpikeBatch>,
    batch: usize,
}

impl Network {
    /// Assemble a network. Layer ℓ+1's input count must equal layer ℓ's
    /// neuron count; every layer must share the same batch capacity.
    pub fn new(layers: Vec<KoopmanLayer>, batch: usize) -> Result<Self, SnnError> {
        if layers.is_empty() {
            return Err(SnnError::InvalidParameter(
                "network needs at least one layer".into(),
            ));
        }
        for w in layers.windows(2) {
            if w[1].n_inputs() != w[0].n_neurons() {
                return Err(SnnError::DimensionMismatch(format!(
                    "layer input count {} does not match previous layer's {} neurons",
                    w[1].n_inputs(),
                    w[0].n_neurons()
                )));
            }
        }
        let mut states = Vec::with_capacity(layers.len());
        let mut spike_bufs = Vec::with_capacity(layers.len());
        let mut batch_bufs = Vec::with_capacity(layers.len());
        for layer in &layers {
            states.push(LayerState::zeros(
                layer.n_neurons(),
                layer.n_state_vars(),
                batch,
            )?);
            spike_bufs.push(SpikeVec::new(layer.n_neurons()));
            batch_bufs.push(SpikeBatch::zeros(layer.n_neurons(), batch)?);
        }
        Ok(Self {
            layers,
            states,
            spike_bufs,
            batch_bufs,
            batch,
        })
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, l: usize) -> &KoopmanLayer {
        &self.layers[l]
    }

    /// Mutable layer access (the optimizer updates weights through this).
    pub fn layer_mut(&mut self, l: usize) -> &mut KoopmanLayer {
        &mut self.layers[l]
    }

    pub fn state(&self, l: usize) -> &LayerState {
        &self.states[l]
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Clone the network with a different batch capacity: same layers,
    /// weights, and dynamics; fresh zeroed state. The trainer's data-parallel
    /// path uses this to give each thread its own chunk-sized copy.
    pub fn clone_with_batch(&self, batch: usize) -> Result<Self, SnnError> {
        let layers: Vec<KoopmanLayer> = self
            .layers
            .iter()
            .map(|l| l.clone_with_batch(batch))
            .collect::<Result<_, _>>()?;
        Self::new(layers, batch)
    }

    /// One batched, taped network step: per layer, the pre-reset potentials
    /// land in `v_pre[l]` and the emitted spikes in `s_out[l]` (each
    /// `n_l × batch`). The training tape owns those buffers.
    pub fn step_batch_taped(
        &mut self,
        input: &SpikeBatch,
        v_pre: &mut [faer::Mat<f64>],
        s_out: &mut [SpikeBatch],
    ) -> Result<(), SnnError> {
        self.step_batch_taped_impl(input, v_pre, s_out, None)
    }

    /// [`step_batch_taped`](Self::step_batch_taped) with the extra tape the
    /// learnable-τ gradient needs: `x_pre[l]` receives the layer's **pre-step**
    /// state (`k_l·n_l × batch`) and `drive[l]` the drive `W·s_in (+ rec)` the
    /// step computed (`n_l × batch`).
    pub fn step_batch_taped_tau(
        &mut self,
        input: &SpikeBatch,
        v_pre: &mut [faer::Mat<f64>],
        s_out: &mut [SpikeBatch],
        x_pre: &mut [faer::Mat<f64>],
        drive: &mut [faer::Mat<f64>],
    ) -> Result<(), SnnError> {
        if x_pre.len() != self.layers.len() || drive.len() != self.layers.len() {
            return Err(SnnError::DimensionMismatch(format!(
                "τ-tape buffers cover {} / {} layers, network has {}",
                x_pre.len(),
                drive.len(),
                self.layers.len()
            )));
        }
        self.step_batch_taped_impl(input, v_pre, s_out, Some((x_pre, drive)))
    }

    #[allow(clippy::type_complexity)]
    fn step_batch_taped_impl(
        &mut self,
        input: &SpikeBatch,
        v_pre: &mut [faer::Mat<f64>],
        s_out: &mut [SpikeBatch],
        mut tau_tape: Option<(&mut [faer::Mat<f64>], &mut [faer::Mat<f64>])>,
    ) -> Result<(), SnnError> {
        if v_pre.len() != self.layers.len() || s_out.len() != self.layers.len() {
            return Err(SnnError::DimensionMismatch(format!(
                "tape buffers cover {} / {} layers, network has {}",
                v_pre.len(),
                s_out.len(),
                self.layers.len()
            )));
        }
        for l in 0..self.layers.len() {
            // (τ tape) the state BEFORE this layer advances.
            if let Some((x_pre, _)) = tau_tape.as_mut() {
                let src = self.states[l].as_mat();
                let dst = &mut x_pre[l];
                for b in 0..src.ncols() {
                    for i in 0..src.nrows() {
                        dst[(i, b)] = src[(i, b)];
                    }
                }
            }
            let (prev, rest) = self.batch_bufs.split_at_mut(l);
            let s_in = if l == 0 { input } else { &prev[l - 1] };
            self.layers[l].step_batch_taped(
                &mut self.states[l],
                s_in,
                &mut rest[0],
                &mut v_pre[l],
            )?;
            // (τ tape) the drive this step computed.
            if let Some((_, drive)) = tau_tape.as_mut() {
                let src = self.layers[l].drive();
                let dst = &mut drive[l];
                for b in 0..src.ncols() {
                    for i in 0..src.nrows() {
                        dst[(i, b)] = src[(i, b)];
                    }
                }
            }
            // Copy the emitted spikes into the caller's tape.
            let src = self.batch_bufs[l].as_mat();
            let mut dst = s_out[l].as_mat_mut();
            for b in 0..src.ncols() {
                for i in 0..src.nrows() {
                    dst[(i, b)] = src[(i, b)];
                }
            }
        }
        Ok(())
    }

    /// Zero every layer state and all episodic recurrent state (start of a
    /// new trial).
    pub fn reset_state(&mut self) {
        for s in &mut self.states {
            s.fill(0.0);
        }
        for l in &mut self.layers {
            l.reset_recurrent();
        }
    }

    /// One network time step, batch = 1, sparse spikes. Returns the last
    /// layer's spikes for this step.
    pub fn step(&mut self, input: &SpikeVec) -> Result<&SpikeVec, SnnError> {
        if self.batch != 1 {
            return Err(SnnError::InvalidParameter(format!(
                "sparse step requires batch = 1 (network batch is {})",
                self.batch
            )));
        }
        for l in 0..self.layers.len() {
            // Split borrows: the input buffer is either the caller's or the
            // previous layer's output, which sits before index l.
            let (prev, rest) = self.spike_bufs.split_at_mut(l);
            let s_in = if l == 0 { input } else { &prev[l - 1] };
            self.layers[l].step(&mut self.states[l], s_in, &mut rest[0])?;
        }
        Ok(self.spike_bufs.last().unwrap())
    }

    /// One batched network time step. Returns the last layer's spikes.
    pub fn step_batch(&mut self, input: &SpikeBatch) -> Result<&SpikeBatch, SnnError> {
        for l in 0..self.layers.len() {
            let (prev, rest) = self.batch_bufs.split_at_mut(l);
            let s_in = if l == 0 { input } else { &prev[l - 1] };
            self.layers[l].step_batch(&mut self.states[l], s_in, &mut rest[0])?;
        }
        Ok(self.batch_bufs.last().unwrap())
    }

    /// Run a sparse input sequence, returning per-step output spike counts of
    /// the last layer (a cheap summary; callers wanting full trains clone the
    /// step outputs).
    pub fn run_counts(&mut self, inputs: &[SpikeVec]) -> Result<Vec<usize>, SnnError> {
        let mut counts = Vec::with_capacity(inputs.len());
        for s_in in inputs {
            counts.push(self.step(s_in)?.count());
        }
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::{Lif, LifParams};
    use faer::Mat;

    fn dense_w(rows: usize, cols: usize, gain: f64) -> Mat<f64> {
        Mat::from_fn(rows, cols, |_, _| gain)
    }

    fn two_layer_net(batch: usize) -> Network {
        let lif = Lif::new(LifParams {
            dt: 0.5,
            ..LifParams::default()
        })
        .unwrap();
        let l0 = KoopmanLayer::lif(&lif, 4, dense_w(4, 2, 1.5), batch).unwrap();
        // Layer-1 gain must be high enough that inter-volley synaptic decay
        // still leaves the membrane supra-threshold.
        let l1 = KoopmanLayer::lif(&lif, 3, dense_w(3, 4, 3.0), batch).unwrap();
        Network::new(vec![l0, l1], batch).unwrap()
    }

    #[test]
    fn dimension_mismatch_between_layers_is_rejected() {
        let lif = Lif::new(LifParams::default()).unwrap();
        let l0 = KoopmanLayer::lif(&lif, 4, dense_w(4, 2, 1.0), 1).unwrap();
        let l1 = KoopmanLayer::lif(&lif, 3, dense_w(3, 5, 1.0), 1).unwrap();
        assert!(Network::new(vec![l0, l1], 1).is_err());
    }

    #[test]
    fn spikes_propagate_through_the_stack() {
        let mut net = two_layer_net(1);
        let drive = SpikeVec::from_indices(vec![0, 1], 2).unwrap();
        let mut saw_output = false;
        for _ in 0..300 {
            if !net.step(&drive).unwrap().is_empty() {
                saw_output = true;
            }
        }
        assert!(saw_output, "constant drive never reached the output layer");
        // And the hidden layer actually charged up.
        assert!(net.state(0).var(1)[(0, 0)] > 0.0);
    }

    #[test]
    fn reset_state_silences_the_network() {
        let mut net = two_layer_net(1);
        let drive = SpikeVec::from_indices(vec![0, 1], 2).unwrap();
        for _ in 0..100 {
            net.step(&drive).unwrap();
        }
        net.reset_state();
        for l in 0..net.n_layers() {
            for i in 0..net.state(l).state_dim() {
                assert_eq!(net.state(l).as_mat()[(i, 0)], 0.0);
            }
        }
        // With no input, a reset network stays silent.
        let quiet = SpikeVec::new(2);
        for _ in 0..50 {
            assert!(net.step(&quiet).unwrap().is_empty());
        }
    }

    #[test]
    fn batched_and_sparse_networks_agree() {
        let mut sparse_net = two_layer_net(1);
        let mut batch_net = two_layer_net(1);
        let drive = SpikeVec::from_indices(vec![0], 2).unwrap();
        let mut dense_drive = SpikeBatch::zeros(2, 1).unwrap();
        dense_drive.as_mat_mut()[(0, 0)] = 1.0;

        for t in 0..300 {
            let a: Vec<u32> = sparse_net.step(&drive).unwrap().active().to_vec();
            let b = batch_net
                .step_batch(&dense_drive)
                .unwrap()
                .column_to_sparse(0);
            assert_eq!(a, b.active(), "paths diverged at step {t}");
        }
    }
}
