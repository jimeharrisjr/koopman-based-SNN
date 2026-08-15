# Building a Spiking Neural Network with Rust and Koopman DMD

I have been developing a Rust-based Koopman package locally here: /Users/jimharris/Documents/rust-dmd and it is on crates.io here: https://crates.io/crates/koopman-dmd

I want to use what we've built to design a performant Spiking Neural Network (SNN) in Rust.

## Background

When transitioning from modeling a single neuron to a multi-layer Spiking Neural Network (SNN), the mathematical complexity scales up rapidly. By combining the Koopman operator with Dynamic Mode Decomposition (DMD), we can extract a finite-dimensional linear operator from the non-linear SNN dynamics, drastically simplifying both forward inference and backpropagation.

Here is the mathematical basis and architectural summary for how this is achieved across network layers.

## The Mathematical Basis: Koopman Meets SNNs

In a standard Leaky Integrate-and-Fire (LIF) network, the membrane potential $\mathbf{V}^{(l)}(t)$ of layer $l$ at time $t$ evolves according to continuous sub-threshold dynamics, punctuated by discrete, non-linear spike resets.

The Koopman operator, $\mathcal{K}$, allows us to represent the non-linear dynamics of this system as a linear system in an infinite-dimensional Hilbert space. For a state $\mathbf{x}_t$ (representing our membrane potentials and synaptic currents), we define observable functions $\mathbf{g}(\mathbf{x}_t)$. The operator advances the state such that:

$$\mathbf{g}(\mathbf{x}_{t+1}) = \mathcal{K} \mathbf{g}(\mathbf{x}_t)$$

Because we cannot compute an infinite-dimensional matrix, we use **Dynamic Mode Decomposition (DMD)** (specifically, extended DMD or EDMD) to find a finite-dimensional approximation, matrix $\mathbf{A}$.

If we record a time-series of network states into snapshot matrices $\mathbf{X}$ and $\mathbf{Y}$ (where $\mathbf{Y}$ is one time-step ahead of $\mathbf{X}$), DMD finds the best-fit linear operator $\mathbf{A}$ such that:

$$\mathbf{Y} \approx \mathbf{A} \mathbf{X}$$

Using Singular Value Decomposition (SVD), where $\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, the DMD matrix is approximated as:

$$\tilde{\mathbf{A}} = \mathbf{U}^T \mathbf{Y} \mathbf{V} \mathbf{\Sigma}^{-1}$$

### Incorporating Spikes (EDMD with Control)

Because the network receives external inputs and internal spikes from previous layers, the unforced DMD equation $\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t$ is insufficient. We treat the incoming spikes $\mathbf{s}^{(l-1)}(t)$ and the self-resetting mechanism as control inputs $\mathbf{u}_t$.

We solve for both the system dynamics matrix $\mathbf{A}$ and the control/input matrix $\mathbf{B}$:

$$\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t + \mathbf{B} \mathbf{u}_t$$

This transforms the SNN from a system of coupled differential equations into a purely linear algebraic stepping function, with the non-linearity explicitly isolated to the thresholding function $\Theta(v)$.

---

---

## Architecture: Inference and Training

### 1. Forward Pass (Inference)

During inference, the continuous membrane buildup is entirely replaced by the DMD matrix multiplication. At each discrete time step, the state vector is multiplied by $\mathbf{A}$, while the incoming spikes from the previous layer are multiplied by the synaptic weight matrix $\mathbf{W}$ (which maps to $\mathbf{B}$ in our control system).

If a potential crosses $V_{threshold}$, a spike is emitted to the next layer, and a reset penalty is applied via the control input.

### 2. Backward Pass (Training)

This is where the Koopman formulation shines. In standard SNNs, Backpropagation Through Time (BPTT) fails at the non-differentiable spike threshold.

By modeling the sub-threshold dynamics as $\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t + \mathbf{B} \mathbf{u}_t$, the gradient flows perfectly backward through the linear time steps. At the exact moment of the spike, we apply a **surrogate gradient** (e.g., a steep Sigmoid or Fast Sigmoid derivative). The Koopman matrix $\mathbf{A}$ provides a stable, mathematically rigorous Jacobin for the continuous portion of the sequence, preventing exploding or vanishing gradients over long time windows.

---

## Implementation Outline

To handle this efficiently, the data structures should cleanly separate the continuous linear state advancement from the discrete non-linear thresholding. Here is a pseudocode outline for a generalized SNN layer utilizing the Koopman-DMD approximation:

```text
// Define the state structures for a given layer
struct SNNLayer {
    weights: Matrix<f64>,        // Synaptic weights (acting as B)
    koopman_A: Matrix<f64>,      // Pre-computed DMD transition matrix (A)
    potentials: Vector<f64>,     // Current state vector (x_t)
    threshold: f64,
    reset_val: f64,
}

// Forward inference step over one time interval dt
fn forward_step(layer: &mut SNNLayer, input_spikes: Vector<f64>) -> Vector<f64> {
    
    // 1. Calculate input forcing (B * u_t)
    let synaptic_current = layer.weights.multiply(input_spikes);
    
    // 2. Advance the sub-threshold continuous dynamics linearly via Koopman matrix
    // x_{t+1} = A * x_t + input
    let next_potentials = layer.koopman_A.multiply(layer.potentials) + synaptic_current;
    
    // 3. Evaluate the discrete non-linear threshold (The Axon discharge)
    let mut output_spikes = Vector::zeros(layer.potentials.len());
    
    for i in 0..next_potentials.len() {
        if next_potentials[i] >= layer.threshold {
            output_spikes[i] = 1.0;
            next_potentials[i] = layer.reset_val; // Hard reset control
        } else {
            output_spikes[i] = 0.0;
        }
    }
    
    // 4. Update state and return emitted spikes for the next layer
    layer.potentials = next_potentials;
    return output_spikes;
}

// Backward pass surrogate gradient application (pseudocode)
fn surrogate_gradient(potential: f64, threshold: f64, steepness: f64) -> f64 {
    // Smooth approximation for the hard step function during BPTT
    let diff = potential - threshold;
    return 1.0 / (steepness * diff.abs() + 1.0).pow(2);
}

```

## Tasks

* Review the information here and search for any trustworthy, relevant literature that might help
* Create as many subagents as needed for this project. One should be the high-level scientist who understands the existing literature on Koopman and both biological and computer Neural Networks, and can document the physical and mathematical foundation. Another is a system architect that focuses on how to translate the theory into systems. The main Claude context will create the code using whatever functions are useful from the koopman-dmd crate. A code quality subagent will ensure the code is well-documented, contains sufficient tests, and uses best practices for Rust.  Test subagents should test and document issues for the architect to review and assign. A skeptic subagent should review the entire project skeptically to determine and document any systemic or reasoning weaknesses which must be addressed. A documentarian agent should ensure the development and findings are adequate, and that the repository contains sufficient documentation to be usable for others.
* Create an implementation plan for building a Koopman-based SNN Rust library

Ask me any questions you may have in the implementation plan markdown, and I can answer inline.


