"""snnTorch calibration baseline for the kdmd-SNN SHD campaign (improvements.md P2.3).

Replicates the R recipe as closely as snnTorch's model zoo allows:
  350 pooled channels (2:1 from 700), 100 x 10 ms bins, one recurrent layer of
  256 synaptic-conductance LIF neurons (RSynaptic: two state variables, the
  same (v, i) pair as kdmd-SNN's exact LIF), subtractive reset, fast-sigmoid
  surrogate (slope 5), spike-count readout logits = R.(sum_t s_t)/T, softmax
  cross-entropy, Adam 5e-3, elementwise grad clip 1.0, batch 32, 6000
  minibatches, the same three-way augmentation (15% event dropout, +/-25
  channel shift, 0.9-1.1 time stretch), full-test-set evaluation (2,240).

Deliberate differences from the Rust harness (disclosed):
  - Decay factors: snnTorch's Synaptic uses the Euler-flavored update
    mem <- beta*mem + syn rather than the exact ZOH propagator's gamma/delta
    coupling; alpha/beta here are set to the exact exponentials, so the
    sub-threshold dynamics are close but not bit-identical.
  - Weight init: input weights U(0, 35/350) as in the harness; recurrent
    weights zero (the campaign's identity discipline); readout is torch's
    default Linear init rather than the harness's deterministic pattern.
  - RNG streams differ entirely; only the architecture/protocol match.

Purpose: an order-of-magnitude external calibration ("does an independent
framework land in the same range as the recorded R-recipe mean 0.850 +/-
0.027?"), not a bit-exact replication. Expectation stated before running:
0.80-0.88.

Usage: python baselines/snntorch_shd.py [seed]
"""

import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
# Optional input-gain scale (post-hoc diagnostic, see results note):
# RSynaptic couples syn into mem with gain 1.0/step where the exact
# propagator's gamma = 0.239, so the transplanted init over-drives
# snnTorch ~4x. GAIN=0.25 roughly compensates.
GAIN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
N_POOLED = 350
T_STEPS = 100
BIN_S = 0.010
N_HIDDEN = 256
N_CLASSES = 20
BATCH = 32
MINIBATCHES = 6000
LR = 5e-3
TAU_M, TAU_S = 20.0, 10.0  # ms
DT_MS = BIN_S * 1e3

AUG_EVENT_DROP = 0.15
AUG_CHANNEL_SHIFT = 25
AUG_STRETCH = (0.9, 1.1)

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"seed {SEED}, device {device}, snntorch {snn.__version__}, torch {torch.__version__}")


def load_shd(path):
    with h5py.File(path, "r") as f:
        times = [np.asarray(t) for t in f["spikes"]["times"]]
        units = [np.asarray(u) for u in f["spikes"]["units"]]
        labels = np.asarray(f["labels"])
    return list(zip(times, units, labels))


def bin_sample(times, units, out):
    """Pool 2:1 and bin into out[T, C] (binary)."""
    t_idx = (times / BIN_S).astype(np.int64)
    keep = t_idx < T_STEPS
    c_idx = units[keep] // 2
    out[t_idx[keep], c_idx] = 1.0


def augment(times, units):
    keep = rng.random(times.shape[0]) >= AUG_EVENT_DROP
    times, units = times[keep], units[keep]
    shift = rng.integers(-AUG_CHANNEL_SHIFT, AUG_CHANNEL_SHIFT + 1)
    units = units.astype(np.int64) + shift
    ok = (units >= 0) & (units < 700)
    stretch = rng.uniform(*AUG_STRETCH)
    return times[ok] * stretch, units[ok].astype(np.uint32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        alpha = float(np.exp(-DT_MS / TAU_S))  # synaptic-current decay
        beta = float(np.exp(-DT_MS / TAU_M))   # membrane decay
        grad = surrogate.fast_sigmoid(slope=5)
        self.fc_in = nn.Linear(N_POOLED, N_HIDDEN, bias=False)
        # Harness init: U(0, 35/fan_in) for the input layer.
        with torch.no_grad():
            self.fc_in.weight.uniform_(0.0, GAIN * 35.0 / N_POOLED)
        self.lif = snn.RSynaptic(
            alpha=alpha,
            beta=beta,
            spike_grad=grad,
            reset_mechanism="subtract",
            threshold=1.0,
            all_to_all=True,
            linear_features=N_HIDDEN,
        )
        # Identity discipline: recurrence grows from zero.
        with torch.no_grad():
            self.lif.recurrent.weight.zero_()
            if self.lif.recurrent.bias is not None:
                self.lif.recurrent.bias.zero_()
        self.readout = nn.Linear(N_HIDDEN, N_CLASSES, bias=False)

    def forward(self, x):  # x: [T, B, C]
        spk, syn, mem = self.lif.init_rsynaptic()
        spk = torch.zeros(x.shape[1], N_HIDDEN, device=x.device)
        syn = torch.zeros_like(spk)
        mem = torch.zeros_like(spk)
        counts = torch.zeros_like(spk)
        for t in range(x.shape[0]):
            cur = self.fc_in(x[t])
            spk, syn, mem = self.lif(cur, spk, syn, mem)
            counts = counts + spk
        return self.readout(counts / x.shape[0])


def main():
    train = load_shd("data/shd/shd_train.h5")
    test = load_shd("data/shd/shd_test.h5")
    print(f"{len(train)} train / {len(test)} test samples")

    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
    recent = []
    for step in range(MINIBATCHES):
        xs = np.zeros((T_STEPS, BATCH, N_POOLED), dtype=np.float32)
        ys = np.zeros(BATCH, dtype=np.int64)
        for b in range(BATCH):
            times, units, label = train[rng.integers(0, len(train))]
            a_times, a_units = augment(times, units)
            bin_sample(a_times, a_units, xs[:, b, :])
            ys[b] = label
        x = torch.from_numpy(xs).to(device)
        y = torch.from_numpy(ys).to(device)
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
        opt.step()
        recent.append(loss.item())
        if len(recent) == 50:
            print(f"  step {step + 1:5}: mean loss {np.mean(recent):.4f}", flush=True)
            recent = []
    train_secs = time.time() - start

    net.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(test) - BATCH + 1, BATCH):
            xs = np.zeros((T_STEPS, BATCH, N_POOLED), dtype=np.float32)
            ys = np.zeros(BATCH, dtype=np.int64)
            for b, (times, units, label) in enumerate(test[i : i + BATCH]):
                bin_sample(times, units, xs[:, b, :])
                ys[b] = label
            pred = net(torch.from_numpy(xs).to(device)).argmax(dim=1).cpu().numpy()
            correct += int((pred == ys).sum())
            total += BATCH
    print(
        f"RESULT [snntorch-R seed {SEED} gain {GAIN}]: test accuracy {correct / total:.4f} "
        f"({correct}/{total}), train {train_secs:.1f}s"
    )


if __name__ == "__main__":
    main()
