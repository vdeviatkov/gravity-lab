# Portable dense Q-policy format

`gravity-lab-dense-q-policy-v1` is the handoff between an external neural-network training
repository and Gravity Lab. It stores a sequential dense Q-network in a small text artifact that
both the dependency-free Python package and C++20 viewer evaluate identically. It is an inference
format, not a training checkpoint: optimizer state, replay buffers, target networks, schedules,
and experiment history remain in the training repository.

## Supported model

The input is normalized elementwise as `x[i] * input_scale[i] + input_bias[i]`. Each layer then
computes:

```text
y[o] = activation(bias[o] + sum_i weights[o, i] * x[i])
```

Supported activations are `relu`, `tanh`, and `linear`. Weights are output-major: all inputs for
output 0, then all inputs for output 1, and so on. This is the same logical shape as a PyTorch
`Linear.weight`, `[out_features, in_features]`. The final values are Q-values in environment action
order; greedy inference selects the first maximum, making ties deterministic.

The v1 grammar is:

```text
gravity-lab-dense-q-policy-v1
environment "gravity-lab-classic-v1"
input_size <N>
input_scale <N finite numbers>
input_bias <N finite numbers>
layers <L>
layer <inputs> <outputs> <relu|tanh|linear>
bias <outputs finite numbers>
weights <inputs * outputs finite numbers in output-major order>
... one layer/bias/weights group per layer ...
end
```

Readers reject unknown versions, disconnected dimensions, non-finite parameters, trailing data,
more than 32 layers, dimensions above 65,536, or more than 100 million weights. Writers replace a
temporary file atomically where the platform permits. V1 deliberately supports only sequential
dense models; CNNs, recurrent networks, dueling heads, and arbitrary computation graphs require a
new format or must be exported as an equivalent supported dense network.

## Export from Python

The package does not import NumPy, PyTorch, or JAX. `DenseLayer.from_values` accepts nested Python
sequences and array/tensor objects with `tolist`; objects with `detach` and `cpu` are converted
first. For a PyTorch model made from `Linear` layers:

```python
from gravity_lab import DenseLayer, DenseQPolicy

policy = DenseQPolicy(
    environment_id="gravity-lab-classic-v1",
    input_scale=[1.0] * 36,
    input_bias=[0.0] * 36,
    layers=[
        DenseLayer.from_values(model.fc1.weight, model.fc1.bias, "relu"),
        DenseLayer.from_values(model.fc2.weight, model.fc2.bias, "relu"),
        DenseLayer.from_values(model.q.weight, model.q.bias, "linear"),
    ],
)
policy.save("artifacts/classic_policy.gdp")
```

Export the online network chosen without using evaluation episodes to tune it. If preprocessing
uses `(x - mean) / std`, set `input_scale = 1 / std` and `input_bias = -mean / std`. Do not also
apply that normalization inside the exported layers.

Validate the artifact without opening a window:

```sh
./build-classic-rl/gravity_lab_classic_viewer \
  --policy artifacts/classic_policy.gdp --validate-only
```

Then render exploration-free episodes on the exact classic physics:

```sh
./build-classic-rl/gravity_lab_classic_viewer \
  --policy artifacts/classic_policy.gdp \
  --group 0 --track 0 --league 0 --frame-skip 2 \
  --max-steps 2000 --episodes 3 --seed 2000007
```

The policy must identify `gravity-lab-classic-v1`, accept 36 observations, and produce nine
Q-values. Escape or closing the window stops playback. Rendering is intentionally separate from
training and must not be included in training or evaluation timing.

## Reproducibility sidecar

Keep a versioned JSON sidecar beside each exported `.gdp` file. At minimum record the training
repository and commit, Gravity Lab commit and environment ID, policy SHA-256, map/pack SHA-256,
group, track, league, frame skip, episode limit, reward definition, architecture, normalization,
all named seeds, training budget, selected checkpoint, dependency versions, OS, compiler, and
CPU/GPU metadata. Framework-native training checkpoints remain the source for resuming training;
the `.gdp` artifact is the portable deployment copy.
