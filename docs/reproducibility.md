# Reproducibility policy

An experiment configuration is incomplete unless it records:

- repository commit and environment-contract version;
- level pack hash (if a custom `.mrg` pack is used);
- `frame_skip`, maximum episode steps, and reward design/version (reward lives in the training
  repository, not here -- see docs/classic-rl.md);
- separate environment, parameter-initialization, exploration, replay-sampling, and evaluation seeds;
- algorithm hyperparameters, episode/transition budget, and checkpoint input;
- compiler, dependency versions, OS, CPU/GPU model, and accelerator backend;
- whether deterministic accelerator algorithms were enabled and any known nondeterministic operation.

The classic v1 environment consumes no environment randomness after reset. It executes the original
integer fixed-point physics and is tested for bitwise-repeatable trajectories in one binary. Seeds
remain explicit for API stability and future stochastic features.

Training and evaluation seed lists must not overlap. Evaluation disables exploration and reports
the distribution over all fixed episodes: mean and median reward, finish rate, crash rate, episode
length, and progress. Report execution time separately and exclude rendering. Never select only
successful recordings or the best checkpoint using evaluation seeds.

`gravity-lab-dense-q-policy-v1` stores sequential dense weights, activations, input normalization,
and environment ID in a portable Python/C++ format. It is not sufficient to resume training.
Loading rejects incompatible environment, observation, or action sizes. Neural training
repositories should save their own versioned metadata JSON plus framework-native weights and
optimizer/replay state atomically -- this repository does not define that format.

Every deployed `.gdp` policy needs a sidecar recording the training and environment repository
commits, policy and level-pack hashes, full environment and algorithm configuration, separately
named seeds, training budget, checkpoint-selection rule, dependency versions, and hardware. See
[policy-format.md](policy-format.md). Keep framework-native checkpoints as the source of truth and
verify the portable policy's Q-values against the training framework on fixed observations before
publishing it.

`gravity_lab_classic_headless` emits CSV with the versioned columns `environment`, `track`,
`episode`, `seed`, `steps`, `progress`, `finished`, `crashed`, `truncated`, and `wheelie` -- no
reward column, since this repository does not define one. Experiment tooling should add its full
configuration and metadata (including whatever reward it used) beside this episode table rather
than encoding configuration in a file name. Generated results and checkpoints belong under ignored
`artifacts/` unless a small reviewed fixture is intentionally added for a test.
