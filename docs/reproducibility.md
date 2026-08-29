# Reproducibility policy

An experiment configuration is incomplete unless it records:

- repository commit and environment-contract version;
- map path and SHA-256 hash;
- `time_step`, `frame_skip`, maximum episode steps, and reward version;
- separate environment, parameter-initialization, exploration, replay-sampling, and evaluation seeds;
- algorithm hyperparameters, episode/transition budget, and checkpoint input;
- compiler, dependency versions, OS, CPU/GPU model, and accelerator backend;
- whether deterministic accelerator algorithms were enabled and any known nondeterministic operation.

Environment v1 itself performs deterministic scalar CPU arithmetic and currently consumes no random
numbers after reset. The seed remains explicit so future stochastic maps can evolve without an API
break. Bitwise equality is expected for the same binary/platform; small floating-point differences
may occur across compilers and CPU architectures.

Training and evaluation seed lists must not overlap. Evaluation disables epsilon exploration and
reports the distribution over all fixed episodes: mean and median reward, finish rate, crash rate,
episode length, and progress. Report execution time separately and exclude rendering. Never select
only successful recordings or the best checkpoint using evaluation seeds.

The tabular example writes JSON with format tag, map, seed, training episode count, hyperparameters,
and values. Future neural checkpoints should use a versioned metadata JSON plus framework-native
weights, saved atomically. Loading must reject incompatible observation/action sizes and map or
environment versions unless an explicit override is recorded.

The headless runner emits CSV with the versioned semantic columns `map`, `episode`, `seed`, `reward`,
`steps`, `progress`, `finished`, `crashed`, and `truncated`. Experiment tooling should add its full
configuration and metadata beside this episode table rather than encoding configuration in a file
name. Generated results and checkpoints belong under ignored `artifacts/` unless a small reviewed
fixture is intentionally added for a test.
