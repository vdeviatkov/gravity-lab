# Faithful classic reinforcement-learning API

Environment ID `gravity-lab-classic-v1` runs the vendored classic game's `LevelLoader` and integer
fixed-point `GamePhysics` directly. It does not approximate the classic game with the lightweight
sandbox, and it does not create an SDL window during reset, stepping, training, or tests. The
playable `GravityDefied` executable and the headless environment therefore share physics and level
data while keeping rendering and menu timing outside the learning loop.

## Build

SDL2, SDL2_image, SDL2_ttf, pkg-config, CMake 3.20+, and a C++20 compiler are required. On macOS:

```sh
brew install sdl2 sdl2_image sdl2_ttf pkg-config
cmake -S . -B build-classic-rl \
  -DGRAVITY_LAB_BUILD_CLASSIC=ON \
  -DGRAVITY_LAB_BUILD_DESKTOP=OFF
cmake --build build-classic-rl --config Release
ctest --test-dir build-classic-rl -C Release --output-on-failure
```

Use `libsdl2-dev`, `libsdl2-image-dev`, and `libsdl2-ttf-dev` on Debian/Ubuntu. On Windows, use the
MinGW toolchain supported by the vendored port. Multi-configuration generators put executables and
the shared library under `Release/`.

The important outputs are:

- `classic/GravityDefied`: human-playable game;
- `gravity_lab_classic_headless`: random and throttle baselines;
- `gravity_lab_classic_q`: standard-library-only C++ tabular Q-learning;
- `gravity_lab_classic_viewer`: faithful graphical playback of an exported dense Q-policy;
- `libgravity_lab_classic`: stable C ABI used by Python and other languages.

## Environment contract

Construct an environment with `level_group` in `[0, 2]`, a valid zero-based `track`, `league` in
`[0, 3]`, `frame_skip` in `[1, 100]`, an episode limit, and a seed. An empty level-pack path uses
the embedded `levels.mrg`; a path selects a compatible custom `.mrg` pack. `frame_skip` repeats one
agent action across that many original physics updates. The original game does not expose a safe
continuous time-step setting, so v1 deliberately keeps its internal tick unchanged.

`reset(seed)` restores the selected track and bike. `step(action)` returns the next observation,
reward, `terminated`, `truncated`, finish/crash details, and the original physics return code. A
finish or crash terminates; the configured agent-step limit truncates. Algorithms may bootstrap
through truncation but not through termination. `wheelie_finish` mirrors the original game's
"Wheelie!" condition: the finish was reached without the front wheel having touched the track.

Actions are identical in C++ and Python:

| Value | Action |
|---:|---|
| 0 | coast |
| 1 | throttle |
| 2 | brake |
| 3 | lean back/left |
| 4 | lean forward/right |
| 5 | throttle + lean back |
| 6 | throttle + lean forward |
| 7 | brake + lean back |
| 8 | brake + lean forward |

The observation is 36 `double` values:

| Indices | Meaning |
|---|---|
| 0 | normalized course progress `(center_x - start_x) / (finish_x - start_x)` |
| 1 | `1 - progress` |
| 2 | start-line-crossed flag |
| 3 | bike league divided by 3 |
| 4..27 | six physics points, four values each: relative x/10, relative y/10, vx/20, vy/20 |
| 28..35 | obstacle-distance sensor: 8 rays cast from the center point, evenly spaced by full turns, each the distance (in `[0, 1]`, `1.0` = nothing in range) to the nearest track polyline segment that ray intersects |

Physics point 0 is the center reference, 1 is the front wheel, 2 is the rear wheel, and 3–5 are the
remaining frame/rider constraint points from the original engine. Positions are relative to point
0; velocities remain absolute. Values are not clipped. This dense vector is appropriate for a
small neural network. The tabular examples intentionally coarsen six selected values, so their
state space is not equivalent to the full observation.

The obstacle sensor treats the track's ground polyline as a chain of bounded line segments (each
just its two endpoints) rather than infinite lines: a ray only counts as hitting a segment if the
intersection falls within that segment's own span. Ray 0 points along the direction of increasing
progress (`+x`); the remaining rays are spaced 45° apart around it. Rays search only the polyline
segments near the bike's current position (`kObstacleSearchRadius` on each side) and report
`kObstacleMaxRange` (normalized to `1.0`) when nothing is hit; both constants live next to
`kObstacleRayCount` in `classic_environment.cpp`/`.hpp` and are tunable.

The v1 reward is:

```text
r = 0.1 * (center_x_after - center_x_before) - 0.001
    + 10 if finished
    - 5 if crashed
```

The environment, action order, scaling, reward, and ending rules are versioned together. Any
semantic change requires a new environment ID.

## Python training

The wrapper is dependency-free and Gym-like without requiring Gymnasium. It locates the shared
library in `build-classic-rl/`; set `GRAVITY_LAB_CLASSIC_LIBRARY` for another build directory.

```python
from gravity_lab import ClassicAction, ClassicConfig, ClassicGravityEnv

config = ClassicConfig(level_group=0, track=0, league=0, frame_skip=2,
                       max_episode_steps=2_000, seed=7)
with ClassicGravityEnv(config) as env:
    observation = env.reset(7)
    while True:
        transition = env.step(ClassicAction.THROTTLE)
        observation = transition.observation
        if transition.terminated or transition.truncated:
            break
```

Run the supplied baselines from the repository root:

```sh
PYTHONPATH=python python3 python/examples/classic_random_agent.py \
  --group 0 --track 0 --episodes 20 --seed 7

PYTHONPATH=python python3 python/examples/classic_tabular_q.py \
  --group 0 --track 0 --episodes 2000 --train-seed 7 \
  --eval-seed 1000007 --eval-episodes 50 \
  --checkpoint artifacts/classic_tabular_q.json

PYTHONPATH=python python3 python/examples/classic_tabular_q.py \
  --group 0 --track 0 --eval-only --eval-seed 1000007 \
  --eval-episodes 50 --checkpoint artifacts/classic_tabular_q.json
```

Add `--level-pack path/to/levels.mrg` to use a custom pack. The evaluation pass is greedy: epsilon
is exactly zero. It reports every requested episode through aggregate mean/median reward, mean
length/progress, and finish/crash rates; it never selects only the best game.

## C++ training and embedding

The ready-to-run standard-library trainer uses the same environment and tabular encoder:

```sh
./build-classic-rl/gravity_lab_classic_headless \
  --group 0 --track 0 --policy random --episodes 20 --seed 7

./build-classic-rl/gravity_lab_classic_q \
  --group 0 --track 0 --episodes 2000 --train-seed 7 \
  --eval-seed 1000007 --eval-episodes 50 \
  --checkpoint artifacts/classic_q.tsv

./build-classic-rl/gravity_lab_classic_q \
  --group 0 --track 0 --eval-only --eval-seed 1000007 \
  --eval-episodes 50 --checkpoint artifacts/classic_q.tsv
```

For your own native agent, link `gravity_lab_classic_core`, include
`gravity_lab/classic_environment.hpp`, and use `Environment::reset`/`step`. Non-C++ programs can
use the declarations in `gravity_lab/classic_c_api.h` and link `gravity_lab_classic`.

Tabular Q-learning updates one action value after each transition:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

The `gamma * max Q` term is zero for a terminal finish/crash. The examples use a separately seeded
epsilon-greedy policy during training and a deterministic greedy policy during evaluation.

## External neural training and playback

Keep DQN training in a separate repository and use `ClassicGravityEnv` as its environment. The
environment returns the same 36-value observation and accepts the same nine actions as the native
API. A typical external loop should maintain independently seeded initialization, exploration,
replay sampling, environment, and evaluation RNGs; keep replay and target-network updates out of
the environment; and evaluate with epsilon exactly zero on fixed seeds that were not used for
training or checkpoint selection.

After evaluation, export a sequential dense online Q-network as
`gravity-lab-dense-q-policy-v1`. The format includes the environment ID, elementwise input
normalization, activations, biases, and output-major weights. Python can create it directly from
PyTorch `Linear` tensors, and C++ loads it without PyTorch, Python, NumPy, or an RL framework.

```sh
./build-classic-rl/gravity_lab_classic_viewer \
  --policy artifacts/classic_policy.gdp --validate-only

./build-classic-rl/gravity_lab_classic_viewer \
  --policy artifacts/classic_policy.gdp \
  --group 0 --track 0 --league 0 --frame-skip 2 \
  --max-steps 2000 --episodes 3 --seed 2000007
```

The viewer performs deterministic greedy argmax and renders the original track, bike sprites, HUD,
and fixed-point `GamePhysics` state used by the headless environment. It never creates a second
simulation. Escape or the window close button stops playback. See
[policy-format.md](policy-format.md) for the full format, Python export example, model limitations,
and metadata requirements.

## Reproducibility and limitations

- Keep training and evaluation seeds disjoint. The classic v1 environment itself is deterministic;
  its seed is explicit for experiment identity and forward compatibility.
- Record the repository commit, environment ID, group/track/league, custom-pack hash, frame skip,
  episode limit, reward version, algorithm configuration, compiler, OS, and CPU.
- Python checkpoints use `gravity-lab-classic-tabular-q-json-v1`; C++ checkpoints use
  `gravity-lab-classic-tabular-q-tsv-v1`. Both save through a temporary file. They are not directly
  interchangeable.
- Dense neural inference uses the cross-language `gravity-lab-dense-q-policy-v1` format. It is a
  deployment artifact, not a resumable training checkpoint; keep optimizer/replay state and a
  complete experiment sidecar in the external training repository.
- The upstream engine stores important level and bike parameters in process-global static state.
  Therefore v1 permits only one active classic environment per process. Use multiple worker
  processes—not threads or multiple environment objects—for parallel collection.
- Rendering is optional and excluded from training/evaluation timing. Do not commit videos, large
  logs, or generated checkpoints; `artifacts/` is ignored.
- The included tabular learner is a tested educational baseline, not a claimed performance result.
  DQN training, replay buffers, target networks, and measured benchmarks remain external experiment
  work; Gravity Lab deliberately provides only the tested environment and inference boundary.
