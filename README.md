# Gravity Lab

Gravity Lab combines a faithful, GPL-licensed desktop port of the classic motorcycle-trials game
with a deterministic environment for reinforcement-learning experiments. The training simulation
is a dependency-free C++20 library exposed through native C++, a stable C ABI, and Python `ctypes`;
rendering never runs during headless training.

The faithful game in `classic/` is vendored from
[`rgimad/gravity_defied_cpp`](https://github.com/rgimad/gravity_defied_cpp) at upstream commit
`91bd283959b96a7ea07e1c4c0040460334c85458`. It retains the original port's authorship and GPL-2.0
license. This project is not affiliated with Codebrew Software or the owners of the *Gravity Defied*
name and branding.

## What works

- Faithful classic game with its original ported physics, levels, menus, sprites, HUD, and renderer
- Separate deterministic piecewise-linear RL sandbox with inspectable lightweight physics
- Selectable built-in/custom classic levels, bike league, action repeat (`frame_skip`), episode
  limit, and seed
- Nine discrete actions, including simultaneous throttle/brake and rider lean
- Versioned observations: 36 values from classic fixed-point physics or 12 from the RL sandbox
- Explicit reward, finish/crash termination, and time-limit truncation signals
- Native C++ API, shared C ABI, and dependency-free Python wrapper
- Random rollouts and tabular Q-learning in both C++20 and Python
- Versioned, framework-neutral dense Q-policy export with matching Python/C++ inference
- Faithful graphical policy viewer using the same physics state as headless training
- Optional SDL2 RL-sandbox viewer with arrow/WASD controls
- C++ and Python determinism/contract tests on macOS, Linux, and Windows

## Build the faithful game

The classic game needs SDL2, SDL2_image, and SDL2_ttf. On macOS:

```sh
brew install sdl2 sdl2_image sdl2_ttf pkg-config
cmake -S classic -B build-classic
cmake --build build-classic --config Release
./build-classic/GravityDefied
```

On Debian/Ubuntu, install `libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev`. The vendored upstream
CMake configuration downloads its MinGW SDL dependencies when building on Windows. The executable
uses Up/Down for throttle/brake and Left/Right to move the rider. Menus, track/league selection,
high scores, help, classic levels, and custom `.mrg` packs behave as in the reference port:

```sh
./build-classic/GravityDefied path/to/custom-levels.mrg
```

For the faithful game's headless learning API, use the integrated build:

```sh
cmake -S . -B build-classic-rl \
  -DGRAVITY_LAB_BUILD_CLASSIC=ON \
  -DGRAVITY_LAB_BUILD_DESKTOP=OFF
cmake --build build-classic-rl --config Release
ctest --test-dir build-classic-rl -C Release --output-on-failure
```

This produces the playable game under `build-classic-rl/classic/` plus the native environment,
shared Python library, headless runner, C++ Q-learning executable, and learned-policy viewer.

## Build the RL environment

You need CMake 3.20+ and a C++20 compiler (Apple Clang, GCC, MSVC, or Clang). SDL2 is optional.

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

If SDL2 is available, CMake also builds `gravity_lab_sandbox`. Install it with Homebrew
(`brew install sdl2`), apt (`sudo apt install libsdl2-dev`), or vcpkg on Windows. To build only the
portable simulation and training interfaces:

```sh
cmake -S . -B build -DGRAVITY_LAB_BUILD_DESKTOP=OFF
cmake --build build --config Release
```

## View the RL sandbox

Run from the repository root so relative map paths resolve:

```sh
./build/gravity_lab_sandbox maps/hills.gdmap
```

Use Up/W to accelerate, Down/S to brake, Left/A to lean back, Right/D to lean forward, R to reset,
and Escape to quit. Acceleration and leaning can be held together. Windows multi-configuration
builds place the executable under `build/Release/`.

## Train and evaluate

### Faithful classic physics

Build with `GRAVITY_LAB_BUILD_CLASSIC=ON`, then run a Python baseline or trainer:

```sh
PYTHONPATH=python python3 python/examples/classic_random_agent.py \
  --group 0 --track 0 --episodes 20 --seed 7
PYTHONPATH=python python3 python/examples/classic_tabular_q.py \
  --group 0 --track 0 --episodes 2000 --train-seed 7 \
  --eval-seed 1000007 --eval-episodes 50 \
  --checkpoint artifacts/classic_tabular_q.json
```

The same workflow is available without Python:

```sh
./build-classic-rl/gravity_lab_classic_headless \
  --group 0 --track 0 --policy random --episodes 20 --seed 7
./build-classic-rl/gravity_lab_classic_q \
  --group 0 --track 0 --episodes 2000 --train-seed 7 \
  --eval-seed 1000007 --eval-episodes 50 \
  --checkpoint artifacts/classic_q.tsv
```

Both trainers save a checkpoint and immediately run exploration-free evaluation. Add
`--eval-only` to load a checkpoint, or `--level-pack path/to/custom-levels.mrg` to select a custom
pack. No performance is claimed by the repository; run measured experiments before drawing
conclusions. See [docs/classic-rl.md](docs/classic-rl.md) for the complete API, observations,
actions, rewards, native embedding example, and limitations.

Train a neural network in a separate experiment repository against `ClassicGravityEnv`, then
export its dense layers to the portable policy format. The exported policy can be validated and
played by C++ without Python or a neural-network framework:

```sh
./build-classic-rl/gravity_lab_classic_viewer \
  --policy artifacts/classic_policy.gdp --validate-only
./build-classic-rl/gravity_lab_classic_viewer \
  --policy artifacts/classic_policy.gdp \
  --group 0 --track 0 --frame-skip 2 --episodes 3 --seed 2000007
```

See [docs/policy-format.md](docs/policy-format.md) for the PyTorch-compatible export example,
format contract, input normalization, and required reproducibility sidecar.

### Lightweight sandbox

The headless executable is useful for C++ baselines and automation:

```sh
./build/gravity_lab_headless --map maps/training.gdmap --policy random --episodes 10 --seed 7
./build/gravity_lab_headless --map maps/steps.gdmap --policy heuristic --dt 0.008333333333 --frame-skip 4
```

It prints one CSV row per episode and never renders. For Python, build the shared library first,
then point `PYTHONPATH` at the source package:

```sh
PYTHONPATH=python python3 python/examples/random_agent.py --episodes 10 --seed 7
PYTHONPATH=python python3 python/examples/tabular_q.py --episodes 500 --map maps/training.gdmap
PYTHONPATH=python python3 -m unittest discover -s tests/python -v
```

The wrapper automatically searches common `build/` locations. Set `GRAVITY_LAB_LIBRARY` to an
absolute shared-library path for a custom build layout.

Native agents link `gravity_lab_core` and include `gravity_lab/environment.hpp`:

```cpp
auto map = gravity_lab::Map::load("maps/training.gdmap");
gravity_lab::Environment env(std::move(map), {.seed = 42});
auto observation = env.reset(42);
while (!env.done()) {
    auto transition = env.step(gravity_lab::Action::Throttle);
}
```

See [docs/environment.md](docs/environment.md) for the sandbox Markov decision process contract and
[docs/reproducibility.md](docs/reproducibility.md) before comparing agents.

## Repository layout

```text
apps/                 SDL2 players, learned-policy viewer, rollouts, and C++ tabular trainer
classic/              vendored faithful GPL C++/SDL2 port and original port assets
include/gravity_lab/  public C++, C, environment, map, and data-type APIs
src/                  deterministic simulation and C ABI implementation
maps/                 version-controlled curriculum maps
python/gravity_lab/   dependency-free ctypes wrapper and portable dense-policy exporter
python/examples/      random baselines and tabular Q-learning examples
tests/                native and cross-language contract tests
docs/                 environment semantics and reproducibility rules
```

## Scope

There are two physics implementations. `gravity-lab-classic-v1` directly wraps the faithful
vendored fixed-point `GamePhysics` used for normal play. `gravity-lab-sandbox-v1` is a separate,
small, inspectable simulation. A policy trained in the sandbox must not be described as trained on
classic physics, and direct transfer is not guaranteed. The distinct IDs, observations, and
checkpoint metadata keep their experiment data separate.

Neural training, replay buffers, target networks, and experiment tracking belong in a separate
training repository. This repository supplies the stable environment and portable deployment
boundary. Scores and target performance are deliberately not claimed until reproducible experiment
runs are published with their configurations and metadata.

## License

GPL-2.0-only. See [LICENSE](LICENSE) and the complete text in [classic/LICENSE.md](classic/LICENSE.md).
