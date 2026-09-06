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
- Selectable built-in/custom classic levels, bike league, action repeat (`frame_skip`), episode
  limit, and seed
- Nine discrete actions, including simultaneous throttle/brake and rider lean
- Versioned, fixed-region observation (see docs/classic-rl.md) from the classic fixed-point physics
- Finish/crash termination and time-limit truncation signals (reward is left to the caller --
  see docs/classic-rl.md)
- Native C++ API, shared C ABI, and dependency-free Python wrapper
- Versioned, framework-neutral dense Q-policy export with matching Python/C++ inference
- Faithful graphical policy viewer using the same physics state as headless training
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

## Build the RL environment

For the faithful game's headless learning API, use the integrated build (CMake 3.20+ and a C++20
compiler are required; SDL2/SDL2_image/SDL2_ttf are needed too, since the classic game and its
policy viewer both render with them):

```sh
cmake -S . -B build-classic-rl -DGRAVITY_LAB_BUILD_CLASSIC=ON
cmake --build build-classic-rl --config Release
ctest --test-dir build-classic-rl -C Release --output-on-failure
```

This produces the playable game under `build-classic-rl/classic/` plus the native environment,
shared Python library, headless runner, and learned-policy viewer. Multi-configuration generators
(Windows) put executables and the shared library under `Release/`.

## Train and evaluate

Build with `GRAVITY_LAB_BUILD_CLASSIC=ON`, then drive the environment directly, without Python:

```sh
./build-classic-rl/gravity_lab_classic_headless \
  --group 0 --track 0 --policy random --episodes 20 --seed 7
```

It prints one CSV row per episode and never renders -- a quick native sanity check of the
environment after a change, not a training tool. From Python:

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

No performance is claimed by the repository; run measured experiments before drawing conclusions.
See [docs/classic-rl.md](docs/classic-rl.md) for the complete API, observations, actions, reward
guidance, native embedding example, and limitations.

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

## Repository layout

```text
apps/                 headless baseline runner and learned-policy viewer
classic/              vendored faithful GPL C++/SDL2 port and original port assets
include/gravity_lab/  public C++/C environment and dense-policy APIs
src/                  classic environment, C ABI, and dense-policy implementation
python/gravity_lab/   dependency-free ctypes wrapper and portable dense-policy exporter
python/examples/      minimal classic-environment usage example
tests/                native and cross-language contract tests
docs/                 environment semantics and reproducibility rules
```

## Scope

This repository supplies exactly one environment (`gravity-lab-classic-v1`, the faithful vendored
fixed-point `GamePhysics`) and a portable deployment boundary for trained policies. Neural
training, replay buffers, target networks, reward design, and experiment tracking belong in a
separate training repository. Scores and target performance are deliberately not claimed until
reproducible experiment runs are published with their configurations and metadata.

## License

GPL-2.0-only. See [LICENSE](LICENSE) and the complete text in [classic/LICENSE.md](classic/LICENSE.md).
