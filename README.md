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
- Selectable map, integration time step, action repeat (`frame_skip`), episode limit, and seed
- Nine discrete actions, including simultaneous throttle/brake and rider lean
- Fixed 12-number observation vector and explicit reward/termination signals
- Native C++ API, shared C ABI, and dependency-free Python wrapper
- Random and heuristic C++ rollouts plus random and tabular-Q Python examples
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

It can also be included in the top-level build with
`-DGRAVITY_LAB_BUILD_CLASSIC=ON`; the resulting executable is under the build tree's `classic/`
directory.

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

See [docs/environment.md](docs/environment.md) for the full Markov decision process contract and
[docs/reproducibility.md](docs/reproducibility.md) before comparing agents.

## Repository layout

```text
apps/                 SDL2 player and headless rollout executable
classic/              vendored faithful GPL C++/SDL2 port and original port assets
include/gravity_lab/  public C++, C, environment, map, and data-type APIs
src/                  deterministic simulation and C ABI implementation
maps/                 version-controlled curriculum maps
python/gravity_lab/   dependency-free ctypes wrapper
python/examples/      random baseline and tabular Q-learning example
tests/                native and cross-language contract tests
docs/                 environment semantics and reproducibility rules
```

## Scope

There are currently two physics implementations. `classic/` is the faithful game used for normal
play and visual reference. `gravity_lab_core` is the deterministic, instrumented RL sandbox. A
policy trained in the sandbox must not be described as trained on the classic physics, and direct
policy transfer is not guaranteed. The next integration milestone is a headless adapter around the
classic fixed-point `GamePhysics`, after which classic and sandbox environment IDs will remain
explicit so experiment data cannot mix them accidentally.

The next learning milestone is a measured DQN baseline built outside the environment library,
followed by replay/target-network ablations. Scores and target performance are deliberately not
claimed until reproducible experiment runs are checked in as small metadata files.

## License

GPL-2.0-only. See [LICENSE](LICENSE) and the complete text in [classic/LICENSE.md](classic/LICENSE.md).
