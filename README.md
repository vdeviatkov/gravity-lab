# Gravity Lab

Gravity Lab is a small, deterministic 2D motorcycle-trials game built for both human play and
reinforcement-learning experiments. The simulation is a dependency-free C++20 library. The same
environment is exposed to native C++, a stable C ABI, and Python through `ctypes`; rendering never
runs during headless training.

This is an original educational project inspired by the control style of classic motorcycle-trials
games. It is not a source port, includes no third-party game code or assets, and is not affiliated
with the owners of *Gravity Defied*.

## What works

- Deterministic piecewise-linear terrain and lightweight motorcycle physics
- Selectable map, integration time step, action repeat (`frame_skip`), episode limit, and seed
- Nine discrete actions, including simultaneous throttle/brake and rider lean
- Fixed 12-number observation vector and explicit reward/termination signals
- Native C++ API, shared C ABI, and dependency-free Python wrapper
- Random and heuristic C++ rollouts plus random and tabular-Q Python examples
- Optional SDL2 renderer with arrow/WASD controls
- C++ and Python determinism/contract tests on macOS, Linux, and Windows

## Build

You need CMake 3.20+ and a C++20 compiler (Apple Clang, GCC, MSVC, or Clang). SDL2 is optional.

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

If SDL2 is available, CMake also builds `gravity_lab_play`. Install it with Homebrew
(`brew install sdl2`), apt (`sudo apt install libsdl2-dev`), or vcpkg on Windows. To build only the
portable simulation and training interfaces:

```sh
cmake -S . -B build -DGRAVITY_LAB_BUILD_DESKTOP=OFF
cmake --build build --config Release
```

## Play

Run from the repository root so relative map paths resolve:

```sh
./build/gravity_lab_play maps/hills.gdmap
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
include/gravity_lab/  public C++, C, environment, map, and data-type APIs
src/                  deterministic simulation and C ABI implementation
maps/                 version-controlled curriculum maps
python/gravity_lab/   dependency-free ctypes wrapper
python/examples/      random baseline and tabular Q-learning example
tests/                native and cross-language contract tests
docs/                 environment semantics and reproducibility rules
```

## Scope

The current physics model is intentionally compact and fully inspectable; it is not a rigid-body
simulator or a byte-for-byte recreation of another game. The next useful milestone is a measured
DQN baseline built outside the environment library, followed by replay/target-network ablations.
Scores and target performance are deliberately not claimed until reproducible experiment runs are
checked in as small metadata files.

## License

MIT. See [LICENSE](LICENSE).
