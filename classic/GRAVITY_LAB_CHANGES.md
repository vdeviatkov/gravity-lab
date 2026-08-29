# Gravity Lab vendoring record

This directory was imported from
[`rgimad/gravity_defied_cpp`](https://github.com/rgimad/gravity_defied_cpp), commit
`91bd283959b96a7ea07e1c4c0040460334c85458` dated 2025-05-12.

The upstream authors listed in its README are rgimad, AntonEvmenenko, and Max Logaev. The original
game credits in that README are preserved. All files remain under GPL-2.0; see `LICENSE.md`.

Gravity Lab modifications:

- 2026-08-28: changed `FileStream` to inherit publicly from `std::fstream`, fixing destruction and
  compilation with modern Apple libc++.
- 2026-08-28: corrected the filename case of the `Time.h` include for portable filesystems.
- 2026-08-28: split the executable build into a reusable `GravityDefiedEngine` library and frontend.
- 2026-08-28: completed the physics snapshot buffer with all component velocities and angular
  velocities, allowing observations without changing the simulated state.
- 2026-08-28: marked read-only track/physics queries const for use by the public environment API.
- 2026-08-28: removed renderer dependencies from the level-loader public header for headless builds.
- 2026-08-28: released the level-loader-owned `GameLevel` during destruction so repeated headless
  environment creation does not leak one level instance each time, and defined the previously
  declaration-only `GameLevel` destructor needed for that cleanup.
- 2026-08-28: ignored Homebrew's stale SDL2main pkg-config entry on macOS.

The classic executable remains the faithful human-play/reference implementation. The top-level
`gravity_lab_classic_core` adapter now drives the same fixed-point `GamePhysics` and built-in level
data without creating a window. The separate `gravity_lab_core` is still a lightweight RL sandbox;
its environment ID and experiment results must not be mixed with the classic adapter.
