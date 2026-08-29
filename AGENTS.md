# Contributor instructions

- Preserve the boundary between deterministic simulation (`src/`, `include/`) and agents/rendering.
- Do not add an RL framework or renderer dependency to `gravity_lab_core`.
- Keep maps and experiment inputs text-based, small, versioned, and reviewable.
- Any observation, action, reward, physics, or termination change is an environment-contract change;
  update documentation and cross-language tests in the same commit.
- New randomness must have a named, independently configurable seed. Never use global RNG state.
- Tests and training must run headlessly. Do not commit generated checkpoints, videos, or large logs.
- Do not claim performance without fixed evaluation seeds, full distributions, and recorded metadata.
- Run CTest and Python unittest before submitting changes.
- The vendored `classic/` tree is GPL-2.0-only. Preserve upstream authorship and asset notices.
- Mark modifications to files under `classic/` with the date and summarize them in
  `classic/GRAVITY_LAB_CHANGES.md`.
- `gravity-lab-classic-v1` must continue to execute the vendored `GamePhysics`; do not substitute
  the lightweight sandbox physics while retaining the classic environment ID.
- The upstream engine uses process-global physics and level-loader state. Until that is removed,
  preserve the one-active-classic-environment guard and use subprocesses for parallel rollouts.
