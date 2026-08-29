from __future__ import annotations

import argparse
import random
from pathlib import Path

from gravity_lab import ACTION_COUNT, Config, GravityEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible random-agent baseline")
    parser.add_argument("--map", type=Path, default=Path("maps/training.gdmap"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3_000)
    args = parser.parse_args()

    exploration_rng = random.Random(args.seed + 10_000)
    config = Config(args.dt, args.frame_skip, args.max_steps, args.seed)
    with GravityEnv(args.map, config) as env:
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            while True:
                result = env.step(exploration_rng.randrange(ACTION_COUNT))
                total_reward += result.reward
                steps += 1
                if result.terminated or result.truncated:
                    break
            print(
                f"episode={episode} reward={total_reward:.4f} steps={steps} "
                f"finished={result.finished} crashed={result.crashed} truncated={result.truncated}"
            )


if __name__ == "__main__":
    main()
