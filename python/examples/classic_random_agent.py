from __future__ import annotations

import argparse
import random
from pathlib import Path

from gravity_lab import CLASSIC_ACTION_COUNT, ClassicConfig, ClassicGravityEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Random baseline on the faithful classic physics")
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--level-pack", type=Path)
    parser.add_argument("--track", type=int, default=0)
    parser.add_argument("--league", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = ClassicConfig(args.group, args.track, args.league, args.frame_skip, args.max_steps, args.seed)
    exploration = random.Random(args.seed + 10_000)
    with ClassicGravityEnv(config, args.level_pack) as env:
        for episode in range(args.episodes):
            env.reset(args.seed + episode)
            total_reward = 0.0
            steps = 0
            while True:
                result = env.step(exploration.randrange(CLASSIC_ACTION_COUNT))
                total_reward += result.reward
                steps += 1
                if result.terminated or result.truncated:
                    break
            print(
                f"episode={episode} track={env.track_name!r} reward={total_reward:.6f} steps={steps} "
                f"progress={result.observation[0]:.6f} finished={result.finished} "
                f"crashed={result.crashed} truncated={result.truncated}"
            )


if __name__ == "__main__":
    main()
