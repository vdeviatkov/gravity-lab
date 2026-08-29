"""Small educational baseline; continuous observations are deliberately coarsened."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from gravity_lab import ACTION_COUNT, Config, GravityEnv


def encode(observation: tuple[float, ...]) -> tuple[int, ...]:
    selected = observation[0], observation[2], observation[4], observation[7], observation[9], observation[11]
    scales = 20.0, 8.0, 6.0, 6.0, 8.0, 1.0
    return tuple(round(max(-2.0, min(2.0, value)) * scale) for value, scale in zip(selected, scales))


def key(state: tuple[int, ...]) -> str:
    return ",".join(map(str, state))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact tabular Q-learning baseline")
    parser.add_argument("--map", type=Path, default=Path("maps/training.gdmap"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tabular_q.json"))
    parser.add_argument("--max-steps", type=int, default=1_500)
    args = parser.parse_args()

    exploration_rng = random.Random(args.seed + 20_000)
    q: dict[str, list[float]] = {}
    alpha, gamma = 0.15, 0.99
    config = Config(max_episode_steps=args.max_steps, seed=args.seed)
    with GravityEnv(args.map, config) as env:
        for episode in range(args.episodes):
            observation = env.reset(seed=args.seed + episode)
            state = encode(observation)
            epsilon = 0.05 + 0.95 * math.exp(-episode / max(1.0, args.episodes * 0.25))
            total_reward = 0.0
            while True:
                values = q.setdefault(key(state), [0.0] * ACTION_COUNT)
                action = exploration_rng.randrange(ACTION_COUNT) if exploration_rng.random() < epsilon else max(
                    range(ACTION_COUNT), key=values.__getitem__
                )
                result = env.step(action)
                next_state = encode(result.observation)
                next_values = q.setdefault(key(next_state), [0.0] * ACTION_COUNT)
                bootstrap = 0.0 if result.terminated else gamma * max(next_values)
                values[action] += alpha * (result.reward + bootstrap - values[action])
                total_reward += result.reward
                state = next_state
                if result.terminated or result.truncated:
                    break
            if episode % max(1, args.episodes // 10) == 0:
                print(f"episode={episode} reward={total_reward:.4f} epsilon={epsilon:.4f} states={len(q)}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "gravity-lab-tabular-q-v1",
        "map": str(args.map),
        "seed": args.seed,
        "episodes": args.episodes,
        "alpha": alpha,
        "gamma": gamma,
        "q": q,
    }
    args.checkpoint.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
