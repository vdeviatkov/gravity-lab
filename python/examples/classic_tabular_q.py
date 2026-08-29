"""Reproducible tabular Q-learning baseline for gravity-lab-classic-v1."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from gravity_lab import CLASSIC_ACTION_COUNT, ClassicConfig, ClassicGravityEnv


def encode(observation: tuple[float, ...]) -> tuple[int, ...]:
    # Coarsened engineered state: progress, center velocity, wheel geometry, and start status.
    values = observation[0], observation[6], observation[7], observation[9], observation[13], observation[2]
    scales = 30.0, 12.0, 12.0, 30.0, 30.0, 1.0
    return tuple(round(max(-3.0, min(3.0, value)) * scale) for value, scale in zip(values, scales))


def state_key(state: tuple[int, ...]) -> str:
    return ",".join(map(str, state))


def greedy(values: list[float]) -> int:
    return max(range(CLASSIC_ACTION_COUNT), key=values.__getitem__)


def epsilon_at(episode: int, episodes: int) -> float:
    return 0.05 + 0.95 * math.exp(-episode / max(1.0, episodes * 0.30))


def q_learning_update(
    current: float,
    reward: float,
    next_values: list[float],
    terminated: bool,
    alpha: float,
    gamma: float,
) -> float:
    bootstrap = 0.0 if terminated else gamma * max(next_values)
    return current + alpha * (reward + bootstrap - current)


def load_checkpoint(path: Path) -> tuple[dict[str, list[float]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "gravity-lab-classic-tabular-q-json-v1":
        raise ValueError("unsupported checkpoint format")
    q = payload.get("q")
    if not isinstance(q, dict):
        raise ValueError("checkpoint has no Q table")
    if any(
        not isinstance(values, list)
        or len(values) != CLASSIC_ACTION_COUNT
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
        for values in q.values()
    ):
        raise ValueError("checkpoint action dimension does not match classic-v1")
    return q, payload


def save_checkpoint(path: Path, q: dict[str, list[float]], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "gravity-lab-classic-tabular-q-json-v1", **metadata, "q": q}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def evaluate(
    env: ClassicGravityEnv,
    q: dict[str, list[float]],
    episodes: int,
    seed: int,
) -> None:
    rewards: list[float] = []
    lengths: list[int] = []
    progresses: list[float] = []
    finishes = crashes = 0
    for episode in range(episodes):
        observation = env.reset(seed + episode)
        total_reward = 0.0
        steps = 0
        while True:
            action = greedy(q.get(state_key(encode(observation)), [0.0] * CLASSIC_ACTION_COUNT))
            result = env.step(action)
            total_reward += result.reward
            steps += 1
            observation = result.observation
            if result.terminated or result.truncated:
                finishes += int(result.finished)
                crashes += int(result.crashed)
                break
        rewards.append(total_reward)
        lengths.append(steps)
        progresses.append(observation[0])
    print(
        f"evaluation_episodes={episodes} mean_reward={statistics.fmean(rewards):.6f} "
        f"median_reward={statistics.median(rewards):.6f} mean_length={statistics.fmean(lengths):.2f} "
        f"mean_progress={statistics.fmean(progresses):.6f} finish_rate={finishes / episodes:.6f} "
        f"crash_rate={crashes / episodes:.6f} exploration=0"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular Q-learning on faithful classic physics")
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--level-pack", type=Path)
    parser.add_argument("--track", type=int, default=0)
    parser.add_argument("--league", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--episodes", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=1_000_001)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/classic_tabular_q.json"))
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if not args.eval_only and args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0.0 < args.alpha <= 1.0:
        parser.error("--alpha must be in (0, 1]")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0, 1]")

    config = ClassicConfig(
        args.group, args.track, args.league, args.frame_skip, args.max_steps, args.train_seed
    )
    q: dict[str, list[float]] = {}
    if args.eval_only:
        q, payload = load_checkpoint(args.checkpoint)
        expected = [args.group, args.track, args.league, args.frame_skip, args.max_steps]
        if payload.get("environment_config") != expected:
            raise ValueError("checkpoint environment configuration does not match CLI configuration")
        expected_pack = str(args.level_pack.resolve()) if args.level_pack else "embedded"
        if payload.get("level_pack") != expected_pack:
            raise ValueError("checkpoint level pack does not match CLI configuration")

    exploration = random.Random(args.train_seed + 10_000)
    with ClassicGravityEnv(config, args.level_pack) as env:
        if not args.eval_only:
            for episode in range(args.episodes):
                observation = env.reset(args.train_seed + episode)
                state = encode(observation)
                epsilon = epsilon_at(episode, args.episodes)
                total_reward = 0.0
                while True:
                    values = q.setdefault(state_key(state), [0.0] * CLASSIC_ACTION_COUNT)
                    action = exploration.randrange(CLASSIC_ACTION_COUNT) if exploration.random() < epsilon else greedy(values)
                    result = env.step(action)
                    next_state = encode(result.observation)
                    next_values = q.setdefault(state_key(next_state), [0.0] * CLASSIC_ACTION_COUNT)
                    values[action] = q_learning_update(
                        values[action], result.reward, next_values, result.terminated, args.alpha, args.gamma
                    )
                    total_reward += result.reward
                    state = next_state
                    if result.terminated or result.truncated:
                        break
                if episode % max(1, args.episodes // 10) == 0:
                    print(
                        f"episode={episode} reward={total_reward:.6f} epsilon={epsilon:.6f} "
                        f"states={len(q)} finished={result.finished}"
                    )
            save_checkpoint(
                args.checkpoint,
                q,
                {
                    "environment_id": env.environment_id,
                    "environment_config": [args.group, args.track, args.league, args.frame_skip, args.max_steps],
                    "level_pack": str(args.level_pack.resolve()) if args.level_pack else "embedded",
                    "track_name": env.track_name,
                    "train_seed": args.train_seed,
                    "exploration_seed": args.train_seed + 10_000,
                    "evaluation_seed": args.eval_seed,
                    "episodes": args.episodes,
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "encoder": "classic-tabular-v1",
                },
            )
            print(f"saved {args.checkpoint}")
        evaluate(env, q, args.eval_episodes, args.eval_seed)


if __name__ == "__main__":
    main()
