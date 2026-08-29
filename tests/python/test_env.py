from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from gravity_lab import Action, Config, GravityEnv  # noqa: E402


class EnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.map = ROOT / "maps" / "training.gdmap"

    def test_seed_and_trajectory_are_reproducible(self) -> None:
        config = Config(seed=7, max_episode_steps=20)
        with GravityEnv(self.map, config) as first, GravityEnv(self.map, config) as second:
            self.assertEqual(first.reset(123), second.reset(123))
            for action in [Action.THROTTLE, Action.THROTTLE_LEAN_BACK, Action.LEAN_FORWARD]:
                self.assertEqual(first.step(action), second.step(action))

    def test_shapes_and_time_limit(self) -> None:
        with GravityEnv(self.map, Config(max_episode_steps=2)) as env:
            self.assertEqual(len(env.reset()), 12)
            first = env.step(Action.COAST)
            self.assertFalse(first.truncated)
            second = env.step(Action.COAST)
            self.assertTrue(second.truncated)
            self.assertFalse(second.terminated)

    def test_invalid_action_reports_native_error(self) -> None:
        with GravityEnv(self.map) as env:
            with self.assertRaisesRegex(RuntimeError, "action"):
                env.step(99)


if __name__ == "__main__":
    unittest.main()
