from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from gravity_lab import (  # noqa: E402
    ClassicAction,
    ClassicConfig,
    ClassicGravityEnv,
    classic_library_available,
)


@unittest.skipUnless(classic_library_available(), "classic RL native library is not built")
class ClassicEnvironmentTest(unittest.TestCase):
    def trajectory(self) -> list[object]:
        with ClassicGravityEnv(ClassicConfig(max_episode_steps=20)) as env:
            env.reset(123)
            return [env.step(action) for action in (
                ClassicAction.THROTTLE,
                ClassicAction.THROTTLE_LEAN_BACK,
                ClassicAction.THROTTLE_LEAN_FORWARD,
            )]

    def test_trajectory_is_reproducible(self) -> None:
        self.assertEqual(self.trajectory(), self.trajectory())

    def test_contract_and_time_limit(self) -> None:
        with ClassicGravityEnv(ClassicConfig(max_episode_steps=2)) as env:
            observation = env.reset(7)
            self.assertEqual(len(observation), 28)
            self.assertTrue(all(math.isfinite(value) for value in observation))
            self.assertTrue(env.track_name)
            self.assertGreater(env.track_count(0), 0)
            with self.assertRaisesRegex(RuntimeError, "level_group"):
                env.track_count(99)
            env.step(ClassicAction.COAST)
            final = env.step(ClassicAction.COAST)
            self.assertTrue(final.truncated)
            self.assertFalse(final.terminated)

    def test_invalid_action_is_reported(self) -> None:
        with ClassicGravityEnv() as env:
            with self.assertRaisesRegex(RuntimeError, "action"):
                env.step(99)

    def test_invalid_python_configuration_is_rejected_before_native_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_episode_steps"):
            ClassicGravityEnv(ClassicConfig(max_episode_steps=-1))
        with ClassicGravityEnv() as env:
            with self.assertRaisesRegex(ValueError, "seed"):
                env.reset(-1)


if __name__ == "__main__":
    unittest.main()
