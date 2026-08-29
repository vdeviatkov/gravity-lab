from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python" / "examples"))

from classic_tabular_q import (  # noqa: E402
    epsilon_at,
    load_checkpoint,
    q_learning_update,
    save_checkpoint,
)


class ClassicQLearningTest(unittest.TestCase):
    def test_q_update_bootstraps_only_nonterminal_transitions(self) -> None:
        next_values = [0.0, 4.0] + [0.0] * 7
        self.assertAlmostEqual(q_learning_update(2.0, 1.0, next_values, False, 0.5, 0.9), 3.3)
        self.assertAlmostEqual(q_learning_update(2.0, 1.0, next_values, True, 0.5, 0.9), 1.5)

    def test_epsilon_schedule_is_bounded_and_decreases(self) -> None:
        values = [epsilon_at(episode, 1_000) for episode in (0, 100, 1_000, 10_000)]
        self.assertEqual(values[0], 1.0)
        self.assertTrue(all(0.05 <= value <= 1.0 for value in values))
        self.assertEqual(values, sorted(values, reverse=True))

    def test_checkpoint_round_trip_and_format_validation(self) -> None:
        q = {"1,2,3,4,5,6": [float(value) for value in range(9)]}
        metadata: dict[str, object] = {
            "environment_id": "gravity-lab-classic-v1",
            "environment_config": [0, 0, 0, 2, 100],
            "level_pack": "embedded",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            save_checkpoint(path, q, metadata)
            loaded_q, payload = load_checkpoint(path)
            self.assertEqual(loaded_q, q)
            self.assertEqual(payload["environment_config"], metadata["environment_config"])
            self.assertEqual(payload["format"], "gravity-lab-classic-tabular-q-json-v1")


if __name__ == "__main__":
    unittest.main()
