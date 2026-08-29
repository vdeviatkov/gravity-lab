from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from gravity_lab import DenseLayer, DenseQPolicy  # noqa: E402


class DensePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ROOT / "tests" / "data" / "dense_policy_fixture.gdp"

    def test_shared_fixture_inference(self) -> None:
        policy = DenseQPolicy.load(self.fixture)
        self.assertEqual(policy.environment_id, "fixture-v1")
        self.assertEqual(policy.observation_size, 3)
        self.assertEqual(policy.action_count, 2)
        self.assertEqual(policy.evaluate((1.0, 2.0, -1.0)), (4.0, -4.0))
        self.assertEqual(policy.action((1.0, 2.0, -1.0)), 0)

    def test_python_checkpoint_round_trip(self) -> None:
        policy = DenseQPolicy(
            "test-environment",
            [
                DenseLayer.from_values([[1.0, -1.0], [0.5, 0.5]], [0.0, 1.0], "relu"),
                DenseLayer.from_values([[2.0, -1.0]], [0.25], "linear"),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.gdp"
            policy.save(path)
            loaded = DenseQPolicy.load(path)
            self.assertEqual(loaded.evaluate((3.0, 1.0)), policy.evaluate((3.0, 1.0)))

    def test_invalid_dimensions_and_nonfinite_input_are_rejected(self) -> None:
        policy = DenseQPolicy.load(self.fixture)
        with self.assertRaisesRegex(ValueError, "dimension"):
            policy.evaluate((1.0, 2.0))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            policy.evaluate((1.0, math.inf, 2.0))

    def test_argmax_tie_uses_first_action(self) -> None:
        policy = DenseQPolicy(
            "test-environment",
            [DenseLayer.from_values([[0.0], [0.0], [0.0]], [1.0, 1.0, 0.0])],
        )
        self.assertEqual(policy.action((42.0,)), 0)

    def test_malformed_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gdp"
            path.write_text("gravity-lab-dense-q-policy-v1\nenvironment x\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected end"):
                DenseQPolicy.load(path)


if __name__ == "__main__":
    unittest.main()
