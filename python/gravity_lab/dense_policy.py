from __future__ import annotations

import math
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

POLICY_FORMAT = "gravity-lab-dense-q-policy-v1"
_ACTIVATIONS = {"relu", "tanh", "linear"}
_MAXIMUM_LAYERS = 32
_MAXIMUM_DIMENSION = 65_536
_MAXIMUM_PARAMETERS = 100_000_000


def _plain(value: object) -> object:
    """Convert common tensor/array objects without importing their frameworks."""
    if hasattr(value, "detach"):
        value = value.detach()  # type: ignore[union-attr]
    if hasattr(value, "cpu"):
        value = value.cpu()  # type: ignore[union-attr]
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    return value


def _vector(values: Iterable[float], field: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


@dataclass(frozen=True)
class DenseLayer:
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    activation: str = "linear"

    @classmethod
    def from_values(
        cls,
        weights: object,
        bias: object,
        activation: str = "linear",
    ) -> DenseLayer:
        raw_weights = _plain(weights)
        raw_bias = _plain(bias)
        if not isinstance(raw_weights, Sequence) or not isinstance(raw_bias, Sequence):
            raise ValueError("dense layer weights and bias must be sequences")
        rows = tuple(_vector(_plain(row), "weights") for row in raw_weights)  # type: ignore[arg-type]
        return cls(rows, _vector(raw_bias, "bias"), activation)


class DenseQPolicy:
    """Dependency-free evaluator/exporter for sequential dense Q-networks."""

    def __init__(
        self,
        environment_id: str,
        layers: Sequence[DenseLayer],
        input_scale: Sequence[float] | None = None,
        input_bias: Sequence[float] | None = None,
    ) -> None:
        if not environment_id:
            raise ValueError("environment_id must not be empty")
        if not 0 < len(layers) <= _MAXIMUM_LAYERS:
            raise ValueError("dense policy must contain between 1 and 32 layers")
        first_input = len(layers[0].weights[0]) if layers[0].weights else 0
        if not 0 < first_input <= _MAXIMUM_DIMENSION:
            raise ValueError("dense-policy input size is invalid")
        self.environment_id = environment_id
        self.layers = tuple(layers)
        self.input_scale = _vector(input_scale if input_scale is not None else [1.0] * first_input, "input_scale")
        self.input_bias = _vector(input_bias if input_bias is not None else [0.0] * first_input, "input_bias")
        if len(self.input_scale) != first_input or len(self.input_bias) != first_input:
            raise ValueError("input normalization dimension mismatch")

        expected_input = first_input
        total_parameters = 0
        for layer in self.layers:
            if layer.activation not in _ACTIVATIONS:
                raise ValueError(f"unsupported dense-policy activation: {layer.activation}")
            if not layer.weights or len(layer.bias) != len(layer.weights):
                raise ValueError("dense-policy output and bias dimensions differ")
            if len(layer.weights) > _MAXIMUM_DIMENSION:
                raise ValueError("dense-policy output size is invalid")
            if any(len(row) != expected_input for row in layer.weights):
                raise ValueError("dense-policy layers do not connect")
            total_parameters += expected_input * len(layer.weights)
            if total_parameters > _MAXIMUM_PARAMETERS:
                raise ValueError("dense policy has too many parameters")
            if any(not math.isfinite(value) for row in layer.weights for value in row):
                raise ValueError("weights contains a non-finite value")
            if any(not math.isfinite(value) for value in layer.bias):
                raise ValueError("bias contains a non-finite value")
            expected_input = len(layer.weights)

    @property
    def observation_size(self) -> int:
        return len(self.input_scale)

    @property
    def action_count(self) -> int:
        return len(self.layers[-1].bias)

    def evaluate(self, observation: Sequence[float]) -> tuple[float, ...]:
        if len(observation) != self.observation_size:
            raise ValueError("policy observation dimension mismatch")
        values = [float(value) * scale + bias for value, scale, bias in zip(
            observation, self.input_scale, self.input_bias
        )]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("policy observation contains a non-finite value")
        for layer in self.layers:
            next_values = [
                layer.bias[output] + sum(weight * value for weight, value in zip(row, values))
                for output, row in enumerate(layer.weights)
            ]
            if layer.activation == "relu":
                next_values = [max(0.0, value) for value in next_values]
            elif layer.activation == "tanh":
                next_values = [math.tanh(value) for value in next_values]
            if any(not math.isfinite(value) for value in next_values):
                raise RuntimeError("dense-policy inference produced a non-finite value")
            values = next_values
        return tuple(values)

    def action(self, observation: Sequence[float]) -> int:
        values = self.evaluate(observation)
        return max(range(len(values)), key=values.__getitem__)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        quote = lambda value: '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines = [
            POLICY_FORMAT,
            f"environment {quote(self.environment_id)}",
            f"input_size {self.observation_size}",
            "input_scale " + " ".join(f"{value:.17g}" for value in self.input_scale),
            "input_bias " + " ".join(f"{value:.17g}" for value in self.input_bias),
            f"layers {len(self.layers)}",
        ]
        for layer in self.layers:
            lines.append(f"layer {len(layer.weights[0])} {len(layer.weights)} {layer.activation}")
            lines.append("bias " + " ".join(f"{value:.17g}" for value in layer.bias))
            lines.append("weights " + " ".join(
                f"{value:.17g}" for row in layer.weights for value in row
            ))
        lines.append("end")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> DenseQPolicy:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        position = 0

        def tokens(label: str | None = None) -> list[str]:
            nonlocal position
            if position >= len(lines):
                raise ValueError("unexpected end of dense policy")
            try:
                result = shlex.split(lines[position])
            except ValueError as error:
                raise ValueError("malformed dense policy") from error
            position += 1
            if label is not None and (not result or result[0] != label):
                raise ValueError(f"malformed dense policy: expected {label}")
            return result

        if tokens() != [POLICY_FORMAT]:
            raise ValueError("unsupported dense-policy format")
        environment = tokens("environment")
        if len(environment) != 2:
            raise ValueError("malformed dense-policy environment")
        input_size_line = tokens("input_size")
        if len(input_size_line) != 2:
            raise ValueError("malformed dense-policy input size")
        input_size = int(input_size_line[1])
        if not 0 < input_size <= _MAXIMUM_DIMENSION:
            raise ValueError("invalid dense-policy input size")

        scale_line = tokens("input_scale")
        bias_line = tokens("input_bias")
        if len(scale_line) != input_size + 1 or len(bias_line) != input_size + 1:
            raise ValueError("malformed dense-policy input normalization")
        input_scale = _vector(map(float, scale_line[1:]), "input_scale")
        input_bias = _vector(map(float, bias_line[1:]), "input_bias")
        layers_line = tokens("layers")
        if len(layers_line) != 2 or not 0 < int(layers_line[1]) <= _MAXIMUM_LAYERS:
            raise ValueError("invalid dense-policy layer count")

        layers: list[DenseLayer] = []
        total_parameters = 0
        for _ in range(int(layers_line[1])):
            layer_line = tokens("layer")
            if len(layer_line) != 4:
                raise ValueError("malformed dense-policy layer")
            inputs, outputs = int(layer_line[1]), int(layer_line[2])
            if not 0 < inputs <= _MAXIMUM_DIMENSION or not 0 < outputs <= _MAXIMUM_DIMENSION:
                raise ValueError("invalid dense-policy layer dimensions")
            total_parameters += inputs * outputs
            if total_parameters > _MAXIMUM_PARAMETERS:
                raise ValueError("dense policy has too many parameters")
            bias_values = tokens("bias")
            weight_values = tokens("weights")
            if len(bias_values) != outputs + 1 or len(weight_values) != inputs * outputs + 1:
                raise ValueError("dense-policy parameter dimensions do not match layer dimensions")
            flat = _vector(map(float, weight_values[1:]), "weights")
            rows = tuple(tuple(flat[offset:offset + inputs]) for offset in range(0, len(flat), inputs))
            layers.append(DenseLayer(rows, _vector(map(float, bias_values[1:]), "bias"), layer_line[3]))
        if tokens() != ["end"] or position != len(lines):
            raise ValueError("unexpected data after dense-policy end marker")
        return cls(environment[1], layers, input_scale, input_bias)
