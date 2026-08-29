from __future__ import annotations

import ctypes
import enum
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

OBSERVATION_SIZE = 12
ACTION_COUNT = 9


class Action(enum.IntEnum):
    COAST = 0
    THROTTLE = 1
    BRAKE = 2
    LEAN_BACK = 3
    LEAN_FORWARD = 4
    THROTTLE_LEAN_BACK = 5
    THROTTLE_LEAN_FORWARD = 6
    BRAKE_LEAN_BACK = 7
    BRAKE_LEAN_FORWARD = 8


@dataclass(frozen=True)
class Config:
    time_step: float = 1.0 / 120.0
    frame_skip: int = 4
    max_episode_steps: int = 3_000
    seed: int = 1


@dataclass(frozen=True)
class StepResult:
    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    finished: bool
    crashed: bool


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("time_step", ctypes.c_double),
        ("frame_skip", ctypes.c_uint32),
        ("max_episode_steps", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
    ]


class _CStepResult(ctypes.Structure):
    _fields_ = [
        ("observation", ctypes.c_double * OBSERVATION_SIZE),
        ("reward", ctypes.c_double),
        ("terminated", ctypes.c_int),
        ("truncated", ctypes.c_int),
        ("finished", ctypes.c_int),
        ("crashed", ctypes.c_int),
    ]


def _candidate_libraries() -> Sequence[Path]:
    override = os.environ.get("GRAVITY_LAB_LIBRARY")
    suffix = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
    name = ("gravity_lab" if platform.system() == "Windows" else "libgravity_lab") + suffix
    root = Path(__file__).resolve().parents[2]
    candidates = []
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            root / "build" / name,
            root / "build" / "lib" / name,
            root / "build" / "Debug" / name,
            root / "build" / "Release" / name,
            Path.cwd() / name,
        ]
    )
    return candidates


def _load_library() -> ctypes.CDLL:
    candidates = _candidate_libraries()
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        checked = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Gravity Lab native library was not found. Build it with "
            f"'cmake -S . -B build && cmake --build build'. Checked:\n  {checked}"
        )
    library = ctypes.CDLL(str(found))
    library.gd_create.argtypes = [ctypes.c_char_p, _CConfig]
    library.gd_create.restype = ctypes.c_void_p
    library.gd_destroy.argtypes = [ctypes.c_void_p]
    library.gd_destroy.restype = None
    library.gd_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_double)]
    library.gd_reset.restype = ctypes.c_int
    library.gd_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_CStepResult)]
    library.gd_step.restype = ctypes.c_int
    library.gd_last_error.argtypes = []
    library.gd_last_error.restype = ctypes.c_char_p
    return library


class GravityEnv:
    """A small Gym-like API with no dependency on Gym or an RL framework."""

    observation_size = OBSERVATION_SIZE
    action_count = ACTION_COUNT

    def __init__(self, map_path: str | Path, config: Config = Config()) -> None:
        self._library = _load_library()
        self._handle = self._library.gd_create(
            os.fsencode(Path(map_path).resolve()),
            _CConfig(config.time_step, config.frame_skip, config.max_episode_steps, config.seed),
        )
        if not self._handle:
            self._raise_native_error()
        self.config = config

    def _raise_native_error(self) -> None:
        message = self._library.gd_last_error()
        raise RuntimeError(message.decode("utf-8") if message else "unknown native error")

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        actual_seed = self.config.seed if seed is None else seed
        output = (ctypes.c_double * OBSERVATION_SIZE)()
        if self._library.gd_reset(self._handle, actual_seed, output) != 0:
            self._raise_native_error()
        return tuple(output)

    def step(self, action: int | Action) -> StepResult:
        output = _CStepResult()
        if self._library.gd_step(self._handle, int(action), ctypes.byref(output)) != 0:
            self._raise_native_error()
        return StepResult(
            tuple(output.observation),
            output.reward,
            bool(output.terminated),
            bool(output.truncated),
            bool(output.finished),
            bool(output.crashed),
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.gd_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> GravityEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
