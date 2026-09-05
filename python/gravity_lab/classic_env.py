from __future__ import annotations

import ctypes
import enum
import os
import platform
from dataclasses import dataclass
from pathlib import Path

CLASSIC_OBSERVATION_SIZE = 134
CLASSIC_ACTION_COUNT = 9


class ClassicAction(enum.IntEnum):
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
class ClassicConfig:
    level_group: int = 0
    track: int = 0
    league: int = 0
    frame_skip: int = 2
    max_episode_steps: int = 5_000
    seed: int = 1
    obstacle_ray_count: int = 8


@dataclass(frozen=True)
class ClassicStepResult:
    observation: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    finished: bool
    crashed: bool
    wheelie_finish: bool
    physics_code: int


class _CClassicConfig(ctypes.Structure):
    _fields_ = [
        ("level_group", ctypes.c_uint32),
        ("track", ctypes.c_uint32),
        ("league", ctypes.c_uint32),
        ("frame_skip", ctypes.c_uint32),
        ("max_episode_steps", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("obstacle_ray_count", ctypes.c_uint32),
    ]


class _CClassicStepResult(ctypes.Structure):
    _fields_ = [
        ("observation", ctypes.c_double * CLASSIC_OBSERVATION_SIZE),
        ("reward", ctypes.c_double),
        ("terminated", ctypes.c_int),
        ("truncated", ctypes.c_int),
        ("finished", ctypes.c_int),
        ("crashed", ctypes.c_int),
        ("wheelie_finish", ctypes.c_int),
        ("physics_code", ctypes.c_int),
    ]


def _library_candidates() -> list[Path]:
    suffix = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
    name = "gravity_lab_classic" + suffix
    root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []
    override = os.environ.get("GRAVITY_LAB_CLASSIC_LIBRARY")
    if override:
        candidates.append(Path(override))
    for directory in ("build-classic-rl", "build", "build-classic", "build-integrated"):
        candidates.extend(
            [root / directory / name, root / directory / "lib" / name,
             root / directory / "Debug" / name, root / directory / "Release" / name]
        )
    candidates.append(Path.cwd() / name)
    return candidates


def classic_library_available() -> bool:
    return any(path.is_file() for path in _library_candidates())


def _load_library() -> ctypes.CDLL:
    candidates = _library_candidates()
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        checked = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Classic RL library was not found. Configure with "
            "'-DGRAVITY_LAB_BUILD_CLASSIC=ON' and build. Checked:\n  " + checked
        )
    library = ctypes.CDLL(str(found))
    library.gdc_create.argtypes = [ctypes.c_char_p, _CClassicConfig]
    library.gdc_create.restype = ctypes.c_void_p
    library.gdc_destroy.argtypes = [ctypes.c_void_p]
    library.gdc_destroy.restype = None
    library.gdc_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_double)]
    library.gdc_reset.restype = ctypes.c_int
    library.gdc_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_CClassicStepResult)]
    library.gdc_step.restype = ctypes.c_int
    library.gdc_track_name.argtypes = [ctypes.c_void_p]
    library.gdc_track_name.restype = ctypes.c_char_p
    library.gdc_track_count.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    library.gdc_track_count.restype = ctypes.c_int
    library.gdc_last_error.argtypes = []
    library.gdc_last_error.restype = ctypes.c_char_p
    return library


class ClassicGravityEnv:
    """Headless API over the vendored classic fixed-point physics."""

    observation_size = CLASSIC_OBSERVATION_SIZE
    action_count = CLASSIC_ACTION_COUNT
    environment_id = "gravity-lab-classic-v1"

    def __init__(
        self,
        config: ClassicConfig = ClassicConfig(),
        level_pack: str | Path | None = None,
    ) -> None:
        if not 0 <= config.level_group < 3:
            raise ValueError("level_group must be in [0, 2]")
        if config.track < 0:
            raise ValueError("track must be nonnegative")
        if not 0 <= config.league < 4:
            raise ValueError("league must be in [0, 3]")
        if not 1 <= config.frame_skip <= 100:
            raise ValueError("frame_skip must be in [1, 100]")
        if config.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not 0 <= config.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if not 1 <= config.obstacle_ray_count <= 32:
            raise ValueError("obstacle_ray_count must be in [1, 32]")
        self._library = _load_library()
        path = None if level_pack is None else os.fsencode(Path(level_pack).resolve())
        self._handle = self._library.gdc_create(
            path,
            _CClassicConfig(
                config.level_group,
                config.track,
                config.league,
                config.frame_skip,
                config.max_episode_steps,
                config.seed,
                config.obstacle_ray_count,
            ),
        )
        if not self._handle:
            self._raise_native_error()
        self.config = config

    def _raise_native_error(self) -> None:
        message = self._library.gdc_last_error()
        raise RuntimeError(message.decode("utf-8") if message else "unknown classic native error")

    @property
    def track_name(self) -> str:
        result = self._library.gdc_track_name(self._handle)
        if not result:
            self._raise_native_error()
        return result.decode("utf-8")

    def track_count(self, level_group: int) -> int:
        result = ctypes.c_uint32()
        if self._library.gdc_track_count(self._handle, level_group, ctypes.byref(result)) != 0:
            self._raise_native_error()
        return result.value

    def reset(self, seed: int | None = None) -> tuple[float, ...]:
        actual_seed = self.config.seed if seed is None else seed
        if not 0 <= actual_seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        output = (ctypes.c_double * CLASSIC_OBSERVATION_SIZE)()
        if self._library.gdc_reset(self._handle, actual_seed, output) != 0:
            self._raise_native_error()
        return tuple(output)

    def step(self, action: int | ClassicAction) -> ClassicStepResult:
        output = _CClassicStepResult()
        if self._library.gdc_step(self._handle, int(action), ctypes.byref(output)) != 0:
            self._raise_native_error()
        return ClassicStepResult(
            tuple(output.observation), output.reward, bool(output.terminated), bool(output.truncated),
            bool(output.finished), bool(output.crashed), bool(output.wheelie_finish), output.physics_code,
        )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._library.gdc_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> ClassicGravityEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
