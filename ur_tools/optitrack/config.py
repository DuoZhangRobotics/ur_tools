"""Configuration loading for gated OptiTrack/UR calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from ipaddress import ip_address
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a calibration profile is incomplete or unsafe."""


@dataclass(frozen=True)
class MotionConfig:
    home_joints: tuple[float, ...]
    poses: tuple[tuple[str, tuple[float, ...]], ...]
    joint_speed: float
    joint_acceleration: float
    tool_speed: float
    tool_acceleration: float


@dataclass(frozen=True)
class SamplingConfig:
    settle_s: float
    window_s: float
    minimum_messages: int
    maximum_translation_std_m: float
    maximum_rotation_std_deg: float
    maximum_retries: int
    initial_tracking_s: float


@dataclass(frozen=True)
class SolverConfig:
    holdout_every: int
    minimum_samples: int
    rotation_residual_scale_m_per_rad: float
    robust_loss_scale: float
    translation_rms_max_m: float
    translation_max_m: float
    rotation_rms_max_deg: float
    rotation_max_deg: float


@dataclass(frozen=True)
class CalibrationConfig:
    source_path: Path
    robot_ip: str
    motive_ip: str
    poses_topic: str
    target_name: str
    base_frame: str
    mocap_frame: str
    output_root: Path
    motion: MotionConfig
    sampling: SamplingConfig
    solver: SolverConfig


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a number") from exc
    if not isfinite(result):
        raise ConfigError(f"{field} must be finite")
    if positive and result <= 0:
        raise ConfigError(f"{field} must be positive")
    return result


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _vector(value: Any, field: str, size: int) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
    ):
        raise ConfigError(f"{field} must contain exactly {size} numbers")
    return tuple(_number(item, field) for item in value)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value.strip()


def load_config(path: str | Path) -> CalibrationConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"configuration file does not exist: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(raw, "configuration")
    if root.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")

    robot = _mapping(root.get("robot"), "robot")
    mocap = _mapping(root.get("motion_capture"), "motion_capture")
    frames = _mapping(root.get("frames"), "frames")
    motion = _mapping(root.get("motion"), "motion")
    sampling = _mapping(root.get("sampling"), "sampling")
    solver = _mapping(root.get("solver"), "solver")

    robot_ip = _nonempty_string(robot.get("ip"), "robot.ip")
    motive_ip = _nonempty_string(mocap.get("motive_ip"), "motion_capture.motive_ip")
    try:
        ip_address(robot_ip)
        ip_address(motive_ip)
    except ValueError as exc:
        raise ConfigError(
            f"robot and Motive addresses must be valid IP addresses: {exc}"
        ) from exc

    raw_poses = motion.get("poses")
    if not isinstance(raw_poses, list) or not raw_poses:
        raise ConfigError("motion.poses must be a non-empty list")
    poses: list[tuple[str, tuple[float, ...]]] = []
    names: set[str] = set()
    for index, raw_pose in enumerate(raw_poses):
        item = _mapping(raw_pose, f"motion.poses[{index}]")
        name = _nonempty_string(item.get("name"), f"motion.poses[{index}].name")
        if name in names:
            raise ConfigError(f"duplicate motion pose name: {name}")
        names.add(name)
        poses.append(
            (
                name,
                _vector(item.get("tcp_pose"), f"motion.poses[{index}].tcp_pose", 6),
            )
        )

    output_value = root.get("output_root", "calibration_results")
    output_path = Path(_nonempty_string(output_value, "output_root")).expanduser()
    if not output_path.is_absolute():
        output_path = (source.parent.parent / output_path).resolve()

    config = CalibrationConfig(
        source_path=source,
        robot_ip=robot_ip,
        motive_ip=motive_ip,
        poses_topic=_nonempty_string(
            mocap.get("poses_topic"), "motion_capture.poses_topic"
        ),
        target_name=_nonempty_string(
            mocap.get("target_name"), "motion_capture.target_name"
        ),
        base_frame=_nonempty_string(frames.get("base"), "frames.base"),
        mocap_frame=_nonempty_string(frames.get("mocap"), "frames.mocap"),
        output_root=output_path,
        motion=MotionConfig(
            home_joints=_vector(robot.get("home_joints"), "robot.home_joints", 6),
            poses=tuple(poses),
            joint_speed=_number(
                motion.get("joint_speed"), "motion.joint_speed", positive=True
            ),
            joint_acceleration=_number(
                motion.get("joint_acceleration"),
                "motion.joint_acceleration",
                positive=True,
            ),
            tool_speed=_number(
                motion.get("tool_speed"), "motion.tool_speed", positive=True
            ),
            tool_acceleration=_number(
                motion.get("tool_acceleration"),
                "motion.tool_acceleration",
                positive=True,
            ),
        ),
        sampling=SamplingConfig(
            settle_s=_number(
                sampling.get("settle_s"), "sampling.settle_s", positive=True
            ),
            window_s=_number(
                sampling.get("window_s"), "sampling.window_s", positive=True
            ),
            minimum_messages=_integer(
                sampling.get("minimum_messages"), "sampling.minimum_messages"
            ),
            maximum_translation_std_m=_number(
                sampling.get("maximum_translation_std_m"),
                "sampling.maximum_translation_std_m",
                positive=True,
            ),
            maximum_rotation_std_deg=_number(
                sampling.get("maximum_rotation_std_deg"),
                "sampling.maximum_rotation_std_deg",
                positive=True,
            ),
            maximum_retries=_integer(
                sampling.get("maximum_retries"), "sampling.maximum_retries"
            ),
            initial_tracking_s=_number(
                sampling.get("initial_tracking_s"),
                "sampling.initial_tracking_s",
                positive=True,
            ),
        ),
        solver=SolverConfig(
            holdout_every=_integer(
                solver.get("holdout_every"), "solver.holdout_every", minimum=2
            ),
            minimum_samples=_integer(
                solver.get("minimum_samples"), "solver.minimum_samples", minimum=6
            ),
            rotation_residual_scale_m_per_rad=_number(
                solver.get("rotation_residual_scale_m_per_rad"),
                "solver.rotation_residual_scale_m_per_rad",
                positive=True,
            ),
            robust_loss_scale=_number(
                solver.get("robust_loss_scale"),
                "solver.robust_loss_scale",
                positive=True,
            ),
            translation_rms_max_m=_number(
                solver.get("translation_rms_max_m"),
                "solver.translation_rms_max_m",
                positive=True,
            ),
            translation_max_m=_number(
                solver.get("translation_max_m"),
                "solver.translation_max_m",
                positive=True,
            ),
            rotation_rms_max_deg=_number(
                solver.get("rotation_rms_max_deg"),
                "solver.rotation_rms_max_deg",
                positive=True,
            ),
            rotation_max_deg=_number(
                solver.get("rotation_max_deg"),
                "solver.rotation_max_deg",
                positive=True,
            ),
        ),
    )
    if len(config.motion.poses) < config.solver.minimum_samples:
        raise ConfigError(
            f"profile has {len(config.motion.poses)} poses but solver.minimum_samples "
            f"is {config.solver.minimum_samples}"
        )
    if config.base_frame == config.mocap_frame:
        raise ConfigError("frames.base and frames.mocap must be different")
    if not config.poses_topic.startswith("/"):
        raise ConfigError("motion_capture.poses_topic must be absolute")
    return config
