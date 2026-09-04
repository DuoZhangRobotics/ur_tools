"""Runtime coordinate conversion using the accepted OptiTrack calibration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .transforms import invert, validate_transform

DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[2] / "config" / "optitrack_to_ur_base.yaml"
)


@dataclass(frozen=True)
class WorldBaseTransform:
    """Convert states and goals between Motive world and UR base."""

    base_from_world: np.ndarray
    world_from_base: np.ndarray
    base_frame: str
    world_frame: str
    source_path: Path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIBRATION) -> WorldBaseTransform:
        source = Path(path).expanduser().resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("frame transform file must use schema_version 1")
        if raw.get("accepted") is not True:
            raise ValueError("frame transform requires an accepted calibration")
        frames = raw.get("frames")
        transforms = raw.get("transforms")
        if not isinstance(frames, dict) or not isinstance(transforms, dict):
            raise TypeError("frame transform file must define frames and transforms")
        base_frame = frames.get("base")
        world_frame = frames.get("mocap")
        if not isinstance(base_frame, str) or not base_frame:
            raise ValueError("frames.base must be a non-empty string")
        if not isinstance(world_frame, str) or not world_frame:
            raise ValueError("frames.mocap must be a non-empty string")
        if base_frame == world_frame:
            raise ValueError("base and world frames must differ")
        base_from_world = validate_transform(
            np.asarray(transforms.get("base_from_mocap"), dtype=float),
            "base_from_mocap",
        )
        return cls(
            base_from_world=base_from_world,
            world_from_base=invert(base_from_world),
            base_frame=base_frame,
            world_frame=world_frame,
            source_path=source,
        )

    @staticmethod
    def _point(value: Sequence[float], name: str) -> np.ndarray:
        point = np.asarray(value, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{name} must contain three finite values")
        return point

    @staticmethod
    def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        return transform[:3, :3] @ point + transform[:3, 3]

    @staticmethod
    def _transform_vector(transform: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return transform[:3, :3] @ vector

    def world_to_base(self, point_world: Sequence[float]) -> np.ndarray:
        return self._transform_point(
            self.base_from_world, self._point(point_world, "point_world")
        )

    def base_to_world(self, point_base: Sequence[float]) -> np.ndarray:
        return self._transform_point(
            self.world_from_base, self._point(point_base, "point_base")
        )

    def world_vector_to_base(self, vector_world: Sequence[float]) -> np.ndarray:
        return self._transform_vector(
            self.base_from_world, self._point(vector_world, "vector_world")
        )

    def base_vector_to_world(self, vector_base: Sequence[float]) -> np.ndarray:
        return self._transform_vector(
            self.world_from_base, self._point(vector_base, "vector_base")
        )

    def world_pose_to_base(self, world_from_object: np.ndarray) -> np.ndarray:
        return self.base_from_world @ validate_transform(
            world_from_object, "world_from_object"
        )

    def base_pose_to_world(self, base_from_object: np.ndarray) -> np.ndarray:
        return self.world_from_base @ validate_transform(
            base_from_object, "base_from_object"
        )
