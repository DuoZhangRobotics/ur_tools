"""Load and publish the accepted OptiTrack-to-UR-base transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from geometry_msgs.msg import TransformStamped

from .transforms import quaternion_xyzw, validate_transform

DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[2] / "config" / "optitrack_to_ur_base.yaml"
)


@dataclass(frozen=True)
class StoredStaticTransform:
    parent_frame: str
    child_frame: str
    matrix: np.ndarray
    source_path: Path


def load_static_transform(path: str | Path) -> StoredStaticTransform:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("static transform file must use schema_version 1")
    if raw.get("accepted") is not True:
        raise ValueError("static transform file must contain an accepted calibration")
    frames = raw.get("frames")
    transforms = raw.get("transforms")
    if not isinstance(frames, dict) or not isinstance(transforms, dict):
        raise TypeError("static transform file must define frames and transforms")
    parent = frames.get("base")
    child = frames.get("mocap")
    if not isinstance(parent, str) or not parent:
        raise ValueError("frames.base must be a non-empty string")
    if not isinstance(child, str) or not child:
        raise ValueError("frames.mocap must be a non-empty string")
    if parent == child:
        raise ValueError("static transform parent and child frames must differ")
    matrix = validate_transform(
        np.asarray(transforms.get("base_from_mocap"), dtype=float),
        "base_from_mocap",
    )
    return StoredStaticTransform(parent, child, matrix, source)


def transform_message(stored: StoredStaticTransform, stamp) -> TransformStamped:
    quaternion = quaternion_xyzw(stored.matrix)
    message = TransformStamped()
    message.header.stamp = stamp
    message.header.frame_id = stored.parent_frame
    message.child_frame_id = stored.child_frame
    message.transform.translation.x = float(stored.matrix[0, 3])
    message.transform.translation.y = float(stored.matrix[1, 3])
    message.transform.translation.z = float(stored.matrix[2, 3])
    message.transform.rotation.x = float(quaternion[0])
    message.transform.rotation.y = float(quaternion[1])
    message.transform.rotation.z = float(quaternion[2])
    message.transform.rotation.w = float(quaternion[3])
    return message
