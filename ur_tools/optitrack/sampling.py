"""Thread-safe pose buffering and stationary-window aggregation."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import transform_from_quaternion_translation


@dataclass(frozen=True)
class PoseObservation:
    received_monotonic_s: float
    source_time_s: float
    frame_id: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def transform(self) -> np.ndarray:
        return transform_from_quaternion_translation(
            self.quaternion_xyzw, self.position
        )


@dataclass(frozen=True)
class AveragedPose:
    transform: np.ndarray
    message_count: int
    translation_std_m: float
    rotation_std_deg: float
    source_time_s: float


class PoseBuffer:
    def __init__(self, maximum_size: int = 2000) -> None:
        self._items: deque[PoseObservation] = deque(maxlen=maximum_size)
        self._lock = Lock()

    def append(self, observation: PoseObservation) -> None:
        observation.transform()
        with self._lock:
            self._items.append(observation)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> tuple[PoseObservation, ...]:
        with self._lock:
            return tuple(self._items)


def average_observations(
    observations: Iterable[PoseObservation],
    *,
    expected_frame: str,
    minimum_messages: int,
    now_monotonic_s: float,
    maximum_age_s: float,
) -> AveragedPose:
    items = tuple(observations)
    if len(items) < minimum_messages:
        raise ValueError(
            f"received {len(items)} poses; at least {minimum_messages} are required"
        )
    if any(item.frame_id != expected_frame for item in items):
        frames = sorted({item.frame_id for item in items})
        raise ValueError(
            f"pose frame mismatch: expected {expected_frame!r}, received {frames}"
        )
    newest = max(item.received_monotonic_s for item in items)
    if now_monotonic_s - newest > maximum_age_s:
        raise ValueError("latest OptiTrack pose is stale")

    translations = np.asarray([item.position for item in items], dtype=float)
    rotations = Rotation.from_quat(
        np.asarray([item.quaternion_xyzw for item in items], dtype=float)
    )
    mean_translation = translations.mean(axis=0)
    mean_rotation = rotations.mean()
    translation_std = float(
        np.sqrt(np.mean(np.sum((translations - mean_translation) ** 2, axis=1)))
    )
    relative = mean_rotation.inv() * rotations
    rotation_std_deg = float(
        np.rad2deg(np.sqrt(np.mean(np.sum(relative.as_rotvec() ** 2, axis=1))))
    )
    transform = np.eye(4)
    transform[:3, :3] = mean_rotation.as_matrix()
    transform[:3, 3] = mean_translation
    return AveragedPose(
        transform=transform,
        message_count=len(items),
        translation_std_m=translation_std,
        rotation_std_deg=rotation_std_deg,
        source_time_s=float(np.mean([item.source_time_s for item in items])),
    )
