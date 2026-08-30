import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ur_tools.optitrack.collector import safe_stop_robot
from ur_tools.optitrack.sampling import (
    PoseBuffer,
    PoseObservation,
    average_observations,
)


def _observation(index: int, position_noise: float = 0.0) -> PoseObservation:
    rotation = Rotation.from_rotvec([0.1, -0.2, 0.3])
    position = np.array([0.4, -0.1, 0.7])
    position[0] += position_noise * (-1 if index % 2 else 1)
    return PoseObservation(
        received_monotonic_s=10.0 + index * 0.001,
        source_time_s=100.0 + index * 0.001,
        frame_id="world",
        position=tuple(position),
        quaternion_xyzw=tuple(rotation.as_quat()),
    )


def test_stationary_pose_window_is_averaged() -> None:
    result = average_observations(
        [_observation(index, 0.0005) for index in range(30)],
        expected_frame="world",
        minimum_messages=20,
        now_monotonic_s=10.04,
        maximum_age_s=0.1,
    )

    assert result.message_count == 30
    assert result.translation_std_m == pytest.approx(0.0005)
    assert result.rotation_std_deg < 1e-6
    assert result.transform[:3, 3] == pytest.approx([0.4, -0.1, 0.7])


def test_stale_or_wrong_frame_window_is_rejected() -> None:
    observations = [_observation(index) for index in range(20)]
    with pytest.raises(ValueError, match="stale"):
        average_observations(
            observations,
            expected_frame="world",
            minimum_messages=20,
            now_monotonic_s=20.0,
            maximum_age_s=0.1,
        )
    with pytest.raises(ValueError, match="frame mismatch"):
        average_observations(
            observations,
            expected_frame="mocap",
            minimum_messages=20,
            now_monotonic_s=10.02,
            maximum_age_s=0.1,
        )


def test_nan_and_zero_quaternion_never_enter_buffer() -> None:
    buffer = PoseBuffer()
    invalid = PoseObservation(
        received_monotonic_s=1.0,
        source_time_s=1.0,
        frame_id="world",
        position=(float("nan"), 0.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError):
        buffer.append(invalid)
    assert buffer.snapshot() == ()


def test_safe_stop_requests_linear_and_joint_stop() -> None:
    class FakeControl:
        def __init__(self) -> None:
            self.calls = []

        def stopL(self, acceleration):
            self.calls.append(("stopL", acceleration))

        def stopJ(self, acceleration):
            self.calls.append(("stopJ", acceleration))

    control = FakeControl()
    safe_stop_robot(control)

    assert control.calls == [("stopL", 1.0), ("stopJ", 1.0)]
