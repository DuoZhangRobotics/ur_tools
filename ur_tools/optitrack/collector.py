"""Live ROS/RTDE sample collection with explicit motion gating."""

from __future__ import annotations

import time
from contextlib import suppress
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np

from .config import CalibrationConfig
from .dataset import CalibrationDataset, CalibrationSample, save_dataset
from .sampling import PoseBuffer, PoseObservation, average_observations
from .transforms import pose_error, transform_from_pose_vector


class CollectionError(RuntimeError):
    pass


class RosPoseSource:
    """Own a small rclpy subscriber without importing ROS during offline use."""

    def __init__(self, topic: str, target_name: str) -> None:
        try:
            import rclpy
            from motion_capture_tracking_interfaces.msg import NamedPoseArray
            from rclpy.qos import qos_profile_sensor_data
        except ImportError as exc:
            raise CollectionError(
                "ROS 2 Python packages are unavailable; source ROS and activate the "
                "ur_tools uv environment"
            ) from exc

        self._rclpy = rclpy
        self._target_name = target_name
        self.buffer = PoseBuffer()
        self.invalid_messages = 0
        rclpy.init(args=None)
        self.node = rclpy.create_node("ur_optitrack_calibration_collector")
        self.node.create_subscription(
            NamedPoseArray, topic, self._callback, qos_profile_sensor_data
        )
        self._thread = Thread(
            target=rclpy.spin, args=(self.node,), daemon=True, name="rclpy-spin"
        )
        self._thread.start()

    def _callback(self, message: Any) -> None:
        source_time = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) / 1_000_000_000.0
        )
        for named_pose in message.poses:
            if named_pose.name != self._target_name:
                continue
            pose = named_pose.pose
            try:
                self.buffer.append(
                    PoseObservation(
                        received_monotonic_s=time.monotonic(),
                        source_time_s=source_time,
                        frame_id=message.header.frame_id,
                        position=(
                            float(pose.position.x),
                            float(pose.position.y),
                            float(pose.position.z),
                        ),
                        quaternion_xyzw=(
                            float(pose.orientation.x),
                            float(pose.orientation.y),
                            float(pose.orientation.z),
                            float(pose.orientation.w),
                        ),
                    )
                )
            except ValueError:
                self.invalid_messages += 1

    def close(self) -> None:
        with suppress(Exception):
            self.node.destroy_node()
        with suppress(Exception):
            self._rclpy.shutdown()
        self._thread.join(timeout=2.0)


def safe_stop_robot(control: Any) -> None:
    """Best-effort stop used for Ctrl+C and every exceptional exit."""
    with suppress(Exception):
        control.stopL(1.0)
    with suppress(Exception):
        control.stopJ(1.0)


def _disconnect(interface: Any) -> None:
    method = getattr(interface, "disconnect", None)
    if callable(method):
        with suppress(Exception):
            method()


def _robot_is_stationary(before: np.ndarray, after: np.ndarray) -> bool:
    translation, rotation = pose_error(before, after)
    return np.linalg.norm(translation) <= 0.0005 and np.linalg.norm(
        rotation
    ) <= np.deg2rad(0.1)


def _capture_stationary_pose(
    source: RosPoseSource, receive: Any, config: CalibrationConfig
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    settings = config.sampling
    for attempt in range(1, settings.maximum_retries + 1):
        source.buffer.clear()
        tcp_before = transform_from_pose_vector(receive.getActualTCPPose())
        time.sleep(settings.window_s)
        tcp_after = transform_from_pose_vector(receive.getActualTCPPose())
        try:
            averaged = average_observations(
                source.buffer.snapshot(),
                expected_frame=config.mocap_frame,
                minimum_messages=settings.minimum_messages,
                now_monotonic_s=time.monotonic(),
                maximum_age_s=max(0.1, settings.window_s),
            )
            if not _robot_is_stationary(tcp_before, tcp_after):
                raise ValueError("UR TCP moved during the sampling window")
            if averaged.translation_std_m > settings.maximum_translation_std_m:
                raise ValueError(
                    f"translation standard deviation {averaged.translation_std_m:.6f} m "
                    f"exceeds {settings.maximum_translation_std_m:.6f} m"
                )
            if averaged.rotation_std_deg > settings.maximum_rotation_std_deg:
                raise ValueError(
                    f"rotation standard deviation {averaged.rotation_std_deg:.3f} deg "
                    f"exceeds {settings.maximum_rotation_std_deg:.3f} deg"
                )
            return (
                tcp_after,
                averaged.transform,
                averaged.message_count,
                averaged.translation_std_m,
                averaged.rotation_std_deg,
                averaged.source_time_s,
            )
        except ValueError as exc:
            if attempt == settings.maximum_retries:
                raise CollectionError(
                    f"stationary pose capture failed after {attempt} attempts: {exc}"
                ) from exc
            print(f"Capture attempt {attempt} rejected: {exc}; retrying...")
    raise AssertionError("unreachable")


def _wait_for_tracking(source: RosPoseSource, config: CalibrationConfig) -> None:
    source.buffer.clear()
    time.sleep(config.sampling.initial_tracking_s)
    try:
        average_observations(
            source.buffer.snapshot(),
            expected_frame=config.mocap_frame,
            minimum_messages=config.sampling.minimum_messages,
            now_monotonic_s=time.monotonic(),
            maximum_age_s=max(0.1, config.sampling.window_s),
        )
    except ValueError as exc:
        raise CollectionError(
            f"rigid body {config.target_name!r} is not providing stable full poses: {exc}"
        ) from exc


def collect_dataset(
    config: CalibrationConfig, dataset_path: str | Path
) -> CalibrationDataset:
    """Move through the reviewed poses and atomically append every sample."""
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise CollectionError(
            "ur_rtde is unavailable in the active uv environment"
        ) from exc

    dataset = CalibrationDataset(
        robot_ip=config.robot_ip,
        base_frame=config.base_frame,
        mocap_frame=config.mocap_frame,
        target_name=config.target_name,
    )
    save_dataset(dataset_path, dataset)
    source = RosPoseSource(config.poses_topic, config.target_name)
    control = None
    receive = None
    try:
        print(
            f"Waiting {config.sampling.initial_tracking_s:.1f} s for "
            f"{config.target_name!r} on {config.poses_topic}..."
        )
        _wait_for_tracking(source, config)
        control = RTDEControlInterface(config.robot_ip)
        receive = RTDEReceiveInterface(config.robot_ip, use_upper_range_registers=False)
        print("Moving to reviewed joint home...")
        if (
            control.moveJ(
                list(config.motion.home_joints),
                config.motion.joint_speed,
                config.motion.joint_acceleration,
            )
            is False
        ):
            raise CollectionError("moveJ to calibration home failed")

        for index, (name, pose) in enumerate(config.motion.poses, start=1):
            print(f"[{index}/{len(config.motion.poses)}] Moving to {name}: {pose}")
            if (
                control.moveL(
                    list(pose),
                    config.motion.tool_speed,
                    config.motion.tool_acceleration,
                )
                is False
            ):
                raise CollectionError(f"moveL failed for {name}")
            time.sleep(config.sampling.settle_s)
            (
                base_from_tcp,
                mocap_from_target,
                count,
                translation_std,
                rotation_std,
                source_time,
            ) = _capture_stationary_pose(source, receive, config)
            dataset.samples.append(
                CalibrationSample(
                    name=name,
                    captured_at_unix_s=source_time,
                    base_from_tcp=base_from_tcp,
                    mocap_from_target=mocap_from_target,
                    message_count=count,
                    translation_std_m=translation_std,
                    rotation_std_deg=rotation_std,
                )
            )
            save_dataset(dataset_path, dataset)
            print(
                f"Captured {name}: {count} poses, "
                f"translation std={translation_std * 1000:.2f} mm, "
                f"rotation std={rotation_std:.3f} deg"
            )
        dataset.complete = True
        save_dataset(dataset_path, dataset)
        return dataset
    except BaseException:
        if control is not None:
            safe_stop_robot(control)
        save_dataset(dataset_path, dataset)
        raise
    finally:
        source.close()
        if control is not None:
            _disconnect(control)
        if receive is not None:
            _disconnect(receive)
