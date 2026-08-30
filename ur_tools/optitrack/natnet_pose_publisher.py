"""Publish Motive 2.0 NatNet rigid bodies as Crazyswarm2 NamedPoseArray."""

from __future__ import annotations

import argparse
from threading import Lock

import rclpy
from motion_capture_tracking_interfaces.msg import NamedPose, NamedPoseArray
from natnet import DataDescriptions, DataFrame, NatNetClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class NatNetPosePublisher(Node):
    def __init__(
        self,
        *,
        server_ip: str,
        local_ip: str,
        multicast_ip: str,
        topic: str,
        frame_id: str,
        target_name: str | None,
    ) -> None:
        super().__init__("ur_optitrack_natnet")
        self.frame_id = frame_id
        self.target_name = target_name
        self._name_lock = Lock()
        self._names_by_id: dict[int, str] = {}
        self.publisher = self.create_publisher(
            NamedPoseArray, topic, qos_profile_sensor_data
        )
        self.client = NatNetClient(
            server_ip_address=server_ip,
            local_ip_address=local_ip,
            multicast_address=multicast_ip,
            use_multicast=True,
        )
        self.client.on_data_description_received_event.handlers.append(
            self._description_callback
        )
        self.client.on_data_frame_received_event.handlers.append(self._frame_callback)
        self.client.connect()
        self.client.request_modeldef()
        self.client.run_async()
        self.get_logger().info(
            f"Connected to Motive {self.client.server_info.server_version} "
            f"(NatNet {self.client.protocol_version}) at {server_ip} via {local_ip}"
        )

    def _description_callback(self, descriptions: DataDescriptions) -> None:
        names = {
            item.id_num: item.name for item in descriptions.rigid_bodies if item.name
        }
        with self._name_lock:
            self._names_by_id = names
        self.get_logger().info(
            "Motive rigid bodies: "
            + ", ".join(
                f"{name} (id={id_num})" for id_num, name in sorted(names.items())
            )
        )

    def _frame_callback(self, frame: DataFrame) -> None:
        if frame.suffix.tracked_models_changed:
            self.client.request_modeldef()
        with self._name_lock:
            names = dict(self._names_by_id)
        message = NamedPoseArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        for rigid_body in frame.rigid_bodies:
            if rigid_body.tracking_valid is False:
                continue
            name = names.get(rigid_body.id_num)
            if name is None or (
                self.target_name is not None and name != self.target_name
            ):
                continue
            item = NamedPose()
            item.name = name
            item.pose.position.x = float(rigid_body.pos[0])
            item.pose.position.y = float(rigid_body.pos[1])
            item.pose.position.z = float(rigid_body.pos[2])
            item.pose.orientation.x = float(rigid_body.rot[0])
            item.pose.orientation.y = float(rigid_body.rot[1])
            item.pose.orientation.z = float(rigid_body.rot[2])
            item.pose.orientation.w = float(rigid_body.rot[3])
            message.poses.append(item)
        if message.poses and rclpy.ok():
            try:
                self.publisher.publish(message)
            except RuntimeError:
                pass

    def close(self) -> None:
        self.client.shutdown()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish Motive rigid bodies to a NamedPoseArray topic."
    )
    parser.add_argument("--server-ip", default="172.16.90.213")
    parser.add_argument("--local-ip", default="172.16.90.195")
    parser.add_argument("--multicast-ip", default="239.255.42.99")
    parser.add_argument("--topic", default="/poses")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--target-name", default="ur_calib")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rclpy.init()
    node = None
    try:
        node = NatNetPosePublisher(
            server_ip=args.server_ip,
            local_ip=args.local_ip,
            multicast_ip=args.multicast_ip,
            topic=args.topic,
            frame_id=args.frame_id,
            target_name=args.target_name or None,
        )
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
