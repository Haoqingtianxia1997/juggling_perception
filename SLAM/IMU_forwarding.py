#!/usr/bin/env python3
"""ROS2 IMU 转发节点。

以 200Hz 从 RobotClient 读取底层 IMU，并通过 sensor_msgs/msg/Imu 发布。
"""

import argparse
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from robot_bridge_py.robot_client import RobotClient


def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """四元数转旋转矩阵。"""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class IMUForwardingNode(Node):
    def __init__(
        self,
        publish_topic: str,
        stamp_source_topic: str,
        frame_id: str,
        publish_hz: float,
        gravity_mps2: float,
    ) -> None:
        super().__init__('imu_forwarding_node')

        self.publish_topic = publish_topic
        self.frame_id = frame_id
        self.publish_hz = max(1.0, float(publish_hz))
        self.gravity_mps2 = float(gravity_mps2)
        self.latest_stamp = None

        self.publisher = self.create_publisher(Imu, self.publish_topic, qos_profile_sensor_data)
        self.create_subscription(Imu, stamp_source_topic, self.stamp_cb, qos_profile_sensor_data)

        self.robot: Optional[RobotClient]
        try:
            self.robot = RobotClient(
                node=self,
                robot_type="H1",
                num_dof=20,
                control_frequency=self.publish_hz,
                interpolation_order=0.0,
            )
            self.get_logger().info("RobotClient initialized")
        except Exception as exc:
            self.robot = None
            self.get_logger().error(f"Failed to initialize RobotClient: {exc}")

        period = 1.0 / self.publish_hz
        self.timer = self.create_timer(period, self.publish_imu)
        self.get_logger().info(
            f"Publishing IMU on {self.publish_topic} at {self.publish_hz:.1f} Hz, frame_id={self.frame_id}"
        )

    def publish_imu(self) -> None:
        if self.robot is None:
            return

        try:
            self.robot.update_robot_state()
        except Exception as exc:
            self.get_logger().warn(f"Robot state update failed: {exc}")
            return

        quat = self.robot.quat
        if quat is None or len(quat) < 4:
            return

        angular_vel = self.robot.angular_velocity

        msg = Imu()
        if self.latest_stamp is not None:
            msg.header.stamp = self.latest_stamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        if angular_vel is not None and len(angular_vel) >= 3:
            msg.angular_velocity.x = float(angular_vel[0])
            msg.angular_velocity.y = float(angular_vel[1])
            msg.angular_velocity.z = float(angular_vel[2])
        else:
            msg.angular_velocity_covariance[0] = -1.0
        
        msg.orientation.w = 1.0
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        
        # 使用四元数计算重力方向，作为线加速度输出
        # robot.quat 格式: [w, x, y, z]
        qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        rot = quat_to_rot(qw, qx, qy, qz)
        gravity_world = np.array([0.0, 0.0, 1.0], dtype=np.float64) * self.gravity_mps2
        gravity_body = rot.T @ gravity_world
        msg.linear_acceleration.x = float(gravity_body[0])
        msg.linear_acceleration.y = float(gravity_body[1])
        msg.linear_acceleration.z = float(gravity_body[2])

        msg.linear_acceleration_covariance = [0.0] * 9

        self.publisher.publish(msg)

    def stamp_cb(self, msg: Imu) -> None:
        self.latest_stamp = msg.header.stamp


def main() -> None:
    parser = argparse.ArgumentParser(description='IMU forwarding node (190Hz)')
    parser.add_argument('--publish-topic', type=str, default='/robot/imu', help='IMU 输出话题')
    parser.add_argument('--stamp-topic', type=str, default='/livox/imu', help='时间戳来源话题')
    parser.add_argument('--frame-id', type=str, default='livox_frame', help='IMU frame_id')
    parser.add_argument('--publish-hz', type=float, default=100.0, help='IMU 发布频率 (Hz)')
    parser.add_argument('--gravity', type=float, default=1.0, help='重力大小 (默认 1.0 表示 g)')
    args = parser.parse_args()

    rclpy.init()
    node = IMUForwardingNode(
        publish_topic=args.publish_topic,
        stamp_source_topic=args.stamp_topic,
        frame_id=args.frame_id,
        publish_hz=args.publish_hz,
        gravity_mps2=args.gravity,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
