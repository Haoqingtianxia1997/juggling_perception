#!/usr/bin/env python3
"""实时订阅 `world_model_prediction` 并绘制 3D 轨迹。

默认假设话题类型为 `std_msgs/msg/Float32MultiArray`：
- data[0:3] -> [x, y, z]
- data[3:6] -> [vx, vy, vz]（若存在）

用途：快速检查 world_model_prediction 是否稳定发布，以及位置/速度在三维空间中的变化。
"""

from __future__ import annotations

import argparse
from collections import deque
import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray


class WorldModel3DViewer(Node):
    def __init__(self, topic: str, history: int = 1000):
        super().__init__('world_model_3d_viewer')
        self.topic = topic
        self.positions = deque(maxlen=max(100, int(history)))
        self.velocities = deque(maxlen=max(100, int(history)))
        self.timestamps = deque(maxlen=max(100, int(history)))

        self.latest_pos = None
        self.latest_vel = None
        self.last_msg_time = None
        self.msg_count = 0

        self.create_subscription(
            Float32MultiArray,
            self.topic,
            self.callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f'Subscribed topic: {self.topic}')

    def callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) < 3:
            self.get_logger().warn(
                f'world_model_prediction 数据长度不足，至少需要3个数，实际={len(data)}'
            )
            return

        pos = np.array(data[:3], dtype=np.float64)
        vel = None
        if len(data) >= 6:
            vel = np.array(data[3:6], dtype=np.float64)

        now = time.time()
        if self.last_msg_time is None:
            rel_t = 0.0
        else:
            rel_t = now - self.last_msg_time
        self.last_msg_time = now

        self.positions.append(pos)
        self.velocities.append(vel)
        self.timestamps.append(now)
        self.latest_pos = pos
        self.latest_vel = vel
        self.msg_count += 1

        if self.msg_count % 20 == 0:
            if vel is not None:
                speed = float(np.linalg.norm(vel))
                self.get_logger().info(
                    f'[{self.msg_count}] pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], '
                    f'vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}], speed={speed:.3f} m/s, '
                    f'dt={rel_t*1000.0:.1f} ms'
                )
            else:
                self.get_logger().info(
                    f'[{self.msg_count}] pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], dt={rel_t*1000.0:.1f} ms'
                )


def set_equal_axes(ax, points: np.ndarray, margin: float = 0.2):
    if points.shape[0] == 0:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        return

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 0.2)
    half = 0.5 * span * (1.0 + margin)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def draw_world_frame(ax, origin=np.zeros(3), scale: float = 0.2):
    ox, oy, oz = origin
    ax.quiver(ox, oy, oz, scale, 0, 0, color='r', linewidth=2)
    ax.quiver(ox, oy, oz, 0, scale, 0, color='g', linewidth=2)
    ax.quiver(ox, oy, oz, 0, 0, scale, color='b', linewidth=2)


def draw_velocity_arrow(ax, pos: np.ndarray, vel: np.ndarray, scale: float = 0.25):
    speed = float(np.linalg.norm(vel))
    if speed < 1e-6:
        return

    direction = vel / speed
    length = max(speed * scale, 0.05)
    ax.quiver(
        pos[0], pos[1], pos[2],
        direction[0] * length, direction[1] * length, direction[2] * length,
        color='tab:orange', linewidth=2.5, arrow_length_ratio=0.18
    )


def main():
    parser = argparse.ArgumentParser(description='实时订阅 world_model_prediction 并绘制 3D 图')
    parser.add_argument('--topic', type=str, default='world_model_prediction', help='订阅话题名')
    parser.add_argument('--history', type=int, default=1000, help='轨迹缓存长度')
    parser.add_argument('--refresh-hz', type=float, default=20.0, help='图像刷新频率')
    parser.add_argument('--show-velocity', action='store_true', help='如果消息包含速度，则绘制速度箭头')
    args = parser.parse_args()

    rclpy.init()
    node = WorldModel3DViewer(topic=args.topic, history=args.history)

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f'world_model_prediction 3D Viewer | topic={args.topic} | q/ESC to quit')

    period = 1.0 / max(1.0, float(args.refresh_hz))

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            ax.cla()
            ax.set_title('Realtime 3D world_model_prediction')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.grid(True)

            pts = np.asarray(node.positions, dtype=np.float64)
            if pts.shape[0] > 0:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='tab:blue', linewidth=1.8, label='trajectory')
                ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color='tab:red', s=70, edgecolors='k', label='current')

            if args.show_velocity and node.latest_pos is not None and node.latest_vel is not None:
                draw_velocity_arrow(ax, node.latest_pos, node.latest_vel)

            draw_world_frame(ax, origin=np.zeros(3), scale=0.25)

            if pts.shape[0] > 0:
                pts_for_axes = np.vstack([pts, np.zeros((1, 3), dtype=np.float64)])
            else:
                pts_for_axes = np.zeros((1, 3), dtype=np.float64)
            set_equal_axes(ax, pts_for_axes, margin=0.3)

            ax.text2D(
                0.02,
                0.98,
                f'samples={pts.shape[0]}\n'
                f'messages={node.msg_count}\n'
                f'status={"receiving" if node.latest_pos is not None else "waiting"}',
                transform=ax.transAxes,
                va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
            )

            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(loc='upper right')
            plt.pause(period)

            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()
