#!/usr/bin/env python3
"""ROS2实时3D位姿可视化（Matplotlib）。

需求：动态画出 odom（世界坐标系）以及当前位姿的 frame。
默认订阅话题: /odometry/imu
"""

import argparse
from collections import deque
import csv
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from robot_bridge_py.robot_client import RobotClient

def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class Odom3DVisualizer(Node):
    def __init__(
        self,
        topic: str,
        history: int,
        coord_frame: str = 'lidar',
        lidar_in_body_pos=None,
        lidar_in_body_rot=None,
        save_dir: Path = None,
    ):
        super().__init__('slam_pose_3d_visualizer')
        self.topic = topic
        self.coord_frame = str(coord_frame).lower().strip()
        if self.coord_frame not in ('lidar', 'body'):
            self.get_logger().warn(f"Invalid coord_frame={coord_frame}, fallback to 'lidar'")
            self.coord_frame = 'lidar'

        # lidar→body 外参：T_body_lidar = [R_body_lidar, t_body_lidar]
        # 配置含义：lidar 在 body 坐标系下的位姿
        self.lidar_in_body_pos = np.asarray(
            lidar_in_body_pos if lidar_in_body_pos is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        ).reshape(3)
        self.lidar_in_body_rot = np.asarray(
            lidar_in_body_rot if lidar_in_body_rot is not None else np.eye(3).reshape(-1),
            dtype=np.float64,
        ).reshape(3, 3)

        self.positions = deque(maxlen=max(100, int(history)))
        self.latest_pose = None  # (pos[3], rot[3,3])
        self.latest_lidar_pose = None  # 原始 lidar 位姿 (pos[3], rot[3,3])

        # R_diff 递推均值
        self.rdiff_count = 0
        self.rdiff_mean = np.eye(3, dtype=np.float64)
        
        # 分别记录x,y,z坐标
        self.x_coords = deque(maxlen=max(100, int(history)))
        self.y_coords = deque(maxlen=max(100, int(history)))
        self.z_coords = deque(maxlen=max(100, int(history)))
        self.timestamps = deque(maxlen=max(100, int(history)))
        
        # 轨迹保存：固定保存到 SLAM_trajectory 目录
        self.save_dir = Path(save_dir) if save_dir else Path(__file__).resolve().parent / 'SLAM_trajectory'
        self.trajectory_file = None
        self.start_time = None
        self.buffer_size = 100  # 缓冲区大小
        self.trajectory_buffer = []  # 内存缓冲区
        self.buffer_count = 0  # 缓冲区记录数
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.trajectory_file = self.save_dir / f'trajectory_{self.coord_frame}_{timestamp}.csv'
        with open(self.trajectory_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time(s)', 'x(m)', 'y(m)', 'z(m)'])
        self.get_logger().info(f"Saving trajectory to: {self.trajectory_file}")
        if self.coord_frame == 'body':
            self.get_logger().info(
                f"Using lidar→body extrinsics: position={self.lidar_in_body_pos.tolist()}, "
                f"rotation={self.lidar_in_body_rot.tolist()}"
            )

        self.create_subscription(Odometry, self.topic, self.odom_cb, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribed topic: {self.topic}")
        
        
        self.robot = RobotClient(
                    node=self,
                    robot_type="H1",
                    num_dof=20,
                    control_frequency=50.0,
                    interpolation_order=0.0
                )
                # 创建定时器定期更新 IMU 数据（500Hz）
        self.imu_timer = self.create_timer(0.002, self.update_imu_data)
        self.robot_quat = None
    
    def update_imu_data(self):
        """定期从 RobotClient 获取底层 IMU 数据。记录系统时间戳。"""
        if self.robot is None:
            return  
        try:
            # 记录获取数据的时间戳（系统时间）
            timestamp = time.time()
            
            # 更新机器人状态
            self.robot.update_robot_state()
            
            # 获取四元数 (w, x, y, z)
            self.robot_quat = self.robot.quat
        except Exception as e:
            self.get_logger().debug(f"Error updating IMU data: {e}")
            
    def _lidar_pose_to_selected_pose(self, lidar_pos: np.ndarray, lidar_rot: np.ndarray):
        """根据 coord_frame 选择是返回 lidar 位姿还是 body 位姿。"""
        if self.coord_frame != 'body':
            return lidar_pos, lidar_rot

        lidar_pose_compensated = lidar_pos
        lidar_rot_compensated = lidar_rot
        R_robot_rot = lidar_rot

        # 如果有robot四元数，先补偿位置和姿态
        if self.robot_quat is not None and len(self.robot_quat) == 4:
            # robot_quat 是 [w, x, y, z]
            quat_robot = np.array(self.robot_quat, dtype=np.float64)
            R_robot_rot = quat_to_rot(quat_robot[1], quat_robot[2], quat_robot[3], quat_robot[0])
            # 从lidar_rot（旋转矩阵）提取四元数，用于计算补偿
            R_diff = R_robot_rot @ lidar_rot.T
            # 递推均值平滑（并重新正交化）
            R_diff = self._update_rdiff_mean(R_diff)
            # 补偿位置
            lidar_pose_compensated = R_diff @ lidar_pos
            lidar_rot_compensated = R_diff @ lidar_rot

        # T_body_lidar 已知，计算 lidar 在 body 坐标系中对应的 body 位置
        # 已知 T_body_lidar = [R_body_lidar, t_body_lidar]
        # body 位置（world中）= lidar 位置 + lidar 姿态 @ (-R_body_lidar.T @ t_body_lidar)
        R_body_lidar = self.lidar_in_body_rot
        t_body_lidar = self.lidar_in_body_pos
        body_pos =  lidar_pose_compensated + R_robot_rot @ (-R_body_lidar.T @ t_body_lidar)
        body_rot = R_robot_rot @ R_body_lidar.T
        delta_world = lidar_pose_compensated - body_pos
        delta_body = R_robot_rot.T @ delta_world
        return body_pos, body_rot, lidar_pose_compensated, lidar_rot_compensated, delta_body

    def _update_rdiff_mean(self, R_new: np.ndarray) -> np.ndarray:
        """递推更新 R_diff 的均值，并正交化返回。"""
        self.rdiff_count += 1
        alpha = 1.0 / float(self.rdiff_count)
        self.rdiff_mean = self.rdiff_mean + alpha * (R_new - self.rdiff_mean)
        U, _, Vt = np.linalg.svd(self.rdiff_mean)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        lidar_pos = np.array([p.x, p.y, p.z], dtype=np.float64)

        q = msg.pose.pose.orientation
        qx, qy, qz, qw = float(q.x), float(q.y), float(q.z), float(q.w)

        lidar_rot = quat_to_rot(qx, qy, qz, qw)

        pos, rot, lidar_pos_compensated, lidar_rot_compensated, delta_body = self._lidar_pose_to_selected_pose(lidar_pos, lidar_rot)
        self.positions.append(pos)
        self.latest_lidar_pose = (lidar_pos_compensated, lidar_rot_compensated)
        self.latest_pose = (pos, rot)
        print(f"Delta body: {delta_body}")
        # 分别记录x,y,z
        self.x_coords.append(float(pos[0]))
        self.y_coords.append(float(pos[1]))
        self.z_coords.append(float(pos[2]))
        
        if self.start_time is None:
            self.start_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        elapsed_time = current_time - self.start_time
        self.timestamps.append(elapsed_time)
        
        # 加入缓冲区
        self.trajectory_buffer.append([
            f'{elapsed_time:.6f}',
            f'{pos[0]:.6f}',
            f'{pos[1]:.6f}',
            f'{pos[2]:.6f}'
        ])
        self.buffer_count += 1

        # 当缓冲区达到阈值时，批量写入文件
        if self.buffer_count >= self.buffer_size:
            self.flush_trajectory_buffer()
    
    def flush_trajectory_buffer(self):
        """将缓冲区数据批量写入文件。"""
        if not self.trajectory_file or not self.trajectory_buffer:
            return
        
        try:
            with open(self.trajectory_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.trajectory_buffer)
            self.get_logger().debug(f"Flushed {self.buffer_count} trajectory records to file")
        except Exception as e:
            self.get_logger().warn(f"Failed to write trajectory buffer: {e}")
        finally:
            self.trajectory_buffer = []
            self.buffer_count = 0


def set_equal_3d_axes(ax, pts: np.ndarray, margin: float = 0.2):
    if pts.shape[0] == 0:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        return

    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    span = max(span, 0.2)
    half = 0.5 * span * (1.0 + margin)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def draw_world_frame(ax, origin=np.zeros(3), scale=0.2):
    ox, oy, oz = origin
    ax.quiver(ox, oy, oz, scale, 0, 0, color='r', linewidth=2)
    ax.quiver(ox, oy, oz, 0, scale, 0, color='g', linewidth=2)
    ax.quiver(ox, oy, oz, 0, 0, scale, color='b', linewidth=2)


def draw_pose_frame(ax, pos: np.ndarray, rot: np.ndarray, scale: float = 0.15):
    px, py, pz = pos
    x_axis = rot[:, 0] * scale
    y_axis = rot[:, 1] * scale
    z_axis = rot[:, 2] * scale
    ax.quiver(px, py, pz, x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=2)
    ax.quiver(px, py, pz, y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=2)
    ax.quiver(px, py, pz, z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=2)


def main():
    parser = argparse.ArgumentParser(description='ROS2实时3D位姿可视化节点（Matplotlib）')
    parser.add_argument('--topic', type=str, default='/odometry/imu', help='订阅话题名') #/lio_sam/mapping/odometry   /odometry/imu
    parser.add_argument('--history', type=int, default=3000, help='轨迹缓存长度')
    parser.add_argument('--refresh-hz', type=float, default=200.0, help='可视化刷新频率')
    parser.add_argument('--frame-size', type=float, default=0.15, help='当前位姿frame轴长度')
    parser.add_argument('--coord-frame', type=str, default='body', choices=['lidar', 'body'],
                        help='记录坐标系：lidar（原始里程计）或 body（imu/body 转换后）')
    parser.add_argument('--lidar-pos', type=float, nargs=3, default=[0.169, 0.019, 0.245],
                        help='lidar 在 body 坐标系下的位置 [x, y, z] (m)')
    parser.add_argument('--lidar-rot', type=float, nargs=9, 
                        default=[1, 0, 0, 0, 1, 0, 0, 0, 1], # /odometry/imu
                        help='lidar 在 body 坐标系下的旋转矩阵（9个数，行序）')
    args = parser.parse_args()

    rclpy.init()
    node = Odom3DVisualizer(
        topic=args.topic, 
        history=args.history,
        coord_frame=args.coord_frame,
        lidar_in_body_pos=args.lidar_pos,
        lidar_in_body_rot=args.lidar_rot,
        save_dir=Path(__file__).resolve().parent / 'SLAM_trajectory'
    )

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f"SLAM Odom 3D | {args.topic} | q/ESC to quit")

    period = 1.0 / max(1.0, float(args.refresh_hz))

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            ax.cla()
            ax.set_title('World(odom) + Current Pose Frame')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.grid(True)

            pts = np.asarray(node.positions, dtype=np.float64)
            if pts.shape[0] > 0:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='tab:blue', linewidth=1.8, label='trajectory')

            # 当前位姿点（始终按latest_pose画，增大尺寸并加黑边，保证可见）
            if node.latest_pose is not None:
                pos, _ = node.latest_pose
                ax.scatter(
                    [pos[0]], [pos[1]], [pos[2]],
                    color='tab:red', s=90, edgecolors='k', linewidths=0.8,
                    label='current_pose'
                )

            # 同时绘制 lidar 位姿点
            if node.latest_lidar_pose is not None:
                lidar_pos, _ = node.latest_lidar_pose
                ax.scatter(
                    [lidar_pos[0]], [lidar_pos[1]], [lidar_pos[2]],
                    color='tab:orange', s=70, edgecolors='k', linewidths=0.6,
                    label='lidar_pose'
                )

            # 固定世界坐标系（odom）
            draw_world_frame(ax, origin=np.zeros(3), scale=0.25)

            # 当前位姿frame
            if node.latest_pose is not None:
                pos, rot = node.latest_pose
                draw_pose_frame(ax, pos, rot, scale=float(args.frame_size))

            # lidar 位姿frame
            if node.latest_lidar_pose is not None:
                lidar_pos, lidar_rot = node.latest_lidar_pose
                draw_pose_frame(ax, lidar_pos, lidar_rot, scale=float(args.frame_size))

            # 视野范围：包含轨迹与世界原点
            if pts.shape[0] > 0:
                pts_with_origin = np.vstack([pts, np.zeros((1, 3))])
            else:
                pts_with_origin = np.zeros((1, 3))
            set_equal_3d_axes(ax, pts_with_origin, margin=0.3)

            # 状态文本
            ax.text2D(
                0.02, 0.98,
                f"samples={pts.shape[0]}\n"
                f"topic={args.topic}\n"
                f"status={'receiving' if node.latest_pose is not None else 'waiting'}",
                transform=ax.transAxes,
                va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
            )

            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(loc='upper right')
            plt.pause(period)

            # 窗口关闭即退出
            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 确保缓冲区数据被写入文件
        if node.buffer_count > 0:
            node.flush_trajectory_buffer()
            print(f"Final flush written to {node.trajectory_file}")
        
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()


