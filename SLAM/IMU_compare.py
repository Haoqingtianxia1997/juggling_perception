#!/usr/bin/env python3
"""ROS2实时IMU姿态对比可视化（Matplotlib）。

对比两个IMU源的姿态：
1. /odometry/imu: LIO-SAM 里程计输出的姿态
2. RobotClient（Lowstate）: 机器人底层 IMU 直接计算的姿态
3. /livox/imu: Livox IMU 输出的姿态

实时 3D 可视化三个姿态的坐标框。
"""

import argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import time

from robot_bridge_py.robot_client import RobotClient


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """四元数转旋转矩阵。"""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class IMUCompareVisualizer(Node):
    def __init__(self, odom_topic: str = '/odometry/imu', lio_topic: str = '/lio_sam/mapping/odometry', livox_topic: str = '/livox/imu', use_robot_client: bool = True):
        super().__init__('imu_compare_visualizer')
        
        self.odom_topic = odom_topic
        self.lio_topic = lio_topic
        self.livox_topic = livox_topic
        self.use_robot_client = use_robot_client
        
        # 存储最新的姿态与时间戳
        self.latest_odom_pose = None  # (pos[3], rot[3,3])
        self.latest_odom_timestamp = None  # 时间戳（秒）
        self.latest_imu_pose = None   # (pos[3], rot[3,3])
        self.latest_imu_timestamp = None  # 时间戳（秒）
        self.latest_lio_pose = None  # (pos[3], rot[3,3])
        self.latest_lio_timestamp = None  # 时间戳（秒）
        self.latest_livox_pose = None  # (pos[3], rot[3,3])
        self.latest_livox_timestamp = None  # 时间戳（秒）
        
        # 存储轨迹
        self.odom_positions = deque(maxlen=2000)
        self.imu_positions = deque(maxlen=2000)
        self.lio_positions = deque(maxlen=2000)
        self.livox_positions = deque(maxlen=2000)
        
        # 相对旋转矩阵（从 robot_imu_frame 到 /odometry/imu frame）
        self.R_odom_imu = None
        self.R_odom_lio = None
        self.R_odom_livox = None
        
        # 订阅 /odometry/imu
        self.create_subscription(
            Odometry, 
            self.odom_topic, 
            self.odom_cb, 
            qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribed to {self.odom_topic}")

        # self.create_subscription(
        #     Odometry,
        #     self.lio_topic,
        #     self.lio_odom_cb,
        #     qos_profile_sensor_data
        # )
        # self.get_logger().info(f"Subscribed to {self.lio_topic}")

        # self.create_subscription(
        #     Imu,
        #     self.livox_topic,
        #     self.livox_imu_cb,
        #     qos_profile_sensor_data
        # )
        # self.get_logger().info(f"Subscribed to {self.livox_topic}")
        
        # 初始化 RobotClient 来获取底层 IMU 数据
        if self.use_robot_client:
            try:
                self.robot = RobotClient(
                    node=self,
                    robot_type="H1",
                    num_dof=20,
                    control_frequency=50.0,
                    interpolation_order=0.0
                )
                # 创建定时器定期更新 IMU 数据（500Hz）
                self.imu_timer = self.create_timer(0.002, self.update_imu_data)
                self.get_logger().info("Using RobotClient for Lowstate IMU data")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize RobotClient: {e}")
                self.robot = None
        else:
            self.robot = None

    def odom_cb(self, msg: Odometry):
        """处理 /odometry/imu 话题回调。记录时间戳以确保时间同步。"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        rot = quat_to_rot(q.x, q.y, q.z, q.w)
        
        # 从消息头提取时间戳（ROS2 消息时间）
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        self.latest_odom_pose = (pos, rot)
        self.latest_odom_timestamp = timestamp
        self.odom_positions.append(pos.copy())
        
        # 计算相对旋转矩阵
        self.compute_relative_rotation()

    def lio_odom_cb(self, msg: Odometry):
        """处理 /lio_sam/mapping/odometry 话题回调。记录时间戳以确保时间同步。"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pos = np.array([0, 0, 0], dtype=np.float64)
        rot = quat_to_rot(q.x, q.y, q.z, q.w)

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.latest_lio_pose = (pos, rot)
        self.latest_lio_timestamp = timestamp
        self.lio_positions.append(pos.copy())

        self.compute_relative_rotation()

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
            quat = self.robot.quat
            if quat is None or len(quat) < 4:
                return
            
            # 转换四元数为旋转矩阵
            # quat 格式为 [w, x, y, z]
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            rot = quat_to_rot(qx, qy, qz, qw)
            
            # 位置：使用原点（IMU 参考点）
            pos = np.array([0.1, 0.0, 0.0], dtype=np.float64)
            self.latest_imu_pose = (pos, rot)
            self.latest_imu_timestamp = timestamp
            self.imu_positions.append(pos.copy())
            
            # 计算相对旋转矩阵
            self.compute_relative_rotation()
            
        except Exception as e:
            self.get_logger().debug(f"Error updating IMU data: {e}")

    def livox_imu_cb(self, msg: Imu):
        """处理 /livox/imu 话题回调。记录姿态用于可视化。"""
        q = msg.orientation
        pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        rot = quat_to_rot(q.x, q.y, q.z, q.w)

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.latest_livox_pose = (pos, rot)
        self.latest_livox_timestamp = timestamp
        self.livox_positions.append(pos.copy())

        self.compute_relative_rotation()
    
    def compute_relative_rotation(self):
        """计算 robot_imu / lio / livox 到 /odometry/imu frame 的旋转矩阵。"""
        if self.latest_odom_pose is not None and self.latest_imu_pose is not None:
            _, R_odom = self.latest_odom_pose
            _, R_imu = self.latest_imu_pose
            self.R_odom_imu = R_odom @ R_imu.T

        if self.latest_odom_pose is not None and self.latest_lio_pose is not None:
            _, R_odom = self.latest_odom_pose
            _, R_lio = self.latest_lio_pose
            self.R_odom_lio = R_odom @ R_lio.T

        if self.latest_odom_pose is not None and self.latest_livox_pose is not None:
            _, R_odom = self.latest_odom_pose
            _, R_livox = self.latest_livox_pose
            self.R_odom_livox = R_odom @ R_livox.T


def draw_pose_frame(ax, pos: np.ndarray, rot: np.ndarray, scale: float = 0.2):
    """画坐标框。"""
    px, py, pz = pos
    x_axis = rot[:, 0] * scale
    y_axis = rot[:, 1] * scale
    z_axis = rot[:, 2] * scale
    ax.quiver(px, py, pz, x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=2.5)
    ax.quiver(px, py, pz, y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=2.5)
    ax.quiver(px, py, pz, z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=2.5)


def set_equal_3d_axes(ax, margin: float = 0.3):
    """设置等比例坐标轴。"""
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)


def main():
    parser = argparse.ArgumentParser(description='IMU姿态对比 3D 可视化')  #/lio_sam/mapping/odometry   /odometry/imu
    parser.add_argument('--odom-topic', type=str, default='/odometry/imu', 
                        help='里程计话题')
    parser.add_argument('--lio-topic', type=str, default='/lio_sam/mapping/odometry',
                        help='LIO-SAM里程计话题')
    parser.add_argument('--livox-topic', type=str, default='/livox/imu',
                        help='Livox IMU话题')
    parser.add_argument('--use-robot-client', type=bool, default=True,
                        help='是否使用 RobotClient 获取底层 IMU 数据')
    parser.add_argument('--refresh-hz', type=float, default=30.0, 
                        help='可视化刷新频率')
    parser.add_argument('--frame-size', type=float, default=0.2, 
                        help='坐标框轴长度')
    parser.add_argument('--time-sync-threshold', type=float, default=0.001,
                        help='时间同步阈值（秒），超出此范围的数据以不同透明度显示')
    args = parser.parse_args()

    rclpy.init()
    node = IMUCompareVisualizer(
        odom_topic=args.odom_topic,
        lio_topic=args.lio_topic,
        livox_topic=args.livox_topic,
        use_robot_client=args.use_robot_client
    )

    plt.ion()
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("IMU Pose Comparison (Red:Odom, Blue:RobotClient, Purple:LIO-SAM, Green:Livox) | q/ESC to quit")

    period = 1.0 / max(1.0, float(args.refresh_hz))

    try:
        frame_count = 0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            frame_count += 1
            node.frame_count = frame_count

            ax.cla()
            ax.set_title('IMU Pose Comparison (Red:Odom, Blue:RobotClient, Purple:LIO-SAM, Green:Livox)')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.grid(True)

            # 绘制轨迹
            odom_pts = np.asarray(node.odom_positions, dtype=np.float64)
            if odom_pts.shape[0] > 1:
                ax.plot(odom_pts[:, 0], odom_pts[:, 1], odom_pts[:, 2], 
                       color='red', linewidth=1.5, alpha=0.6, label='Odom trajectory')

            imu_pts = np.asarray(node.imu_positions, dtype=np.float64)
            if imu_pts.shape[0] > 1:
                ax.plot(imu_pts[:, 0], imu_pts[:, 1], imu_pts[:, 2], 
                       color='blue', linewidth=1.5, alpha=0.6, label='RobotClient trajectory')

            lio_pts = np.asarray(node.lio_positions, dtype=np.float64)
            if lio_pts.shape[0] > 1:
                ax.plot(lio_pts[:, 0], lio_pts[:, 1], lio_pts[:, 2], 
                       color='purple', linewidth=1.5, alpha=0.6, label='LIO-SAM trajectory')

            livox_pts = np.asarray(node.livox_positions, dtype=np.float64)
            if livox_pts.shape[0] > 1:
                ax.plot(livox_pts[:, 0], livox_pts[:, 1], livox_pts[:, 2], 
                       color='green', linewidth=1.5, alpha=0.6, label='Livox trajectory')

            # 计算时间戳差异
            time_diff = None
            if node.latest_odom_timestamp is not None and node.latest_imu_timestamp is not None:
                time_diff = abs(node.latest_odom_timestamp - node.latest_imu_timestamp)
            
            # 根据时间同步质量设置透明度
            odom_alpha = 1.0
            imu_alpha = 1.0
            if time_diff is not None and time_diff > args.time_sync_threshold:
                # 时间差超过阈值，降低透明度以标示不同步
                odom_alpha = 0.5
                imu_alpha = 0.5

            # 绘制当前姿态坐标框
            if node.latest_odom_pose is not None:
                pos, rot = node.latest_odom_pose
                ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color='red', s=100, edgecolors='darkred', linewidths=1.5,
                          label='Odom pose', zorder=10, alpha=odom_alpha)
                draw_pose_frame(ax, pos, rot, scale=args.frame_size)

            # # 使用 R_odom_imu 将 odom frame 转到 imu frame 后再绘制一份
            # if node.latest_odom_pose is not None and node.R_odom_imu is not None:
            #     pos_odom, rot_odom = node.latest_odom_pose
            #     R_imu_odom = node.R_odom_imu.T
            #     pos_odom_in_imu = np.array([-0.1, 0, 0], dtype=np.float64)
            #     rot_odom_in_imu = R_imu_odom @ rot_odom
            #     ax.scatter([pos_odom_in_imu[0]], [pos_odom_in_imu[1]], [pos_odom_in_imu[2]],
            #               color='orange', s=80, edgecolors='saddlebrown', linewidths=1.2,
            #               label='Odom in IMU frame', zorder=9, alpha=odom_alpha)
            #     draw_pose_frame(ax, pos_odom_in_imu, rot_odom_in_imu, scale=args.frame_size)

            if node.latest_imu_pose is not None:
                pos, rot = node.latest_imu_pose
                ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color='blue', s=100, edgecolors='darkblue', linewidths=1.5,
                          label='RobotClient pose', zorder=10, alpha=imu_alpha)
                draw_pose_frame(ax, pos, rot, scale=args.frame_size)

            if node.latest_lio_pose is not None:
                pos, rot = node.latest_lio_pose
                ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color='purple', s=100, edgecolors='indigo', linewidths=1.5,
                          label='LIO-SAM pose', zorder=10, alpha=odom_alpha)
                draw_pose_frame(ax, pos, rot, scale=args.frame_size)

            if node.latest_livox_pose is not None:
                pos, rot = node.latest_livox_pose
                ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                          color='green', s=100, edgecolors='darkgreen', linewidths=1.5,
                          label='Livox pose', zorder=10, alpha=imu_alpha)
                draw_pose_frame(ax, pos, rot, scale=args.frame_size)

            # 设置视野范围
            set_equal_3d_axes(ax, margin=0.3)

            # 状态文本
            odom_status = 'receiving' if node.latest_odom_pose is not None else 'waiting'
            imu_status = 'receiving' if node.latest_imu_pose is not None else 'waiting'
            lio_status = 'receiving' if node.latest_lio_pose is not None else 'waiting'
            livox_status = 'receiving' if node.latest_livox_pose is not None else 'waiting'
            
            # 格式化时间戳和时间差信息
            odom_ts_str = f"{node.latest_odom_timestamp:.4f}" if node.latest_odom_timestamp else "N/A"
            imu_ts_str = f"{node.latest_imu_timestamp:.4f}" if node.latest_imu_timestamp else "N/A"
            lio_ts_str = f"{node.latest_lio_timestamp:.4f}" if node.latest_lio_timestamp else "N/A"
            livox_ts_str = f"{node.latest_livox_timestamp:.4f}" if node.latest_livox_timestamp else "N/A"
            time_diff_str = f"{time_diff*1000:.1f}ms" if time_diff is not None else "N/A"
            sync_status = "✓ SYNC" if (time_diff is not None and time_diff < args.time_sync_threshold) else "✗ UNSYNC"
            
            # 打印相对旋转矩阵（每秒一次）
            if node.R_odom_imu is not None and node.frame_count % 30 == 0:
                z_axis_in_odom = node.R_odom_imu[:, 2]
                z_axis_error = z_axis_in_odom - np.array([0.0, 0.0, 1.0], dtype=np.float64)
                z_axis_error_norm = float(np.linalg.norm(z_axis_error))
                z_tilt_rad = float(np.arccos(np.clip(z_axis_in_odom[2], -1.0, 1.0)))
                z_tilt_deg = z_tilt_rad * 180.0 / np.pi
                print("\n" + "="*70)
                print("Rotation Matrix: R_odom_imu (from robot_imu_frame to /odometry/imu frame)")
                print("="*70)
                print(node.R_odom_imu)
                print(f"z-axis in odom frame: {z_axis_in_odom}")
                print(f"z-axis deviation vector: {z_axis_error}")
                print(f"z-axis deviation norm: {z_axis_error_norm:.6f}")
                print(f"z-axis tilt angle: {z_tilt_deg:.4f} deg")
                print("="*70)
            
            ax.text2D(
                0.02, 0.98,
                f"Odom: {odom_status} (samples={odom_pts.shape[0]})\n"
                f"  TS: {odom_ts_str}\n"
                f"RobotClient: {imu_status} (samples={imu_pts.shape[0]})\n"
                f"  TS: {imu_ts_str}\n"
                f"LIO-SAM: {lio_status} (samples={lio_pts.shape[0]})\n"
                f"  TS: {lio_ts_str}\n"
                f"Livox: {livox_status} (samples={livox_pts.shape[0]})\n"
                f"  TS: {livox_ts_str}\n"
                f"Time diff: {time_diff_str} {sync_status}\n"
                f"Threshold: {args.time_sync_threshold*1000:.0f}ms",
                transform=ax.transAxes,
                va='top',
                fontsize=9,
                bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray')
            )

            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(loc='upper right', fontsize=9)
            
            plt.pause(period)

            # 窗口关闭即退出
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
