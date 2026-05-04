#!/usr/bin/env python3
"""
从CSV文件绘制轨迹数据。
用法: python plot_trajectory.py <trajectory_csv_file>
或    python plot_trajectory.py --dir /path/to/csv/dir  # 绘制目录下最新的CSV文件
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DEFAULT_TRAJECTORY_DIR = Path(__file__).resolve().parent / 'SLAM_trajectory'


def load_trajectory_csv(csv_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从CSV文件加载轨迹数据。
    
    返回: (times, x_coords, y_coords, z_coords)
    """
    times = []
    x_coords = []
    y_coords = []
    z_coords = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time(s)']))
            x_coords.append(float(row['x(m)']))
            y_coords.append(float(row['y(m)']))
            z_coords.append(float(row['z(m)']))
    
    return (
        np.array(times, dtype=np.float64),
        np.array(x_coords, dtype=np.float64),
        np.array(y_coords, dtype=np.float64),
        np.array(z_coords, dtype=np.float64),
    )


def find_latest_trajectory_csv(directory: Path) -> Optional[Path]:
    """在目录中查找最新的轨迹CSV文件。"""
    csv_files = sorted(directory.glob('trajectory_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if csv_files:
        return csv_files[0]
    return None


def plot_trajectory_2d_separated(times: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """绘制分离的2D轨迹 (time vs x/y/z)。"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Trajectory - Time Series (Separated)', fontsize=14, fontweight='bold')
    
    # X轨迹
    axes[0].plot(times, x, 'r-', linewidth=2, label='X trajectory')
    axes[0].fill_between(times, x, alpha=0.3, color='red')
    axes[0].set_ylabel('X Position [m]', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    
    # Y轨迹
    axes[1].plot(times, y, 'g-', linewidth=2, label='Y trajectory')
    axes[1].fill_between(times, y, alpha=0.3, color='green')
    axes[1].set_ylabel('Y Position [m]', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')
    
    # Z轨迹
    axes[2].plot(times, z, 'b-', linewidth=2, label='Z trajectory')
    axes[2].fill_between(times, z, alpha=0.3, color='blue')
    axes[2].set_ylabel('Z Position [m]', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_trajectory_2d_combined(times: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """绘制合并的2D轨迹 (time vs x/y/z在同一图表)。"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Trajectory - Time Series (Combined)', fontsize=14, fontweight='bold')
    
    ax.plot(times, x, 'r-', linewidth=2.5, label='X', alpha=0.8)
    ax.plot(times, y, 'g-', linewidth=2.5, label='Y', alpha=0.8)
    ax.plot(times, z, 'b-', linewidth=2.5, label='Z', alpha=0.8)
    
    ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position [m]', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_trajectory_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """绘制3D轨迹。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('3D Trajectory', fontsize=14, fontweight='bold')
    
    # 绘制轨迹线
    ax.plot(x, y, z, 'tab:blue', linewidth=2, label='Trajectory')
    
    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], color='green', s=100, marker='o', label='Start', edgecolors='k', linewidths=1)
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=100, marker='s', label='End', edgecolors='k', linewidths=1)
    
    # 绘制采样点
    if len(x) > 1:
        sample_indices = np.linspace(0, len(x) - 1, min(50, len(x)), dtype=int)
        ax.scatter(x[sample_indices], y[sample_indices], z[sample_indices], 
                  color='cyan', s=20, alpha=0.6, label='Sample points')
    
    ax.set_xlabel('X [m]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z [m]', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # 设置等长的轴
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def plot_trajectory_xy_projection(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """绘制XY平面投影。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('XY Plane Projection', fontsize=14, fontweight='bold')
    
    scatter = ax.scatter(x, y, c=z, cmap='viridis', s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
    ax.plot(x, y, 'b-', linewidth=1, alpha=0.3, label='Trajectory')
    
    # 标记起点和终点
    ax.scatter(x[0], y[0], color='green', s=150, marker='o', label='Start', edgecolors='k', linewidths=1.5, zorder=5)
    ax.scatter(x[-1], y[-1], color='red', s=150, marker='s', label='End', edgecolors='k', linewidths=1.5, zorder=5)
    
    ax.set_xlabel('X [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m]', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Z [m]', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig


def compute_statistics(times: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """计算轨迹统计信息。"""
    total_time = times[-1] - times[0] if len(times) > 1 else 0
    
    # 计算距离
    displacements = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    total_distance = np.sum(displacements)
    
    stats = {
        'num_samples': len(times),
        'total_time': total_time,
        'total_distance': total_distance,
        'x_range': (x.min(), x.max()),
        'y_range': (y.min(), y.max()),
        'z_range': (z.min(), z.max()),
        'x_mean': np.mean(x),
        'y_mean': np.mean(y),
        'z_mean': np.mean(z),
        'x_std': np.std(x),
        'y_std': np.std(y),
        'z_std': np.std(z),
    }
    
    return stats


def print_statistics(stats: dict):
    """打印轨迹统计信息。"""
    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Total time: {stats['total_time']:.2f} s")
    print(f"Total distance: {stats['total_distance']:.4f} m")
    print(f"\nPosition ranges:")
    print(f"  X: [{stats['x_range'][0]:.4f}, {stats['x_range'][1]:.4f}] m (mean={stats['x_mean']:.4f}, std={stats['x_std']:.4f})")
    print(f"  Y: [{stats['y_range'][0]:.4f}, {stats['y_range'][1]:.4f}] m (mean={stats['y_mean']:.4f}, std={stats['y_std']:.4f})")
    print(f"  Z: [{stats['z_range'][0]:.4f}, {stats['z_range'][1]:.4f}] m (mean={stats['z_mean']:.4f}, std={stats['z_std']:.4f})")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='从默认 SLAM_trajectory 目录读取并绘制轨迹数据（x,y,z及3D位置）')
    args = parser.parse_args()
    
    # 直接读取默认目录中最新的CSV文件
    csv_path = find_latest_trajectory_csv(DEFAULT_TRAJECTORY_DIR)
    if csv_path is None:
        print(f"错误: 在默认目录中未找到轨迹CSV文件: {DEFAULT_TRAJECTORY_DIR}")
        return
    
    print(f"加载轨迹数据: {csv_path}")
    
    try:
        times, x, y, z = load_trajectory_csv(csv_path)
    except Exception as e:
        print(f"错误: 无法加载CSV文件 - {e}")
        return
    
    if len(times) == 0:
        print("错误: CSV文件为空")
        return
    
    print(f"成功加载 {len(times)} 个样本")
    
    # 计算统计信息
    stats = compute_statistics(times, x, y, z)
    print_statistics(stats)
    
    # 创建图表
    print("正在生成图表...")
    figs = []
    
    # 1. 分离的时间序列
    figs.append(plot_trajectory_2d_separated(times, x, y, z))
    
    # 2. 合并的时间序列
    figs.append(plot_trajectory_2d_combined(times, x, y, z))
    
    # 3. 3D轨迹
    figs.append(plot_trajectory_3d(x, y, z))
    
    # 4. XY平面投影
    figs.append(plot_trajectory_xy_projection(x, y, z))
    
    print(f"已生成 {len(figs)} 个图表")
    plt.show()


if __name__ == '__main__':
    main()
