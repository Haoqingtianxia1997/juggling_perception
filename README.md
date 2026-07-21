# juggling_perception 使用说明

这个目录是一套基于 ZED RGB-D 图像的抛接球感知与轨迹分析工具。当前代码包含单相机追踪、双相机融合追踪、双相机采集/拼接调试、离线 Kalman 重放、轨迹误差统计、Open3D 点云可视化、MuJoCo 回放，以及 SLAM/IMU 辅助可视化工具。

所有脚本目前都是直接用 Python 运行的独立脚本，不是一个标准 `ament_python` ROS2 package。默认相对路径均以 `src/juggling_perception` 为工作目录或脚本所在目录。

## 快速开始

```bash
cd ~/zed_ws/src/juggling_perception

# 单相机在线追踪
python3 zed_tracker_deploy.py

# 双相机在线融合追踪
python3 dual_zed_tracker_deploy.py

# 查看轨迹和点云
python3 visualize_Tracker_3d.py

# 离线 KF 重放并导出图像/JSON
python3 offline_kf_from_trajectory.py --trajectory-dir trajectory_data

# 误差统计，只保存图不弹窗
python3 cacu_noise.py --trajectory-dir trajectory_data --no-show
```

在线追踪脚本启动时会清空并重建 `trajectory_data/`，如果需要保留旧轨迹，请先备份。

## 依赖

Python 依赖写在 `requirements.txt`：

```bash
pip install -r requirements.txt
```

ROS2 侧依赖通常由系统提供：

```bash
sudo apt install ros-humble-cv-bridge ros-humble-tf2-ros ros-humble-visualization-msgs
```

在线机器人姿态相关脚本依赖 `robot_bridge_py.robot_client.RobotClient`。如果只想离线看轨迹、统计误差、处理保存好的图像，可以不启动 RobotClient。

## 文件结构

```text
juggling_perception/
├── Tracker_config.yaml                 # 相机、外参、检测、KF、轨迹配置
├── perception.py                       # 感知核心：检测、深度重建、双相机融合、关联、KF
├── zed_tracker_deploy.py               # 单相机 ROS2 在线追踪节点
├── dual_zed_tracker_deploy.py          # 双相机 ROS2 在线融合追踪节点
├── zed_image_saver.py                  # 单 ZED RGB/Depth 保存与调试显示
├── visualize_Tracker_3d.py             # Open3D 单步/多轨迹可视化，可写入 gt_pos
├── visualize_Tracker_3d_continuous.py  # Open3D 连续播放版本
├── offline_kf_from_trajectory.py       # 离线 KF 重放与图像导出
├── cacu_noise.py                       # 轨迹拟合与误差统计
├── visuization_in_mujoco.py            # MuJoCo 回放与仿真对比
├── dual_camera_explore/
│   ├── dual_image_collect.py           # 同步采集左右 RGB/Depth，旋转后预览/保存
│   └── find_dual_image_transformation.py # 实时左右 RGB 拼接预览
├── SLAM/
│   ├── IMU_forwarding.py               # RobotClient IMU 转发为 sensor_msgs/Imu
│   ├── IMU_compare.py                  # 多路 IMU/里程计姿态对比
│   ├── SLAM_visualization.py           # /odometry/imu 3D 轨迹实时显示和 CSV 保存
│   ├── plot_trajectory.py              # 绘制 SLAM_trajectory 下最新 CSV
│   └── world_model_test.py             # world_model_prediction 3D 可视化
├── assets/mjcf/                        # MuJoCo XML 模型
├── trajectory_data/                    # 在线追踪输出，启动追踪时会重建
└── dual_camera_explore/saved_dual_image/ # 双相机采集输出
```

## 核心流程

`perception.py` 是在线和离线脚本共用的核心模块：

- `CameraIntrinsics`：保存相机内参，支持像素反投影和相机射线计算。
- `MultiRedBallDetector`：基于颜色、轮廓面积、圆形度等条件检测红球，返回 2D center、contour、area 等信息。
- `BallTracker.localize_ball_in_camera_frame()`：对单张 RGB/Depth 图检测所有球，用深度和内参恢复相机坐标系下的 `point_cam`。
- `BallTracker.detect_and_localize_balls()`：把相机坐标转换到 world 坐标；如果输入是左右相机图像列表，则先做双相机融合。
- `KalmanFilter3D`：三维位置/速度 KF，支持重力、鲁棒 gating、在线 detection 拟合。
- `BallTracker`：管理多个球的预测、关联、更新、落地判断、轨迹记录和 `catch_info` 生成。

当前坐标单位都是米。ZED 深度 topic 通常也是以米为单位的 `float32` 深度图。

## 配置文件

主要配置都在 `Tracker_config.yaml`。

### 相机输入

`camera.profile` 选择当前 profile。已有 profile 包括：

- `left_raw`：raw/fisheye 图像，启用 fisheye undistort。
- `left_rect`：rectified 图像，不做额外去畸变，内参直接写在 profile 里。

单相机脚本 `zed_tracker_deploy.py` 使用：

- `camera.profiles.<profile>.image_topic`
- `camera.profiles.<profile>.depth_topic`
- `camera.profiles.<profile>.intrinsics`，没有时回退到全局 `intrinsics`

双相机脚本 `dual_zed_tracker_deploy.py` 会尝试读取：

- `right_image_topic`
- `left_image_topic`
- `right_depth_topic`
- `left_depth_topic`

当前 YAML profile 里没有显式写这四个键，所以双相机脚本实际使用代码默认值：

```text
/right/zed_node/rgb/image_rect_color
/left/zed_node/rgb/image_rect_color
/right/zed_node/depth/depth_registered
/left/zed_node/depth/depth_registered
```

如果你的实际 topic 改了，建议直接在当前 profile 中补上这四个键。

### 外参

`extrinsics.mode` 当前为 `dual`。配置里同时保留了：

- `extrinsics.single camera`：单相机相对于机器人 `imu_link/body` 的位姿。
- `extrinsics.dual cameras.left/right`：双相机相对于机器人 `imu_link/body` 的位姿。

`dual_zed_tracker_deploy.py` 当前以左相机外参作为融合后的参考相机外参。双相机融合里，右相机检测点会先平移到左相机坐标系，再统一转到 world/body 坐标。

### 追踪参数

常改参数：

- `tracker.runtime.num_balls`：同时追踪球数。
- `tracker.runtime.base_position_source`：机器人本体位置来源，`world_model` 或 `slam`。
- `tracker.runtime.world_model_prediction_topic`：默认 `world_model_prediction`。
- `tracker.runtime.odometry_imu_topic`：默认 `/odometry/imu`。
- `tracker.runtime.dt`：默认 `1/60`。
- `tracker.runtime.ground_z_threshold`：落地判断高度。
- `tracker.detector.center_method`：`nearest` 或 `min_depth`。
- `tracker.detector.ball_radius`：球半径，当前 `0.0375 m`。
- `tracker.association.max_distance`：数据关联距离阈值。
- `tracker.kalman.process_noise` / `measurement_noise`：KF 过程噪声和观测噪声。
- `tracker.trajectory.coord_frame`：轨迹保存坐标系，`world` 或 `body`。

## 在线追踪

### 单相机：zed_tracker_deploy.py

```bash
cd ~/zed_ws/src/juggling_perception
python3 zed_tracker_deploy.py
```

功能：

- 同步订阅一个 RGB topic 和一个 depth topic。
- 检测红球并用 depth 重建 3D 位置。
- 按配置把相机坐标转换到 world/body。
- 进行 KF 预测、数据关联、更新和落地判断。
- 发布 `catch_info` 和 `/ball_markers`。
- 保存 `trajectory_data/trajectory_tracker*.json` 以及对应 RGB/depth 帧。

发布：

- `catch_info`：`Float32MultiArray`，数据为 `[x, y, z, vx, vy, vz]`。
- `/ball_markers`：`visualization_msgs/MarkerArray`，用于 RViz 显示。

### 双相机：dual_zed_tracker_deploy.py

```bash
cd ~/zed_ws/src/juggling_perception
python3 dual_zed_tracker_deploy.py
```

默认订阅四个 topic：

```text
/left/zed_node/rgb/image_rect_color
/left/zed_node/depth/depth_registered
/right/zed_node/rgb/image_rect_color
/right/zed_node/depth/depth_registered
```

当前双相机融合逻辑：

1. 左右 RGB/Depth 用 `ApproximateTimeSynchronizer` 对齐。
2. 分别在左右图像中检测球并恢复各自相机坐标系下的 `point_cam`。
3. 假设双相机竖直平行放置，右相机点云沿相机坐标系 y 方向平移 `0.15 m`，统一到左相机坐标系。
4. 左右检测结果按 `point_cam` 的欧氏距离匹配，阈值当前是 `0.05 m`。
5. 匹配成功的点用距离相关权重融合，未匹配点保留原结果，因此视野会比单相机更大。
6. 融合后的相机坐标点再使用左相机外参转换到 world/body。

注意：

- 当前轨迹保存和 OpenCV 追踪窗口使用左相机 RGB/Depth 做背景图；实际检测结果和 KF 轨迹已经来自双相机融合。
- `trajectory_data/` 同样会在启动时清空并重建。
- 如果左右相机实际间距或摆放方向变了，需要同步修改 `Tracker_config.yaml` 外参和 `perception.py` 里的右相机平移逻辑。

## 双相机采集与拼接调试

### dual_image_collect.py

用于采集已经时间同步的左右 RGB/Depth。每对图像和深度都会顺时针旋转 90 度后再预览/保存。

```bash
cd ~/zed_ws/src/juggling_perception

# 只可视化，不保存，默认行为
python3 dual_camera_explore/dual_image_collect.py

# 可视化并保存
python3 dual_camera_explore/dual_image_collect.py --save-data

# 只保存，不开预览窗口
python3 dual_camera_explore/dual_image_collect.py --save-data --no-visualize
```

常用参数：

- `--output-dir`：默认 `dual_camera_explore/saved_dual_image`
- `--slop`：默认 `0.02 s`
- `--queue-size`：默认 `20`
- `--max-pairs`：达到指定数量后退出
- `--visualize` / `--no-visualize`
- `--save-data` / `--no-save-data`

保存内容：

- `*_left_rgb.png` / `*_right_rgb.png`
- `*_left_depth.npy` / `*_right_depth.npy`
- `*_left_depth_mm.png` / `*_right_depth_mm.png`
- `*_left_depth_vis.png` / `*_right_depth_vis.png`
- `manifest.csv`：记录四路时间戳、文件名和最大同步误差。

### find_dual_image_transformation.py

用于实时查看左右 RGB 的粗拼接效果，不保存数据，不参与主追踪 pipeline。

```bash
cd ~/zed_ws/src/juggling_perception

# 默认 translation 拼接
python3 dual_camera_explore/find_dual_image_transformation.py

# 尝试 homography
python3 dual_camera_explore/find_dual_image_transformation.py --transform-mode homography
```

主要参数：

- `--baseline-m`：显示用相机间距，默认 `0.09`
- `--stitch-scale`：特征匹配用缩放，默认 `0.5`
- `--transform-mode`：`translation` 或 `homography`
- `--transform-interval`：每 N 帧重估一次变换，默认 `5`
- `--min-matches`：ORB 最少匹配数，默认 `30`
- `--blend-width`：融合带宽，默认 `80`
- `--display-scale`：仅影响窗口显示尺寸

该脚本目前只用 RGB 特征做实时拼接。对于近距离物体，左右相机视差会导致重影，这是几何上正常的限制。

## 轨迹数据

在线追踪保存到 `trajectory_data/`。每条轨迹通常包括：

```text
trajectory_tracker0_YYYYMMDD_HHMMSS_0000.json
trajectory_tracker0_YYYYMMDD_HHMMSS_0000/
├── rgb_000000.png
├── rgb_overlay_000000.png
├── depth_000000.npy
└── ...
```

JSON 中常见字段：

- `frames`：逐帧记录。
- `detection_pos`：检测重建位置，坐标系由 `tracker.trajectory.coord_frame` 决定。
- `kf_pos` / `kf_vel`：KF 更新后的位置和速度。
- `kf_predict_pos` / `kf_predict_vel`：预测状态。
- `online_detection_fit_pos` / `online_detection_fit_vel`：在线 detection 拟合结果。
- `gt_pos`：由 Open3D 点云拟合后写入的 ground truth 位置，可选字段。
- `body_pos` / `body_rot`：机器人 body 在 world 中的位姿，用于 body/world 转换。

在线追踪保存的 `rgb_overlay_*.png` 会画出检测 contour、center、有效区域边框等调试信息。

## Open3D 轨迹与点云可视化

```bash
cd ~/zed_ws/src/juggling_perception

# 自动扫描 trajectory_data 下所有轨迹
python3 visualize_Tracker_3d.py

# 查看单条轨迹
python3 visualize_Tracker_3d.py trajectory_data/trajectory_tracker0_20260721_164657_0000.json

# 指定帧范围
python3 visualize_Tracker_3d.py trajectory_data/trajectory_tracker0_20260721_164657_0000.json --start 10 --end 80

# 用点云拟合球心并写入 gt_pos，写回前会生成备份
python3 visualize_Tracker_3d.py --add-gt-pos
```

常用参数：

- `--dir`：轨迹目录。
- `--start` / `--end`：帧范围。
- `--enable-x-filter` / `--x-max`：过滤点云。
- `--hide-gt-marker`：隐藏 GT marker。
- `--no-pre-stats`：不在可视化前打印 Detection-GT 统计。
- `--gt-balls`：GT 聚类候选球数。
- `--gt-match-max-dist`：GT 簇与先验匹配最大距离。
- `--gt-dbscan-eps` / `--gt-dbscan-min-points`：DBSCAN 参数。

`visualize_Tracker_3d_continuous.py` 是连续播放版本，参数基本相同，适合快速查看整段轨迹。

## 离线 KF 重放

```bash
cd ~/zed_ws/src/juggling_perception

python3 offline_kf_from_trajectory.py

python3 offline_kf_from_trajectory.py \
  --trajectory-dir trajectory_data \
  --config Tracker_config.yaml \
  --no-interactive
```

功能：

- 读取在线保存的轨迹 JSON。
- 按当前 `Tracker_config.yaml` 重放 KF。
- 可选导出 N 步预测轨迹。
- 生成离线 JSON、3D 图和时序图。

主要参数：

- `--trajectory-dir`
- `--config`
- `--predict-n`
- `--annotate-every`
- `--no-interactive`
- `--output-dir`
- `--display-scale`
- `--tick-step`
- `--display-frame {auto,world,body}`
- `--frame-axes-every`

输出默认写入：

```text
trajectory_data/offline_kf_outputs/
```

如果 `Tracker_config.yaml` 中 `tracker.offline.enable_predict_n: false`，脚本会忽略 `predict_n`，等效为不画 future prediction。

## 误差统计

```bash
cd ~/zed_ws/src/juggling_perception

python3 cacu_noise.py --trajectory-dir trajectory_data --no-show

python3 cacu_noise.py \
  --trajectory-dir trajectory_data \
  --fit-method huber \
  --fit-source gt \
  --no-show
```

功能：

- 对每条轨迹拟合 GT 或 detection 曲线。
- 统计 `kf_update - gt_fit`、`online_det_fit - gt_fit` 等误差。
- 区分全时段和下降段 `vz <= 0`。
- 为每条轨迹保存位置、速度、创新量、方差等图。

主要参数：

- `--trajectory-dir`
- `--output-dir`：默认 `trajectory_data/detection_plots`
- `--fit-method {ols,ransac,huber}`
- `--fit-source {detection,gt}`
- `--no-show`

## MuJoCo 回放

```bash
cd ~/zed_ws/src/juggling_perception

python3 visuization_in_mujoco.py

python3 visuization_in_mujoco.py \
  --trajectory-dir trajectory_data \
  --model assets/mjcf/h1_juggling_camera.xml \
  --init-source kf
```

功能：

- 读取轨迹 JSON。
- 将 KF/detection 初始化到 MuJoCo 小球状态。
- 对比 KF、运动学预测、GT 运动学和仿真结果。
- 轨迹结束后自动保存误差图和状态图。

参数：

- `--model`：MJCF/XML 模型路径；默认使用脚本内置解析逻辑。
- `--trajectory-dir`：轨迹目录。
- `--init-source {kf,detection}`：小球初始化来源。

MJCF 模型位于 `assets/mjcf/`，包括 `h1_juggling_camera.xml`、`juggling_scene.xml` 等。

## 单相机图像保存

`zed_image_saver.py` 是早期单相机调试工具。它订阅：

```text
/zed/zed_node/left/image_rect_color
/zed/zed_node/depth/depth_registered
```

运行：

```bash
cd ~/zed_ws/src/juggling_perception

# 只显示
python3 zed_image_saver.py

# 显示并保存
python3 zed_image_saver.py --save
```

保存位置：

```text
~/zed_ws/data/rgb/
~/zed_ws/data/depth/
~/zed_ws/data/depth_image/
```

窗口中按 `q` 退出。

## SLAM 和 IMU 工具

这些脚本主要用于验证机器人本体位姿、低层 IMU、Livox/LIO-SAM 输出和 `world_model_prediction`。

### IMU_forwarding.py

从 RobotClient 读取低层 IMU，并发布成 ROS2 `sensor_msgs/Imu`。

```bash
python3 SLAM/IMU_forwarding.py \
  --publish-topic /robot/imu \
  --stamp-topic /livox/imu \
  --frame-id livox_frame \
  --publish-hz 100
```

### IMU_compare.py

比较 `/odometry/imu`、`/lio_sam/mapping/odometry`、`/livox/imu` 和 RobotClient 姿态。

```bash
python3 SLAM/IMU_compare.py
```

常用参数：

- `--odom-topic`
- `--lio-topic`
- `--livox-topic`
- `--use-robot-client`
- `--refresh-hz`
- `--frame-size`
- `--time-sync-threshold`

### SLAM_visualization.py

实时显示 odometry 3D 轨迹，并把轨迹 CSV 保存到 `SLAM/SLAM_trajectory/`。

```bash
python3 SLAM/SLAM_visualization.py --topic /odometry/imu --coord-frame body
```

常用参数：

- `--topic`
- `--history`
- `--refresh-hz`
- `--frame-size`
- `--coord-frame {lidar,body}`
- `--lidar-pos`
- `--lidar-rot`

### plot_trajectory.py

读取 `SLAM/SLAM_trajectory/` 下最新 CSV 并绘制 x/y/z、3D 轨迹和 XY 投影。

```bash
python3 SLAM/plot_trajectory.py
```

### world_model_test.py

订阅 `Float32MultiArray` 格式的 world model 预测并 3D 显示。

```bash
python3 SLAM/world_model_test.py --topic world_model_prediction --show-velocity
```

消息前 3 维是位置 `[x, y, z]`，如果包含 6 维，则后 3 维作为速度 `[vx, vy, vz]`。

## 常见问题

### 看不到图像窗口

- 确认脚本运行在有图形界面的终端里，`DISPLAY` 正常。
- 如果通过 SSH 运行，需要 X11 转发或在本机图形会话里运行。
- OpenCV 窗口需要 `cv2.waitKey()`，不要把脚本放到无 GUI 环境里跑。
- 确认 topic 有数据：`ros2 topic hz <topic>`。

### 双相机没有数据或同步不到

- 检查四个 topic 是否存在。
- 检查左右 RGB 和 depth 时间戳是否接近。
- 增大 `--slop`，例如 `--slop 0.05`。
- 确认左右相机命名空间是 `/left` 和 `/right`，否则改脚本参数或 YAML profile。

### 检测不到红球

- 检查曝光、白平衡、球颜色是否仍满足 `MultiRedBallDetector` 的 HSV/颜色逻辑。
- 调整 `tracker.detector.min_area`、`max_area`、`min_circularity`。
- 调整 `center_method`。`nearest` 使用最近深度点加球半径补偿；`min_depth` 使用中心射线和表面深度补偿。
- 检查深度图是否有 NaN、0 或明显空洞。

### 双相机融合后位置不对

- 先确认 `Tracker_config.yaml` 中左右相机外参真实可靠。
- 确认右相机到左相机的平移方向和距离是否仍是当前代码假设的 y 方向 `0.15 m`。
- 融合阈值 `merge_threshold = 0.05` 写在 `perception.py` 中，匹配不到时会保留左右未匹配结果。
- 当前双相机最终以左相机坐标系和左相机外参为参考。

### 拼接图有重影

`find_dual_image_transformation.py` 只做 RGB 特征拼接，不是真正基于深度的三维重投影。近距离物体视差大时，重影是正常现象。这个脚本主要用于快速查看两个相机视野和大致重叠区域，不建议用它评估三维融合精度。

### 轨迹坐标系混乱

- `tracker.trajectory.coord_frame: world` 时，JSON 中的 `detection_pos` / `kf_pos` 保存 world 坐标。
- `tracker.trajectory.coord_frame: body` 时，保存机器人 body/imu_link 坐标。
- 在线发布的 `catch_info` 来自 `kf_obs_body`，用于机器人本体坐标下的接球控制。
- `/ball_markers` 用于 RViz 显示，按脚本里的 marker 逻辑发布。

## 推荐工作流

1. 启动 ZED 相机，确认 RGB/depth topic 稳定。
2. 用 `dual_camera_explore/dual_image_collect.py` 检查左右图像、深度和时间同步。
3. 用 `dual_zed_tracker_deploy.py` 做在线融合追踪。
4. 用 `visualize_Tracker_3d.py --add-gt-pos` 从点云拟合 GT。
5. 用 `offline_kf_from_trajectory.py` 离线重放 KF。
6. 用 `cacu_noise.py --no-show` 统计误差。
7. 需要机器人/仿真对比时，再用 `SLAM/*` 和 `visuization_in_mujoco.py`。
