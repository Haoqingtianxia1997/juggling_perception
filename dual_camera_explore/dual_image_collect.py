#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


DEFAULT_RIGHT_RGB_TOPIC = "/right/zed_node/rgb/image_rect_color"
DEFAULT_LEFT_RGB_TOPIC = "/left/zed_node/rgb/image_rect_color"
DEFAULT_RIGHT_DEPTH_TOPIC = "/right/zed_node/depth/depth_registered"
DEFAULT_LEFT_DEPTH_TOPIC = "/left/zed_node/depth/depth_registered"


def stamp_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_to_str(stamp):
    return f"{stamp.sec}_{stamp.nanosec:09d}"


def depth_to_uint16_mm(depth_image):
    if depth_image.dtype == np.uint16:
        return depth_image

    depth_float = depth_image.astype(np.float32, copy=False)
    valid = np.isfinite(depth_float) & (depth_float > 0.0)
    depth_mm = np.zeros(depth_float.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(depth_float[valid] * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return depth_mm


def make_depth_colormap(depth_image):
    depth_float = depth_image.astype(np.float32, copy=False)
    valid = np.isfinite(depth_float) & (depth_float > 0.0)
    if not np.any(valid):
        return np.zeros((*depth_float.shape[:2], 3), dtype=np.uint8)

    valid_depth = depth_float[valid]
    min_depth = np.percentile(valid_depth, 1)
    max_depth = np.percentile(valid_depth, 99)
    if max_depth <= min_depth:
        max_depth = min_depth + 1e-6

    clipped = np.clip(depth_float, min_depth, max_depth)
    normalized = np.zeros(depth_float.shape[:2], dtype=np.uint8)
    normalized[valid] = ((clipped[valid] - min_depth) / (max_depth - min_depth) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def label_image(image, label):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (260, 34), (0, 0, 0), -1)
    cv2.putText(labeled, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return labeled


def pad_to_size(image, width, height):
    padded = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    padded[: image.shape[0], : image.shape[1]] = image
    return padded


def make_row(images):
    row_height = max(image.shape[0] for image in images)
    padded_images = [pad_to_size(image, image.shape[1], row_height) for image in images]
    return np.hstack(padded_images)


class DualImageCollector(Node):
    def __init__(
        self,
        output_dir,
        slop,
        queue_size,
        max_pairs,
        right_rgb_topic,
        left_rgb_topic,
        right_depth_topic,
        left_depth_topic,
        visualize,
        save_data,
    ):
        super().__init__("dual_image_collector")

        self.bridge = CvBridge()
        self.max_pairs = max_pairs
        self.processed_count = 0
        self.saved_count = 0
        self.visualize = visualize
        self.save_data = save_data

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.manifest_file = None
        self.manifest_writer = None
        if self.save_data:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_path = self.output_dir / "manifest.csv"
            self.manifest_file = self.manifest_path.open("a", newline="")
            self.manifest_writer = csv.writer(self.manifest_file)
            if self.manifest_path.stat().st_size == 0:
                self.manifest_writer.writerow(
                    [
                        "index",
                        "sync_stamp",
                        "right_rgb_stamp",
                        "right_depth_stamp",
                        "left_rgb_stamp",
                        "left_depth_stamp",
                        "right_rgb_png",
                        "right_depth_npy",
                        "right_depth_mm_png",
                        "right_depth_vis_png",
                        "left_rgb_png",
                        "left_depth_npy",
                        "left_depth_mm_png",
                        "left_depth_vis_png",
                        "max_time_delta_sec",
                    ]
                )
                self.manifest_file.flush()

        right_rgb_sub = message_filters.Subscriber(
            self, Image, right_rgb_topic, qos_profile=qos_profile_sensor_data
        )
        right_depth_sub = message_filters.Subscriber(
            self, Image, right_depth_topic, qos_profile=qos_profile_sensor_data
        )
        left_rgb_sub = message_filters.Subscriber(
            self, Image, left_rgb_topic, qos_profile=qos_profile_sensor_data
        )
        left_depth_sub = message_filters.Subscriber(
            self, Image, left_depth_topic, qos_profile=qos_profile_sensor_data
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [right_rgb_sub, right_depth_sub, left_rgb_sub, left_depth_sub],
            queue_size=queue_size,
            slop=slop,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info("Dual image collector started.")
        self.get_logger().info(f"Right RGB:   {right_rgb_topic}")
        self.get_logger().info(f"Right depth: {right_depth_topic}")
        self.get_logger().info(f"Left RGB:    {left_rgb_topic}")
        self.get_logger().info(f"Left depth:  {left_depth_topic}")
        self.get_logger().info(f"Visualize:   {self.visualize}")
        self.get_logger().info(f"Save data:   {self.save_data}")
        if self.save_data:
            self.get_logger().info(f"Output:      {self.output_dir}")
        self.get_logger().info(f"Sync slop:   {slop:.3f}s, queue_size={queue_size}")
        if self.visualize:
            cv2.namedWindow("dual_image_collect preview", cv2.WINDOW_NORMAL)
            self.get_logger().info("Preview enabled. Press 'q' in the preview window to quit.")

    def synced_callback(self, right_rgb_msg, right_depth_msg, left_rgb_msg, left_depth_msg):
        count_for_limit = self.saved_count if self.save_data else self.processed_count
        if self.max_pairs is not None and count_for_limit >= self.max_pairs:
            return

        try:
            right_rgb = self.bridge.imgmsg_to_cv2(right_rgb_msg, desired_encoding="bgr8")
            left_rgb = self.bridge.imgmsg_to_cv2(left_rgb_msg, desired_encoding="bgr8")
            right_depth = self.bridge.imgmsg_to_cv2(right_depth_msg, desired_encoding="passthrough")
            left_depth = self.bridge.imgmsg_to_cv2(left_depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert ROS images: {exc}")
            return

        right_rgb = cv2.rotate(right_rgb, cv2.ROTATE_90_CLOCKWISE)
        left_rgb = cv2.rotate(left_rgb, cv2.ROTATE_90_CLOCKWISE)
        right_depth = cv2.rotate(right_depth, cv2.ROTATE_90_CLOCKWISE)
        left_depth = cv2.rotate(left_depth, cv2.ROTATE_90_CLOCKWISE)

        right_depth_vis = make_depth_colormap(right_depth)
        left_depth_vis = make_depth_colormap(left_depth)

        stamps = [
            right_rgb_msg.header.stamp,
            right_depth_msg.header.stamp,
            left_rgb_msg.header.stamp,
            left_depth_msg.header.stamp,
        ]
        stamp_values = [stamp_to_float(stamp) for stamp in stamps]
        max_delta = max(stamp_values) - min(stamp_values)

        sync_stamp = stamp_to_str(right_rgb_msg.header.stamp)
        index = self.saved_count
        prefix = f"{index:06d}_{sync_stamp}"

        if self.visualize:
            self.show_preview(right_rgb, right_depth_vis, left_rgb, left_depth_vis, max_delta)

        if self.save_data:
            paths = {
                "right_rgb": self.output_dir / f"{prefix}_right_rgb.png",
                "right_depth_npy": self.output_dir / f"{prefix}_right_depth.npy",
                "right_depth_png": self.output_dir / f"{prefix}_right_depth_mm.png",
                "right_depth_vis": self.output_dir / f"{prefix}_right_depth_vis.png",
                "left_rgb": self.output_dir / f"{prefix}_left_rgb.png",
                "left_depth_npy": self.output_dir / f"{prefix}_left_depth.npy",
                "left_depth_png": self.output_dir / f"{prefix}_left_depth_mm.png",
                "left_depth_vis": self.output_dir / f"{prefix}_left_depth_vis.png",
            }

            ok = True
            ok = cv2.imwrite(str(paths["right_rgb"]), right_rgb) and ok
            ok = cv2.imwrite(str(paths["left_rgb"]), left_rgb) and ok
            ok = cv2.imwrite(str(paths["right_depth_png"]), depth_to_uint16_mm(right_depth)) and ok
            ok = cv2.imwrite(str(paths["left_depth_png"]), depth_to_uint16_mm(left_depth)) and ok
            ok = cv2.imwrite(str(paths["right_depth_vis"]), right_depth_vis) and ok
            ok = cv2.imwrite(str(paths["left_depth_vis"]), left_depth_vis) and ok
            np.save(paths["right_depth_npy"], right_depth)
            np.save(paths["left_depth_npy"], left_depth)

            if not ok:
                self.get_logger().error(f"Failed to write one or more files for pair {index:06d}")
                return

            self.manifest_writer.writerow(
                [
                    index,
                    sync_stamp,
                    stamp_to_str(right_rgb_msg.header.stamp),
                    stamp_to_str(right_depth_msg.header.stamp),
                    stamp_to_str(left_rgb_msg.header.stamp),
                    stamp_to_str(left_depth_msg.header.stamp),
                    paths["right_rgb"].name,
                    paths["right_depth_npy"].name,
                    paths["right_depth_png"].name,
                    paths["right_depth_vis"].name,
                    paths["left_rgb"].name,
                    paths["left_depth_npy"].name,
                    paths["left_depth_png"].name,
                    paths["left_depth_vis"].name,
                    f"{max_delta:.9f}",
                ]
            )
            self.manifest_file.flush()

            self.saved_count += 1
            self.get_logger().info(
                f"Saved pair {index:06d}: max timestamp delta={max_delta * 1000.0:.2f} ms"
            )

        self.processed_count += 1
        count_for_limit = self.saved_count if self.save_data else self.processed_count
        if self.max_pairs is not None and count_for_limit >= self.max_pairs:
            self.get_logger().info(f"Reached max_pairs={self.max_pairs}. Shutting down.")
            rclpy.shutdown()

    def show_preview(self, right_rgb, right_depth_vis, left_rgb, left_depth_vis, max_delta):
        right_rgb_preview = label_image(right_rgb, "right RGB")
        left_rgb_preview = label_image(left_rgb, "left RGB")
        right_depth_preview = label_image(right_depth_vis, "right depth")
        left_depth_preview = label_image(left_depth_vis, "left depth")

        top = make_row([left_rgb_preview, right_rgb_preview])
        bottom = make_row([left_depth_preview, right_depth_preview])
        preview_width = max(top.shape[1], bottom.shape[1])
        top = pad_to_size(top, preview_width, top.shape[0])
        bottom = pad_to_size(bottom, preview_width, bottom.shape[0])
        preview = np.vstack([top, bottom])
        status = (
            f"processed: {self.processed_count} | saved: {self.saved_count} | "
            f"sync delta: {max_delta * 1000.0:.2f} ms | q: quit"
        )
        cv2.rectangle(preview, (0, preview.shape[0] - 34), (preview.shape[1], preview.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            preview,
            status,
            (10, preview.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imshow("dual_image_collect preview", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("Quit requested from preview window.")
            rclpy.shutdown()

    def destroy_node(self):
        if getattr(self, "visualize", False):
            cv2.destroyAllWindows()
        if self.manifest_file is not None and not self.manifest_file.closed:
            self.manifest_file.close()
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect synchronized left/right ZED RGB and depth images."
    )
    default_output_dir = Path(__file__).resolve().parent / "saved_dual_image"
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--slop", type=float, default=0.02, help="Approximate sync tolerance in seconds.")
    parser.add_argument("--queue-size", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=np.inf, help="Stop after saving this many pairs.")
    parser.add_argument("--right-rgb-topic", default=DEFAULT_RIGHT_RGB_TOPIC)
    parser.add_argument("--left-rgb-topic", default=DEFAULT_LEFT_RGB_TOPIC)
    parser.add_argument("--right-depth-topic", default=DEFAULT_RIGHT_DEPTH_TOPIC)
    parser.add_argument("--left-depth-topic", default=DEFAULT_LEFT_DEPTH_TOPIC)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-display", action="store_false", dest="visualize", help=argparse.SUPPRESS)
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = DualImageCollector(
        output_dir=args.output_dir,
        slop=args.slop,
        queue_size=args.queue_size,
        max_pairs=args.max_pairs,
        right_rgb_topic=args.right_rgb_topic,
        left_rgb_topic=args.left_rgb_topic,
        right_depth_topic=args.right_depth_topic,
        left_depth_topic=args.left_depth_topic,
        visualize=args.visualize,
        save_data=args.save_data,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
