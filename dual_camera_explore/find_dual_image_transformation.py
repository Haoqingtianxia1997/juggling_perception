#!/usr/bin/env python3

import argparse

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
WINDOW_NAME = "dual camera panorama"


def stamp_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def label_image(image, label):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (420, 34), (0, 0, 0), -1)
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


def scale_image(image, scale):
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def feather_blend(base, overlay, overlay_mask, blend_width):
    if blend_width <= 0:
        result = base.copy()
        result[overlay_mask] = overlay[overlay_mask]
        return result

    base_mask = np.any(base > 0, axis=2)
    overlap = base_mask & overlay_mask
    result = base.copy()
    result[overlay_mask & ~overlap] = overlay[overlay_mask & ~overlap]
    if not np.any(overlap):
        return result

    distance_to_base_edge = cv2.distanceTransform((overlay_mask.astype(np.uint8) * 255), cv2.DIST_L2, 3)
    alpha = np.clip(distance_to_base_edge / float(blend_width), 0.0, 1.0)
    alpha = alpha[..., None]
    blended = base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha
    result[overlap] = blended[overlap].astype(np.uint8)
    return result


def robust_median_translation(source_pts, target_pts):
    deltas = target_pts - source_pts
    median_delta = np.median(deltas, axis=0)
    residuals = np.linalg.norm(deltas - median_delta, axis=1)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    threshold = max(4.0, 3.0 * 1.4826 * mad)
    inliers = residuals <= threshold
    if np.count_nonzero(inliers) >= 4:
        median_delta = np.median(deltas[inliers], axis=0)
    return float(median_delta[0]), float(median_delta[1]), int(np.count_nonzero(inliers))


class RealtimeDualPanorama(Node):
    def __init__(
        self,
        right_rgb_topic,
        left_rgb_topic,
        slop,
        queue_size,
        baseline_m,
        stitch_scale,
        transform_interval,
        min_matches,
        ransac_reproj_threshold,
        blend_width,
        display_scale,
        transform_mode,
        transform_smoothing,
    ):
        super().__init__("realtime_dual_panorama")

        self.bridge = CvBridge()
        self.baseline_m = baseline_m
        self.stitch_scale = stitch_scale
        self.transform_interval = transform_interval
        self.min_matches = min_matches
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.blend_width = blend_width
        self.display_scale = display_scale
        self.transform_mode = transform_mode
        self.transform_smoothing = transform_smoothing
        self.frame_count = 0
        self.last_transform_right_to_left = None
        self.last_inlier_count = 0
        self.last_dx = 0.0
        self.last_dy = 0.0
        self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        right_sub = message_filters.Subscriber(
            self, Image, right_rgb_topic, qos_profile=qos_profile_sensor_data
        )
        left_sub = message_filters.Subscriber(
            self, Image, left_rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [right_sub, left_sub], queue_size=queue_size, slop=slop
        )
        self.sync.registerCallback(self.synced_callback)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        self.get_logger().info("Realtime dual-camera panorama started.")
        self.get_logger().info(f"Left RGB:    {left_rgb_topic}")
        self.get_logger().info(f"Right RGB:   {right_rgb_topic}")
        self.get_logger().info(f"Baseline:    {baseline_m:.3f} m")
        self.get_logger().info(f"Mode:        {transform_mode}")
        self.get_logger().info(f"Sync slop:   {slop:.3f}s, queue_size={queue_size}")
        self.get_logger().info("Press 'q' in the preview window to quit.")

    def synced_callback(self, right_msg, left_msg):
        try:
            right_rgb = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
            left_rgb = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert ROS images: {exc}")
            return

        right_rgb = cv2.rotate(right_rgb, cv2.ROTATE_90_CLOCKWISE)
        left_rgb = cv2.rotate(left_rgb, cv2.ROTATE_90_CLOCKWISE)

        need_transform = (
            self.last_transform_right_to_left is None
            or self.frame_count % self.transform_interval == 0
        )
        if need_transform:
            transform_right_to_left, inliers, dx, dy = self.estimate_transform(left_rgb, right_rgb)
            if transform_right_to_left is not None:
                self.last_transform_right_to_left = transform_right_to_left
                self.last_inlier_count = inliers
                self.last_dx = dx
                self.last_dy = dy

        panorama = self.stitch_pair(left_rgb, right_rgb, self.last_transform_right_to_left)
        max_delta = abs(stamp_to_float(right_msg.header.stamp) - stamp_to_float(left_msg.header.stamp))
        self.show_preview(left_rgb, right_rgb, panorama, max_delta)
        self.frame_count += 1

    def estimate_transform(self, left_rgb, right_rgb):
        left_small = scale_image(left_rgb, self.stitch_scale)
        right_small = scale_image(right_rgb, self.stitch_scale)
        left_gray = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)

        left_kp, left_desc = self.orb.detectAndCompute(left_gray, None)
        right_kp, right_desc = self.orb.detectAndCompute(right_gray, None)
        if left_desc is None or right_desc is None:
            return None, 0, self.last_dx, self.last_dy
        if len(left_desc) < 2 or len(right_desc) < 2:
            return None, 0, self.last_dx, self.last_dy

        knn_matches = self.matcher.knnMatch(right_desc, left_desc, k=2)
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) != 2:
                continue
            first, second = match_pair
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

        if len(good_matches) < self.min_matches:
            return None, len(good_matches), self.last_dx, self.last_dy

        right_pts = np.float32([right_kp[m.queryIdx].pt for m in good_matches])
        left_pts = np.float32([left_kp[m.trainIdx].pt for m in good_matches])

        if self.transform_mode == "translation":
            dx_small, dy_small, inliers = robust_median_translation(right_pts, left_pts)
            dx = dx_small / self.stitch_scale
            dy = dy_small / self.stitch_scale
            if self.last_transform_right_to_left is not None:
                alpha = self.transform_smoothing
                dx = (1.0 - alpha) * self.last_dx + alpha * dx
                dy = (1.0 - alpha) * self.last_dy + alpha * dy
            transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
            return transform, inliers, dx, dy

        right_pts_h = right_pts.reshape(-1, 1, 2)
        left_pts_h = left_pts.reshape(-1, 1, 2)
        h_scaled, inlier_mask = cv2.findHomography(
            right_pts_h, left_pts_h, cv2.RANSAC, self.ransac_reproj_threshold
        )
        if h_scaled is None or inlier_mask is None:
            return None, 0, self.last_dx, self.last_dy

        scale_matrix = np.array(
            [[self.stitch_scale, 0.0, 0.0], [0.0, self.stitch_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        h_full = np.linalg.inv(scale_matrix) @ h_scaled @ scale_matrix
        inliers = int(inlier_mask.sum())
        return h_full, inliers, float(h_full[0, 2]), float(h_full[1, 2])

    def stitch_pair(self, left_rgb, right_rgb, transform_right_to_left):
        if transform_right_to_left is None:
            separator = np.zeros((left_rgb.shape[0], 8, 3), dtype=np.uint8)
            return np.hstack([left_rgb, separator, right_rgb])

        left_h, left_w = left_rgb.shape[:2]
        right_h, right_w = right_rgb.shape[:2]
        right_corners = np.float32(
            [[0, 0], [right_w, 0], [right_w, right_h], [0, right_h]]
        ).reshape(-1, 1, 2)
        warped_right_corners = cv2.perspectiveTransform(right_corners, transform_right_to_left)
        left_corners = np.float32(
            [[0, 0], [left_w, 0], [left_w, left_h], [0, left_h]]
        ).reshape(-1, 1, 2)
        all_corners = np.vstack([left_corners, warped_right_corners])

        x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
        x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
        translate = np.array([[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]])
        canvas_w = int(x_max - x_min)
        canvas_h = int(y_max - y_min)
        if canvas_w <= 0 or canvas_h <= 0 or canvas_w > 4 * (left_w + right_w) or canvas_h > 4 * max(left_h, right_h):
            return np.hstack([left_rgb, right_rgb])

        warped_right = cv2.warpPerspective(right_rgb, translate @ transform_right_to_left, (canvas_w, canvas_h))
        warped_mask = cv2.warpPerspective(
            np.full((right_h, right_w), 255, dtype=np.uint8),
            translate @ transform_right_to_left,
            (canvas_w, canvas_h),
        ) > 0

        panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        x_offset = int(-x_min)
        y_offset = int(-y_min)
        panorama[y_offset : y_offset + left_h, x_offset : x_offset + left_w] = left_rgb
        return feather_blend(panorama, warped_right, warped_mask, self.blend_width)

    def show_preview(self, left_rgb, right_rgb, panorama, max_delta):
        left_labeled = label_image(left_rgb, "left rotated RGB")
        right_labeled = label_image(right_rgb, "right rotated RGB")
        panorama_labeled = label_image(
            panorama,
            (
                f"panorama | {self.transform_mode} | baseline {self.baseline_m * 100.0:.1f} cm | "
                f"dx {self.last_dx:.1f} dy {self.last_dy:.1f} | inliers {self.last_inlier_count}"
            ),
        )

        top = make_row([left_labeled, right_labeled])
        preview_width = max(top.shape[1], panorama_labeled.shape[1])
        top = pad_to_size(top, preview_width, top.shape[0])
        panorama_labeled = pad_to_size(panorama_labeled, preview_width, panorama_labeled.shape[0])
        preview = np.vstack([top, panorama_labeled])

        status = f"frame: {self.frame_count} | sync delta: {max_delta * 1000.0:.2f} ms | q: quit"
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

        preview = scale_image(preview, self.display_scale)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.get_logger().info("Quit requested from preview window.")
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime panorama preview for two synchronized ZED RGB streams."
    )
    parser.add_argument("--right-rgb-topic", default=DEFAULT_RIGHT_RGB_TOPIC)
    parser.add_argument("--left-rgb-topic", default=DEFAULT_LEFT_RGB_TOPIC)
    parser.add_argument("--slop", type=float, default=0.01, help="Approximate sync tolerance in seconds.")
    parser.add_argument("--queue-size", type=int, default=10)
    parser.add_argument("--baseline-m", type=float, default=0.09, help="Distance between cameras in meters.")
    parser.add_argument("--stitch-scale", type=float, default=0.5, help="Scale used only for feature matching.")
    parser.add_argument("--transform-mode", choices=["translation", "homography"], default="translation")
    parser.add_argument("--transform-interval", type=int, default=5, help="Re-estimate transform every N frames.")
    parser.add_argument("--homography-interval", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-matches", type=int, default=30)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=4.0)
    parser.add_argument("--blend-width", type=int, default=80)
    parser.add_argument("--display-scale", type=float, default=1.0, help="Uniform scale for display only.")
    parser.add_argument("--transform-smoothing", type=float, default=0.25, help="EMA alpha for translation updates.")
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = RealtimeDualPanorama(
        right_rgb_topic=args.right_rgb_topic,
        left_rgb_topic=args.left_rgb_topic,
        slop=args.slop,
        queue_size=args.queue_size,
        baseline_m=args.baseline_m,
        stitch_scale=args.stitch_scale,
        transform_interval=args.transform_interval if args.homography_interval is None else args.homography_interval,
        min_matches=args.min_matches,
        ransac_reproj_threshold=args.ransac_reproj_threshold,
        blend_width=args.blend_width,
        display_scale=args.display_scale,
        transform_mode=args.transform_mode,
        transform_smoothing=args.transform_smoothing,
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
