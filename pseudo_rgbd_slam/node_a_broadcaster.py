#!/usr/bin/env python3
"""
Node A — TUM RGB-D Dataset Broadcaster
=======================================
Pseudo RGB-D SLAM Pipeline | BotLabs Dynamic Assignment

Reads the TUM fr1/desk RGB-D dataset and publishes:
  - /camera/rgb           (sensor_msgs/Image, bgr8, 640×480)
  - /camera/camera_info   (sensor_msgs/CameraInfo, TUM fr1 Kinect intrinsics)
  - /camera/depth_gt      (sensor_msgs/Image, 16UC1, ground truth for evaluation)

The broadcaster respects dataset timestamps for proper trajectory evaluation,
and publishes at a configurable rate (default: 10 Hz) to match downstream
depth inference throughput.

Dataset format (TUM RGB-D):
  - rgb/         → RGB images (640×480, PNG)
  - depth/       → Aligned depth maps (640×480, 16-bit PNG, factor 5000)
  - associations.txt → Timestamp-synchronized RGB-depth pairs
  - groundtruth.txt  → Ground truth trajectory (timestamp tx ty tz qx qy qz qw)

Reference: https://cvg.cit.tum.de/data/datasets/rgbd-dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time


# ─── TUM fr1 Kinect Intrinsics ──────────────────────────────────────────────
# These are the CALIBRATED intrinsics for the Kinect sensor used in TUM fr1
# sequences. Source: TUM RGB-D benchmark documentation.
#
# Camera model: pinhole (Brown-Conrady distortion)
#   K = [fx  0  cx]   [517.3   0    318.6]
#       [ 0 fy  cy] = [  0   516.5  255.3]
#       [ 0  0   1]   [  0     0      1  ]
#
# Distortion: [k1, k2, p1, p2, k3] = [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
# Resolution: 640 × 480
# Depth factor: 5000 (raw_uint16 / 5000 = meters)
TUM_FR1_INTRINSICS = {
    'fx': 517.3, 'fy': 516.5,
    'cx': 318.6, 'cy': 255.3,
    'width': 640, 'height': 480,
    'k1': 0.2624, 'k2': -0.9531,
    'p1': -0.0054, 'p2': 0.0026, 'k3': 1.1633,
    'depth_factor': 5000.0,
}


class DatasetBroadcaster(Node):
    """ROS2 node that reads TUM RGB-D dataset and publishes frames."""

    def __init__(self):
        super().__init__('dataset_broadcaster')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('dataset_path', '/data/rgbd_dataset_freiburg1_desk')
        self.declare_parameter('publish_rate', 10.0)      # Hz
        self.declare_parameter('publish_depth_gt', True)   # Publish ground truth depth
        self.declare_parameter('loop', False)              # Loop dataset

        self.dataset_path = Path(
            self.get_parameter('dataset_path').get_parameter_value().string_value
        )
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.publish_depth_gt = self.get_parameter('publish_depth_gt').get_parameter_value().bool_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value

        # ── Validate dataset ─────────────────────────────────────────────
        if not self.dataset_path.exists():
            self.get_logger().error(f'Dataset path does not exist: {self.dataset_path}')
            self.get_logger().info('Download with: scripts/download_tum_dataset.sh')
            raise FileNotFoundError(f'Dataset not found: {self.dataset_path}')

        # ── Load associations ────────────────────────────────────────────
        self.associations = self._load_associations()
        self.current_idx = 0
        self.total_frames = len(self.associations)
        self.get_logger().info(
            f'Loaded {self.total_frames} frame pairs from {self.dataset_path}'
        )

        # ── QoS Profile ─────────────────────────────────────────────────
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Publishers ───────────────────────────────────────────────────
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb', qos)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', qos)
        if self.publish_depth_gt:
            self.depth_gt_pub = self.create_publisher(Image, '/camera/depth_gt', qos)

        # ── CV Bridge ────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Pre-build CameraInfo message ────────────────────────────────
        self.camera_info_msg = self._build_camera_info()

        # ── Timer (publish loop) ─────────────────────────────────────────
        timer_period = 1.0 / self.publish_rate  # seconds
        self.timer = self.create_timer(timer_period, self._publish_frame)

        self.get_logger().info(
            f'Node A (Broadcaster) started: {self.publish_rate} Hz, '
            f'{self.total_frames} frames'
        )

    def _load_associations(self) -> List[Tuple[float, str, float, str]]:
        """Load associations.txt — timestamp-synchronized RGB-depth pairs.

        Format per line: ts_rgb path_rgb ts_depth path_depth
        Example: 1305031102.175304 rgb/1305031102.175304.png 1305031102.160407 depth/1305031102.160407.png
        """
        assoc_file = self.dataset_path / 'associations.txt'

        if not assoc_file.exists():
            # Try to generate from rgb.txt and depth.txt
            self.get_logger().warn(
                'associations.txt not found, attempting to generate...'
            )
            self._generate_associations()

        associations = []
        with open(assoc_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    ts_rgb = float(parts[0])
                    path_rgb = parts[1]
                    ts_depth = float(parts[2])
                    path_depth = parts[3]
                    associations.append((ts_rgb, path_rgb, ts_depth, path_depth))

        if not associations:
            raise RuntimeError(
                f'No valid associations found in {assoc_file}. '
                'Run: python3 associate.py rgb.txt depth.txt > associations.txt'
            )

        return associations

    def _generate_associations(self):
        """Generate associations.txt from rgb.txt and depth.txt using nearest-neighbor matching."""
        rgb_file = self.dataset_path / 'rgb.txt'
        depth_file = self.dataset_path / 'depth.txt'

        if not rgb_file.exists() or not depth_file.exists():
            raise FileNotFoundError(
                f'Cannot generate associations: '
                f'rgb.txt={rgb_file.exists()}, depth.txt={depth_file.exists()}'
            )

        def read_timestamps(filepath):
            entries = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        entries.append((float(parts[0]), parts[1]))
            return entries

        rgb_entries = read_timestamps(rgb_file)
        depth_entries = read_timestamps(depth_file)

        # Match by closest timestamp (max 0.02s difference)
        max_diff = 0.02
        associations = []
        depth_idx = 0

        for ts_rgb, path_rgb in rgb_entries:
            while depth_idx < len(depth_entries) - 1 and \
                  abs(depth_entries[depth_idx + 1][0] - ts_rgb) < \
                  abs(depth_entries[depth_idx][0] - ts_rgb):
                depth_idx += 1

            if depth_idx < len(depth_entries):
                ts_depth, path_depth = depth_entries[depth_idx]
                if abs(ts_rgb - ts_depth) < max_diff:
                    associations.append(
                        f'{ts_rgb:.6f} {path_rgb} {ts_depth:.6f} {path_depth}'
                    )

        assoc_file = self.dataset_path / 'associations.txt'
        with open(assoc_file, 'w') as f:
            f.write('\n'.join(associations) + '\n')

        self.get_logger().info(
            f'Generated {len(associations)} associations → {assoc_file}'
        )

    def _build_camera_info(self) -> CameraInfo:
        """Build CameraInfo message with TUM fr1 Kinect intrinsics."""
        ci = TUM_FR1_INTRINSICS
        msg = CameraInfo()
        msg.header.frame_id = 'camera_link'
        msg.width = ci['width']
        msg.height = ci['height']
        msg.distortion_model = 'plumb_bob'

        # Distortion coefficients [k1, k2, p1, p2, k3]
        msg.d = [ci['k1'], ci['k2'], ci['p1'], ci['p2'], ci['k3']]

        # Intrinsic camera matrix K (3×3, row-major)
        msg.k = [
            ci['fx'], 0.0,      ci['cx'],
            0.0,      ci['fy'], ci['cy'],
            0.0,      0.0,      1.0,
        ]

        # Rectification matrix (identity for monocular)
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix P (3×4)
        msg.p = [
            ci['fx'], 0.0,      ci['cx'], 0.0,
            0.0,      ci['fy'], ci['cy'], 0.0,
            0.0,      0.0,      1.0,      0.0,
        ]

        return msg

    def _timestamp_to_ros(self, ts: float) -> Time:
        """Convert Unix timestamp to ROS2 Time message."""
        msg = Time()
        msg.sec = int(ts)
        msg.nanosec = int((ts - int(ts)) * 1e9)
        return msg

    def _publish_frame(self):
        """Timer callback: publish next RGB frame + CameraInfo + optional depth GT."""
        if self.current_idx >= self.total_frames:
            if self.loop:
                self.current_idx = 0
                self.get_logger().info('Dataset loop: restarting from frame 0')
            else:
                self.get_logger().info('Dataset complete. All frames published.')
                self.timer.cancel()
                return

        ts_rgb, path_rgb, ts_depth, path_depth = self.associations[self.current_idx]

        # ── Load RGB image ───────────────────────────────────────────────
        rgb_path = self.dataset_path / path_rgb
        rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_img is None:
            self.get_logger().warn(f'Failed to read RGB: {rgb_path}')
            self.current_idx += 1
            return

        # ── Build header with dataset timestamp (critical for evaluation)
        header = Header()
        header.stamp = self._timestamp_to_ros(ts_rgb)
        header.frame_id = 'camera_link'

        # ── Publish RGB ──────────────────────────────────────────────────
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8')
        rgb_msg.header = header
        self.rgb_pub.publish(rgb_msg)

        # ── Publish CameraInfo ───────────────────────────────────────────
        self.camera_info_msg.header = header
        self.info_pub.publish(self.camera_info_msg)

        # ── Publish Ground Truth Depth (for evaluation) ──────────────────
        if self.publish_depth_gt:
            depth_path = self.dataset_path / path_depth
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding='16UC1')
                depth_header = Header()
                depth_header.stamp = self._timestamp_to_ros(ts_depth)
                depth_header.frame_id = 'camera_link'
                depth_msg.header = depth_header
                self.depth_gt_pub.publish(depth_msg)

        # ── Progress logging ─────────────────────────────────────────────
        if self.current_idx % 50 == 0:
            self.get_logger().info(
                f'Frame {self.current_idx}/{self.total_frames} '
                f'(t={ts_rgb:.3f})'
            )

        self.current_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = DatasetBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
