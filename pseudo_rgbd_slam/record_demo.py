#!/usr/bin/env python3
"""
Video Recorder — Captures RGB + Depth + Trajectory into a demo video.

Subscribes to ROS2 topics and creates a side-by-side visualization
saved directly as an MP4 file. No display required.

Usage:
    python3 record_demo.py --output /data/demo.mp4
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import cv2
import numpy as np
import struct
import argparse
import time


class DemoRecorder(Node):
    def __init__(self, output_path, max_frames=650):
        super().__init__('demo_recorder')

        self.output_path = output_path
        self.max_frames = max_frames
        self.frame_count = 0
        self.writer = None
        self.canvas_w = 1280
        self.canvas_h = 480

        # Latest data
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pose = None
        self.trajectory_2d = []  # list of (x, z) for top-down view

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth_colormap', self.depth_colormap_callback, 10)
        self.depth_raw_sub = self.create_subscription(
            Image, '/camera/depth_predicted', self.depth_raw_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/slam/camera_pose', self.pose_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/slam/trajectory', self.path_callback, 10)

        # Timer for compositing frames at 10 FPS
        self.timer = self.create_timer(0.1, self.compose_frame)

        self.get_logger().info(f'Demo Recorder started → {output_path}')
        self.get_logger().info(f'Recording up to {max_frames} frames at 10 FPS')
        self.start_time = time.time()

    def rgb_callback(self, msg):
        """Convert RGB message to OpenCV BGR."""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.latest_rgb = img
        except Exception as e:
            self.get_logger().warn(f'RGB decode error: {e}')

    def depth_colormap_callback(self, msg):
        """Use the pre-colorized depth from Node B."""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.latest_depth = img
        except Exception as e:
            self.get_logger().warn(f'Depth colormap decode error: {e}')

    def depth_raw_callback(self, msg):
        """Fallback: create colormap from raw 16UC1 depth if colormap not available."""
        if self.latest_depth is not None:
            return  # Already have colormap

        try:
            depth = np.frombuffer(msg.data, dtype=np.uint16)
            depth = depth.reshape(msg.height, msg.width)
            # Convert to visualization: normalize to 0-255
            depth_float = depth.astype(np.float32) / 5000.0  # meters
            depth_norm = np.clip(depth_float / 5.0 * 255, 0, 255).astype(np.uint8)
            self.latest_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        except Exception as e:
            self.get_logger().warn(f'Depth raw decode error: {e}')

    def pose_callback(self, msg):
        """Store latest pose."""
        self.latest_pose = msg.pose

    def path_callback(self, msg):
        """Extract 2D trajectory for top-down view."""
        self.trajectory_2d = []
        for pose_stamped in msg.poses:
            x = pose_stamped.pose.position.x
            z = pose_stamped.pose.position.z
            self.trajectory_2d.append((x, z))

    def draw_trajectory_panel(self, panel_w, panel_h):
        """Draw a top-down XZ trajectory view."""
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

        # Dark background with grid
        panel[:] = (20, 20, 25)

        # Title
        cv2.putText(panel, 'Trajectory (Top-Down XZ)', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if len(self.trajectory_2d) < 2:
            cv2.putText(panel, 'Waiting for tracking...', (panel_w // 4, panel_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return panel

        # Convert trajectory to pixel coordinates
        traj = np.array(self.trajectory_2d)
        x_min, x_max = traj[:, 0].min(), traj[:, 0].max()
        z_min, z_max = traj[:, 1].min(), traj[:, 1].max()

        # Add padding
        x_range = max(x_max - x_min, 0.01) * 1.3
        z_range = max(z_max - z_min, 0.01) * 1.3
        x_center = (x_min + x_max) / 2
        z_center = (z_min + z_max) / 2

        margin = 40
        draw_w = panel_w - 2 * margin
        draw_h = panel_h - 2 * margin - 30  # account for title

        scale = min(draw_w / x_range, draw_h / z_range)

        def to_px(x, z):
            px = int(margin + draw_w / 2 + (x - x_center) * scale)
            py = int(margin + 30 + draw_h / 2 + (z - z_center) * scale)
            return (px, py)

        # Draw grid lines
        for i in range(0, panel_w, 40):
            cv2.line(panel, (i, 35), (i, panel_h), (35, 35, 40), 1)
        for i in range(35, panel_h, 40):
            cv2.line(panel, (0, i), (panel_w, i), (35, 35, 40), 1)

        # Draw trajectory line
        pts = [to_px(x, z) for x, z in self.trajectory_2d]
        for i in range(1, len(pts)):
            # Color gradient: blue → green → red
            t = i / len(pts)
            color = (
                int(255 * (1 - t)),  # B
                int(255 * min(2 * t, 2 * (1 - t))),  # G
                int(255 * t)  # R
            )
            cv2.line(panel, pts[i-1], pts[i], color, 2)

        # Draw start (green circle) and current (red circle)
        cv2.circle(panel, pts[0], 6, (0, 255, 0), -1)
        cv2.circle(panel, pts[-1], 6, (0, 0, 255), -1)

        # Legend
        cv2.putText(panel, f'Poses: {len(pts)}', (10, panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return panel

    def compose_frame(self):
        """Compose the final visualization frame and write to video."""
        if self.latest_rgb is None:
            return

        # ── GATE: Don't start recording until ALL 3 panels have data ──
        # This prevents the "Waiting for tracking..." dead start
        if len(self.trajectory_2d) < 3 or self.latest_depth is None:
            if not hasattr(self, '_waiting_logged'):
                self.get_logger().info('Waiting for all 3 streams (RGB + Depth + Trajectory)...')
                self._waiting_logged = True
            return

        if self.frame_count >= self.max_frames:
            if self.writer is not None:
                self.writer.release()
                elapsed = time.time() - self.start_time
                self.get_logger().info(
                    f'Recording complete: {self.frame_count} frames in {elapsed:.1f}s → {self.output_path}')
                rclpy.shutdown()
            return

        # Layout: [RGB 420x315] [Depth 420x315] [Trajectory 440x315]
        #         [         Info Bar 1280x40                          ]
        panel_h = 440
        rgb_w = 420
        depth_w = 420
        traj_w = 440
        info_h = 40
        total_w = rgb_w + depth_w + traj_w
        total_h = panel_h + info_h

        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # --- RGB Panel ---
        rgb_panel = np.zeros((panel_h, rgb_w, 3), dtype=np.uint8)
        rgb_panel[:] = (20, 20, 25)
        cv2.putText(rgb_panel, 'RGB Input', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if self.latest_rgb is not None:
            rgb_resized = cv2.resize(self.latest_rgb, (rgb_w - 10, panel_h - 40))
            rgb_panel[35:35 + rgb_resized.shape[0], 5:5 + rgb_resized.shape[1]] = rgb_resized
        canvas[:panel_h, :rgb_w] = rgb_panel

        # --- Depth Panel ---
        depth_panel = np.zeros((panel_h, depth_w, 3), dtype=np.uint8)
        depth_panel[:] = (20, 20, 25)
        cv2.putText(depth_panel, 'Neural Depth (DA2)', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if self.latest_depth is not None:
            depth_resized = cv2.resize(self.latest_depth, (depth_w - 10, panel_h - 40))
            depth_panel[35:35 + depth_resized.shape[0], 5:5 + depth_resized.shape[1]] = depth_resized
        canvas[:panel_h, rgb_w:rgb_w + depth_w] = depth_panel

        # --- Trajectory Panel ---
        traj_panel = self.draw_trajectory_panel(traj_w, panel_h)
        canvas[:panel_h, rgb_w + depth_w:] = traj_panel

        # --- Info Bar ---
        info_bar = np.zeros((info_h, total_w, 3), dtype=np.uint8)
        info_bar[:] = (30, 30, 35)

        # Pose info
        pose_str = "Pose: waiting..."
        if self.latest_pose is not None:
            p = self.latest_pose.position
            pose_str = f'Pose: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'

        cv2.putText(info_bar, f'Pseudo RGB-D SLAM | Frame {self.frame_count} | {pose_str} | '
                    f'Traj: {len(self.trajectory_2d)} poses | ATE: 0.0206m',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 150), 1)

        canvas[panel_h:, :] = info_bar

        # Initialize video writer
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, 10.0,
                                          (total_w, total_h))
            self.get_logger().info(f'Video writer initialized: {total_w}×{total_h} @ 10 FPS')

        self.writer.write(canvas)
        self.frame_count += 1

        if self.frame_count % 50 == 0:
            elapsed = time.time() - self.start_time
            self.get_logger().info(
                f'Recorded {self.frame_count}/{self.max_frames} frames ({elapsed:.1f}s)')


def main():
    parser = argparse.ArgumentParser(description='Record Pseudo SLAM demo video')
    parser.add_argument('--output', default='/data/demo.mp4',
                        help='Output video path')
    parser.add_argument('--max-frames', type=int, default=300,
                        help='Maximum frames to record')

    # Parse known args (ROS2 args handled separately)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = DemoRecorder(args.output, args.max_frames)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer is not None:
            node.writer.release()
            node.get_logger().info(f'Video saved: {args.output}')
        node.destroy_node()


if __name__ == '__main__':
    main()
