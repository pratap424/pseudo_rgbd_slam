#!/usr/bin/env python3
"""
Node B — Neural Metric Depth Estimator
========================================
Pseudo RGB-D SLAM Pipeline | BotLabs Dynamic Assignment

Subscribes to RGB images and produces metric depth maps using
Depth Anything V2 Metric Indoor Small (ViT-S backbone).

Pipeline:
  1. Receive RGB frame (/camera/rgb)
  2. Preprocess → 518×518 (handled internally by model)
  3. Inference → float32 depth in meters
  4. Confidence masking: filter unreliable depth predictions
  5. Convert → uint16 with DepthMapFactor=5000 for ORB-SLAM3 compatibility
  6. Publish (/camera/depth_predicted, 16UC1)

Model: Depth Anything V2 Metric Indoor Small
  - Backbone: ViT-S (DINOv2), ~25M parameters
  - Training: Synthetic indoor scenes (Hypersim dataset)
  - Output: Metric depth in meters (NOT relative depth)
  - Max depth: 20m (sufficient for TUM fr1/desk indoor scene)
  - Architecture: Dense Prediction Transformer (DPT) decoder

Why this model:
  - Truly metric output (no scale alignment needed for ORB-SLAM3)
  - Smallest viable model → fits 2GB GPU constraint
  - Best speed/accuracy tradeoff among metric depth models
  - Excellent ONNX/TensorRT export support for edge deployment

Reference: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small
"""

import time
import cv2
import numpy as np
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

import torch


class DepthEstimator(Node):
    """ROS2 node for neural metric depth estimation."""

    def __init__(self):
        super().__init__('depth_estimator')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('model_path', '/opt/depth_model')
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('depth_factor', 5000.0)  # ORB-SLAM3 TUM1.yaml
        self.declare_parameter('max_depth', 10.0)  # meters (indoor limit)
        self.declare_parameter('min_depth', 0.1)   # meters
        self.declare_parameter('enable_confidence_mask', True)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('publish_colormap', True)  # For visualization

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        device_str = self.get_parameter('device').get_parameter_value().string_value
        self.depth_factor = self.get_parameter('depth_factor').get_parameter_value().double_value
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value
        self.min_depth = self.get_parameter('min_depth').get_parameter_value().double_value
        self.enable_confidence_mask = self.get_parameter('enable_confidence_mask').get_parameter_value().bool_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.publish_colormap = self.get_parameter('publish_colormap').get_parameter_value().bool_value

        # ── Device selection ─────────────────────────────────────────────
        if device_str == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.get_logger().info(f'Using GPU: {gpu_name} ({gpu_mem:.1f} GB)')
        else:
            self.device = torch.device('cpu')
            self.get_logger().info('Using CPU (GPU not available or not requested)')

        # ── Load Depth Anything V2 Metric Indoor Small ───────────────────
        self.get_logger().info('Loading Depth Anything V2 Metric Indoor Small...')
        self.model = self._load_model()
        self.get_logger().info('Model loaded successfully')

        # ── QoS ──────────────────────────────────────────────────────────
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscriber ───────────────────────────────────────────────────
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb', self._rgb_callback, qos
        )

        # ── Publishers ───────────────────────────────────────────────────
        self.depth_pub = self.create_publisher(Image, '/camera/depth_predicted', qos)
        if self.publish_colormap:
            self.depth_vis_pub = self.create_publisher(
                Image, '/camera/depth_colormap', qos
            )

        # ── CV Bridge ────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Performance tracking ─────────────────────────────────────────
        self.inference_times = []
        self.frame_count = 0

        self.get_logger().info(
            f'Node B (Depth Estimator) started | '
            f'Device: {self.device} | '
            f'Depth factor: {self.depth_factor} | '
            f'Range: [{self.min_depth}, {self.max_depth}]m | '
            f'Confidence mask: {self.enable_confidence_mask}'
        )

    def _load_model(self):
        """Load Depth Anything V2 Metric Indoor Small model.

        Uses the HuggingFace transformers pipeline for clean loading,
        with fallback to direct torch loading from local path.
        """
        try:
            # Try HuggingFace Transformers pipeline first
            from transformers import pipeline

            pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                device=self.device,
                torch_dtype=torch.float32,
            )
            self._use_pipeline = True
            self.get_logger().info('Loaded via HuggingFace Transformers pipeline')
            return pipe

        except Exception as e:
            self.get_logger().warn(f'HF pipeline failed ({e}), trying direct load...')

            try:
                # Direct load from depth-anything repo
                from depth_anything_v2.dpt import DepthAnythingV2

                model = DepthAnythingV2(
                    encoder='vits',
                    features=64,
                    out_channels=[48, 96, 192, 384],
                )
                # Load from local weights
                import glob
                weight_files = glob.glob(f'{self.model_path}/*.pth') + \
                               glob.glob(f'{self.model_path}/*.safetensors')
                if weight_files:
                    model.load_state_dict(
                        torch.load(weight_files[0], map_location=self.device)
                    )
                model = model.to(self.device).eval()
                self._use_pipeline = False
                self.get_logger().info('Loaded via direct DepthAnythingV2 class')
                return model

            except Exception as e2:
                self.get_logger().error(f'Failed to load model: {e2}')
                raise

    @torch.no_grad()
    def _infer_depth(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """Run depth inference on a single BGR image.

        Args:
            rgb_bgr: BGR image (H, W, 3), uint8

        Returns:
            Depth map (H, W), float32, in meters
        """
        if self._use_pipeline:
            # HuggingFace pipeline expects PIL or RGB numpy
            from PIL import Image as PILImage
            rgb_pil = PILImage.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
            result = self.model(rgb_pil)

            # IMPORTANT: result['depth'] is normalized 0-255 (visualization only)
            # result['predicted_depth'] is the raw metric output in meters
            depth = np.array(result['predicted_depth'])

            # Resize to original resolution if needed
            h, w = rgb_bgr.shape[:2]
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

            return depth.astype(np.float32)
        else:
            # Direct model inference
            rgb_for_model = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth = self.model.infer_image(rgb_for_model)
            return depth.astype(np.float32)

    def _apply_confidence_mask(self, depth: np.ndarray) -> np.ndarray:
        """Apply confidence masking to filter unreliable depth predictions.

        Strategy: Use the spatial gradient of the depth map as a proxy for
        prediction uncertainty. High gradients at object boundaries indicate
        depth discontinuities where the neural network is least reliable.

        This is an engineering contribution to handle DPT decoder artifacts
        at depth boundaries — setting masked pixels to 0 prevents corrupted
        3D map points in ORB-SLAM3.
        """
        # Compute Sobel gradients of depth map
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize gradient magnitude relative to local depth
        # (depth gradient is more significant at close range)
        safe_depth = np.maximum(depth, 0.01)  # Avoid division by zero
        relative_grad = grad_mag / safe_depth

        # Threshold: relative gradient > threshold → unreliable
        # Empirically tuned for Depth Anything V2 on indoor scenes
        mask_unreliable = relative_grad > self.confidence_threshold

        # Also mask out-of-range depths
        mask_range = (depth < self.min_depth) | (depth > self.max_depth)

        # Combined mask
        mask = mask_unreliable | mask_range

        # Set masked pixels to 0 (ORB-SLAM3 ignores depth=0)
        depth_masked = depth.copy()
        depth_masked[mask] = 0.0

        # Log masking stats periodically
        if self.frame_count % 50 == 0:
            pct_masked = 100.0 * np.sum(mask) / mask.size
            self.get_logger().info(
                f'Confidence mask: {pct_masked:.1f}% pixels filtered'
            )

        return depth_masked

    def _depth_to_uint16(self, depth_meters: np.ndarray) -> np.ndarray:
        """Convert metric depth (float32, meters) to uint16 for ORB-SLAM3.

        ORB-SLAM3 TUM1.yaml uses DepthMapFactor: 5000.0
        This means: depth_in_meters = raw_uint16_value / 5000.0

        So we need: raw_uint16_value = depth_in_meters × 5000.0

        Example:
          - Object at 2.0m → uint16 value = 10000
          - Object at 0.5m → uint16 value = 2500
          - Max representable: 65535/5000 = 13.1m
          - depth=0 means "no measurement" → ORB-SLAM3 ignores

        This matches the TUM RGB-D depth format exactly.
        """
        depth_scaled = depth_meters * self.depth_factor
        depth_uint16 = np.clip(depth_scaled, 0, 65535).astype(np.uint16)
        return depth_uint16

    def _create_depth_colormap(self, depth_meters: np.ndarray) -> np.ndarray:
        """Create a colorized depth map for visualization.

        Uses TURBO colormap: near=red, far=blue, invalid=black.
        This is published on /camera/depth_colormap for the screen recording.
        """
        # Normalize to 0-255 for colormap
        valid = depth_meters > 0
        depth_vis = np.zeros_like(depth_meters)
        if np.any(valid):
            d_min = np.percentile(depth_meters[valid], 2)
            d_max = np.percentile(depth_meters[valid], 98)
            depth_vis = np.clip((depth_meters - d_min) / max(d_max - d_min, 0.01), 0, 1)

        depth_vis_u8 = (depth_vis * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(depth_vis_u8, cv2.COLORMAP_TURBO)

        # Set invalid pixels to black
        colormap[~valid] = [0, 0, 0]

        return colormap

    def _rgb_callback(self, msg: Image):
        """Process incoming RGB frame → produce depth prediction."""
        t_start = time.perf_counter()

        # ── Convert ROS message to OpenCV ────────────────────────────────
        try:
            rgb_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # ── Run depth inference ──────────────────────────────────────────
        depth_meters = self._infer_depth(rgb_bgr)

        # ── Apply confidence masking ─────────────────────────────────────
        if self.enable_confidence_mask:
            depth_meters = self._apply_confidence_mask(depth_meters)
        else:
            # Still clamp to valid range
            depth_meters = np.where(
                (depth_meters >= self.min_depth) & (depth_meters <= self.max_depth),
                depth_meters, 0.0
            ).astype(np.float32)

        # ── Convert to uint16 (ORB-SLAM3 format) ────────────────────────
        depth_uint16 = self._depth_to_uint16(depth_meters)

        # ── Publish depth map ────────────────────────────────────────────
        depth_msg = self.bridge.cv2_to_imgmsg(depth_uint16, encoding='16UC1')
        depth_msg.header = msg.header  # Preserve timestamp from RGB
        self.depth_pub.publish(depth_msg)

        # ── Publish colormap visualization ───────────────────────────────
        if self.publish_colormap:
            colormap = self._create_depth_colormap(depth_meters)
            vis_msg = self.bridge.cv2_to_imgmsg(colormap, encoding='bgr8')
            vis_msg.header = msg.header
            self.depth_vis_pub.publish(vis_msg)

        # ── Performance tracking ─────────────────────────────────────────
        t_elapsed = time.perf_counter() - t_start
        self.inference_times.append(t_elapsed)
        self.frame_count += 1

        if self.frame_count % 20 == 0:
            recent = self.inference_times[-100:]
            avg_ms = 1000.0 * sum(recent) / len(recent)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0
            self.get_logger().info(
                f'[Node B] Frame {self.frame_count}: '
                f'{avg_ms:.1f}ms/frame ({fps:.1f} FPS) | '
                f'Depth range: [{np.min(depth_meters[depth_meters > 0]):.2f}, '
                f'{np.max(depth_meters):.2f}]m'
            )


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
