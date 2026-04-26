# Pseudo RGB-D SLAM — Neural Depth × ORB-SLAM3

> Replace a physical depth sensor with a neural network. Run ORB-SLAM3 on the predicted depth. See how much you lose.

A modular ROS2 pipeline that swaps out the Microsoft Kinect with [Depth Anything V2](https://depth-anything-v2.github.io/) (Metric Indoor Small) and feeds the predicted depth into [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) for visual SLAM. Built and validated on TUM fr1/desk.

### ⚡ Key Results

| Metric | Pseudo RGB-D (Ours) | Real Kinect (Published) | Ratio |
|---|---|---|---|
| **ATE RMSE** | **0.0206 m** | ~0.016 m | **1.3×** |
| **RPE Translation** | 0.0111 m/frame | — | — |
| **RPE Rotation** | 0.422 °/frame | — | — |
| **Depth Inference** | 141ms (7.1 FPS) | Hardware | — |
| **SLAM Tracking** | 11ms (~90 FPS) | — | — |

**1.3× degradation vs a real depth sensor — neural depth works for SLAM.**

The biggest finding: depth confidence masking (filtering ~0.2% of pixels at object boundaries) makes a **26.9× difference** in accuracy. Without it, the system essentially fails.

### 🎥 Demo

> 3-panel live visualization: RGB input | Neural depth (MAGMA colormap) | Real-time trajectory

https://github.com/user-attachments/assets/demo.mp4

<details>
<summary>Can't see the video? Click to download</summary>

📥 [Download demo.mp4](demo.mp4) (5.3 MB) — Shows the full pipeline running on TUM fr1/desk: RGB frames → DA2 depth inference → ORB-SLAM3 tracking, with real-time trajectory overlay.

</details>

---

## Architecture

```
┌─────────────┐   /camera/rgb   ┌──────────────────┐  /camera/depth_predicted  ┌──────────────────┐
│   Node A    │───────────────►│     Node B        │──────────────────────────►│     Node C       │
│ Broadcaster │  sensor_msgs/  │ Depth Estimator   │  sensor_msgs/Image       │   Pseudo SLAM    │
│ (TUM fr1)   │  Image (bgr8)  │ (DA2 Metric ViTS) │  (16UC1, factor=5000)    │ (ORB-SLAM3 RGBD) │
└─────────────┘                └──────────────────┘                            └──────────────────┘
       │                                                                              │
       └──── /camera/camera_info ──────────────────────────────────────────────────────┘
                                          │
                                    /slam/trajectory → RViz2
                                    /slam/camera_pose
                                    /slam/map_points
```

| Topic | Type | Description |
|---|---|---|
| `/camera/rgb` | `sensor_msgs/Image` (bgr8) | RGB frames from TUM dataset |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | TUM fr1 Kinect intrinsics |
| `/camera/depth_predicted` | `sensor_msgs/Image` (16UC1) | Neural metric depth (×5000) |
| `/camera/depth_gt` | `sensor_msgs/Image` (16UC1) | Kinect ground truth (evaluation) |
| `/camera/depth_colormap` | `sensor_msgs/Image` (bgr8) | Colorized depth for visualization |
| `/slam/camera_pose` | `geometry_msgs/PoseStamped` | Current camera pose |
| `/slam/trajectory` | `nav_msgs/Path` | Full trajectory |
| `/slam/map_points` | `sensor_msgs/PointCloud2` | Sparse 3D map |

---

## Key Design Decisions

### Depth Model: Depth Anything V2 Metric Indoor Small (ViT-S)

| Property | Value |
|---|---|
| **Output** | Metric depth in meters (float32) — NO scale alignment needed |
| **Backbone** | ViT-S (DINOv2), ~25M parameters |
| **Training** | Hypersim synthetic indoor dataset |
| **ONNX/TRT** | Supported for edge deployment |
| **ORB-SLAM3 compat** | `depth_uint16 = (meters × 5000).clip(0, 65535).astype(uint16)` |

Why not Metric3D v2? 2.5× slower. Why not ZoeDepth? Weak on indoor scenes. Why not UniDepthV2? Too big for the 2GB GPU constraint.

### SLAM: ORB-SLAM3 (RGB-D Mode)

ORB-SLAM3 is ORB-SLAM2's direct successor by the same team. The RGB-D tracking pipeline is identical: ORB features → depth backprojection → PnP + RANSAC → Bundle Adjustment on SE(3) → DBoW2 loop closing. I wrote a custom C++ ROS2 wrapper from scratch rather than using off-the-shelf wrappers.

### Value-Add: Depth Confidence Masking

Neural depth is noisiest at object boundaries. I compute the Sobel gradient of the depth map as a confidence proxy and zero out high-gradient regions (depth=0 tells ORB-SLAM3 to ignore the pixel). Ablation showed this is critical: **26.9× ATE improvement** with masking vs without. See report §3.3 for the full analysis.

---

## Quick Start

### Prerequisites
- Windows 11 + WSL2 (Ubuntu 22.04/24.04)
- NVIDIA GPU (GTX 1650 or better, 4GB+ VRAM)
- Docker + NVIDIA Container Toolkit

### 1. Build Docker Image
```bash
cd pseudo_rgbd_slam/docker
docker compose build    # ~30-45 minutes (downloads CUDA, ROS2, ORB-SLAM3, DA2)
```

### 2. Download TUM Dataset
```bash
docker compose run pseudo_slam bash scripts/download_tum_dataset.sh /data
```

### 3. Build ROS2 Package
```bash
docker compose run pseudo_slam bash -c "
    source /opt/ros/humble/setup.bash
    cd /ros2_ws
    colcon build --symlink-install
    source install/setup.bash
"
```

### 4. Run Pipeline
```bash
# Enable X11 forwarding for RViz2
xhost +local:docker

# Launch all 3 nodes + RViz2
docker compose run pseudo_slam bash -c "
    source /opt/ros/humble/setup.bash
    source /ros2_ws/install/setup.bash
    ros2 launch pseudo_rgbd_slam pseudo_slam.launch.py
"
```

### 5. Evaluate
```bash
# Trajectory evaluation (ATE/RPE)
python3 evaluation/trajectory_eval.py \
    --gt /data/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    --est /data/trajectory_pseudo.txt

# Depth quality (vs Kinect GT)
python3 evaluation/depth_quality.py \
    --dataset /data/rgbd_dataset_freiburg1_desk \
    --pred /data/depth_predictions
```

---

## Project Structure

```
pseudo_rgbd_slam/
├── docker/
│   ├── Dockerfile                    # CUDA + ROS2 Humble + ORB-SLAM3 + DA2
│   └── docker-compose.yml
├── config/
│   └── TUM1.yaml                     # ORB-SLAM3 config (fr1 Kinect intrinsics)
├── launch/
│   └── pseudo_slam.launch.py        # Orchestrates all 3 nodes + RViz2
├── pseudo_rgbd_slam/
│   ├── node_a_broadcaster.py        # Node A: TUM dataset reader (Python)
│   └── node_b_depth_estimator.py    # Node B: DA2 depth inference (Python)
├── src/
│   └── node_c_pseudo_slam.cpp       # Node C: ORB-SLAM3 wrapper (C++)
├── evaluation/
│   ├── depth_quality.py             # Depth AbsRel/RMSE/δ evaluation
│   └── trajectory_eval.py           # ATE/RPE with Umeyama alignment
├── scripts/
│   └── download_tum_dataset.sh
├── report/
│   └── report.md                    # Technical report with math derivations
├── CMakeLists.txt                    # Mixed C++/Python ament_cmake build
├── package.xml                       # ROS2 package manifest
└── README.md                        # This file
```

---

## Mathematical Background

See [report/report.md](report/report.md) for full derivations. Key equations:

**Depth Backprojection:** $P_{cam} = d \cdot K^{-1} [u, v, 1]^T$

**Reprojection Error (BA):** $\hat{T} = \arg\min_{T \in SE(3)} \sum_i \rho_H(\|u_i - \pi(T \cdot P_i)\|^2_{\Sigma_i})$

**Levenberg-Marquardt:** $(J^TJ + \lambda D)\Delta\xi = -J^Tr$

---

## Related Work

This project builds on my earlier **Aerial Guardian** drone MOT work (same assignment series), where I implemented:
- ORB feature extraction + RANSAC affine estimation for camera motion compensation
- Real-time perception on Jetson Orin Nano (14.5–36.9 FPS with TensorRT)
- Modular pipeline with ablation studies

The ORB pipeline in Aerial Guardian's `camera_motion.py` is essentially the same thing ORB-SLAM3's frontend does — same FAST corners, same binary descriptors, same Hamming matching, same RANSAC.

---

## License

MIT
