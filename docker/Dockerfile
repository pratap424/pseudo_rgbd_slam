# =============================================================================
# Pseudo RGB-D SLAM — Docker Build Environment
# =============================================================================
# Base: ROS2 Humble on Ubuntu 22.04 (required for ORB-SLAM3 compatibility)
# GPU: NVIDIA CUDA for Depth Anything V2 inference
# SLAM: ORB-SLAM3 built from source with RGB-D support
# =============================================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ─── System Dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl unzip \
    pkg-config libssl-dev \
    # OpenCV deps
    libopencv-dev \
    # ORB-SLAM3 deps
    libeigen3-dev libglew-dev libgtk2.0-dev libgtk-3-dev \
    libatlas-base-dev libsuitesparse-dev \
    libtbb-dev libtbb2 \
    # Pangolin deps
    libgl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols \
    libegl1-mesa-dev libgles2-mesa-dev \
    libavcodec-dev libavutil-dev libavformat-dev libswscale-dev \
    libpng-dev libjpeg-dev \
    # Python
    python3 python3-pip python3-dev python3-numpy \
    # ROS2 tools
    locales software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ─── Locale Setup ────────────────────────────────────────────────────────────
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# ─── ROS2 Humble Installation ────────────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-opencv \
    ros-dev-tools \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Source ROS2 in all shells
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# ─── Pangolin (for ORB-SLAM3 viewer) ─────────────────────────────────────────
WORKDIR /opt
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git && \
    cd Pangolin && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_PANGOLIN_PYTHON=OFF \
             -DBUILD_EXAMPLES=OFF && \
    make -j$(nproc) && make install && \
    ldconfig

# ─── Sophus (system install for ROS2 wrapper later) ──────────────────────────
# Header-only v1.22.10 — for our Node C wrapper, NOT for ORB-SLAM3 itself
RUN cd /opt && git clone https://github.com/strasdat/Sophus.git && \
    cd Sophus && git checkout 1.22.10 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_TESTS=OFF \
             -DSOPHUS_USE_BASIC_LOGGING=ON && \
    make -j$(nproc) && make install && \
    ldconfig

# ─── ORB-SLAM3 ───────────────────────────────────────────────────────────────
# IMPORTANT: ORB-SLAM3 uses its OWN bundled Sophus (Thirdparty/Sophus).
# Do NOT redirect to system Sophus — the APIs differ and it will break.
RUN cd /opt && git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git && \
    cd ORB_SLAM3 && \
    # --- Patch C++14 across all CMakeLists ---
    sed -i 's/++11/++14/g' CMakeLists.txt && \
    sed -i 's/++11/++14/g' Thirdparty/DBoW2/CMakeLists.txt && \
    sed -i 's/++11/++14/g' Thirdparty/g2o/CMakeLists.txt && \
    # --- Patch known GCC 11+ compilation issues ---
    # Fix missing includes in several source files
    sed -i '1s/^/#include <algorithm>\n#include <numeric>\n/' src/Tracking.cc && \
    sed -i '1s/^/#include <algorithm>\n/' src/LocalMapping.cc && \
    sed -i '1s/^/#include <algorithm>\n/' src/LoopClosing.cc && \
    sed -i '1s/^/#include <algorithm>\n/' src/CameraModels/KannalaBrandt8.cpp && \
    # Fix deprecated/removed usage patterns
    sed -i 's/CV_LOAD_IMAGE_UNCHANGED/cv::IMREAD_UNCHANGED/g' src/*.cc Examples/*.cc || true && \
    # --- Build ORB-SLAM3 ---
    chmod +x build.sh && ./build.sh

ENV ORB_SLAM3_ROOT=/opt/ORB_SLAM3

# ─── Python Dependencies for Node B (Depth Estimation) ──────────────────────
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    transformers huggingface_hub \
    opencv-python-headless \
    scipy matplotlib Pillow --upgrade && \
    pip3 install "numpy<2.0"

# ─── Pre-download Depth Anything V2 Metric Indoor Small ─────────────────────
# Use the HF Transformers version (-hf suffix) for clean pipeline loading
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf', \
                      local_dir='/opt/depth_model')"

# ─── Download ORB Vocabulary ─────────────────────────────────────────────────
RUN cp /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /opt/ORBvoc.txt

# ─── Workspace Setup ─────────────────────────────────────────────────────────
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# ─── Entry Point ─────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
