cmake_minimum_required(VERSION 3.8)
project(pseudo_rgbd_slam)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ─── Find ROS2 dependencies ─────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# ─── Find ORB-SLAM3 ─────────────────────────────────────────────────────────
# Option to skip Node C build (for testing Python nodes only)
option(SKIP_NODE_C "Skip building Node C (C++ SLAM wrapper)" OFF)

set(ORB_SLAM3_ROOT $ENV{ORB_SLAM3_ROOT})
if(SKIP_NODE_C)
  message(STATUS "SKIP_NODE_C=ON, skipping Node C build")
  set(BUILD_NODE_C OFF)
elseif(NOT ORB_SLAM3_ROOT)
  message(WARNING "ORB_SLAM3_ROOT not set, skipping Node C build")
  set(BUILD_NODE_C OFF)
else()
  set(BUILD_NODE_C ON)
  find_package(OpenCV REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(Pangolin REQUIRED)
endif()

# ─── Node C: Pseudo SLAM (C++) ──────────────────────────────────────────────
if(BUILD_NODE_C)
  add_executable(node_c_pseudo_slam
    src/node_c_pseudo_slam.cpp
  )

  # Disable warnings from ORB-SLAM3 headers (they have many -Wreorder etc.)
  target_compile_options(node_c_pseudo_slam PRIVATE -Wno-reorder -Wno-deprecated-declarations -Wno-sign-compare)

  target_include_directories(node_c_pseudo_slam PRIVATE
    ${ORB_SLAM3_ROOT}/include
    ${ORB_SLAM3_ROOT}/include/CameraModels
    ${ORB_SLAM3_ROOT}
    # Use ORB-SLAM3's BUNDLED Sophus (not system Sophus — APIs differ)
    ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
  )

  target_link_libraries(node_c_pseudo_slam
    ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
  )

  ament_target_dependencies(node_c_pseudo_slam
    rclcpp
    sensor_msgs
    geometry_msgs
    nav_msgs
    std_msgs
    cv_bridge
    message_filters
    tf2
    tf2_ros
  )

  install(TARGETS node_c_pseudo_slam
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

# ─── Python Nodes (A and B) ─────────────────────────────────────────────────
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts with short names for ros2 run
install(PROGRAMS
  pseudo_rgbd_slam/node_a_broadcaster.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME node_a
)
install(PROGRAMS
  pseudo_rgbd_slam/node_b_depth_estimator.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME node_b
)

# ─── Install launch, config, etc. ───────────────────────────────────────────
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)

ament_package()
