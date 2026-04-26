#!/usr/bin/env python3
"""
Launch file for the Pseudo RGB-D SLAM pipeline.
Launches all 3 nodes + RViz2 for visualization.

Usage:
    ros2 launch pseudo_rgbd_slam pseudo_slam.launch.py
    ros2 launch pseudo_rgbd_slam pseudo_slam.launch.py dataset_path:=/data/my_dataset
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Launch Arguments ─────────────────────────────────────────────────
    dataset_path_arg = DeclareLaunchArgument(
        'dataset_path',
        default_value='/data/rgbd_dataset_freiburg1_desk',
        description='Path to TUM RGB-D dataset directory'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='Frame publish rate in Hz (match to depth inference speed)'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Depth inference device: cuda or cpu'
    )

    use_viewer_arg = DeclareLaunchArgument(
        'use_viewer',
        default_value='true',
        description='Enable ORB-SLAM3 Pangolin viewer'
    )

    # ── Node A: Dataset Broadcaster ──────────────────────────────────────
    node_a = Node(
        package='pseudo_rgbd_slam',
        executable='node_a',
        name='dataset_broadcaster',
        parameters=[{
            'dataset_path': LaunchConfiguration('dataset_path'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'publish_depth_gt': True,
            'loop': False,
        }],
        output='screen',
    )

    # ── Node B: Depth Estimator ──────────────────────────────────────────
    # Delayed start: wait for Node A to start publishing before Node B subscribes
    node_b = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='pseudo_rgbd_slam',
                executable='node_b',
                name='depth_estimator',
                parameters=[{
                    'device': LaunchConfiguration('device'),
                    'depth_factor': 5000.0,
                    'max_depth': 10.0,
                    'min_depth': 0.1,
                    'enable_confidence_mask': True,
                    'confidence_threshold': 0.5,
                    'publish_colormap': True,
                }],
                output='screen',
            )
        ]
    )

    # ── Node C: Pseudo SLAM ──────────────────────────────────────────────
    # Delayed start: wait for both A and B to produce output
    node_c = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='pseudo_rgbd_slam',
                executable='node_c_pseudo_slam',
                name='pseudo_slam',
                parameters=[{
                    'vocabulary_path': '/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt',
                    'config_path': os.path.join(
                        os.path.dirname(__file__), '..', 'config', 'TUM1.yaml'
                    ),
                    'use_viewer': LaunchConfiguration('use_viewer'),
                    'save_trajectory': True,
                    'trajectory_file': '/data/trajectory_pseudo.txt',
                }],
                output='screen',
            )
        ]
    )

    # ── RViz2 ────────────────────────────────────────────────────────────
    rviz = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', os.path.join(
                    os.path.dirname(__file__), '..', 'config', 'rviz_config.rviz'
                )],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        dataset_path_arg,
        publish_rate_arg,
        device_arg,
        use_viewer_arg,
        node_a,
        node_b,
        node_c,
        rviz,
    ])
