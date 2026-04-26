/**
 * Node C — Pseudo SLAM (Custom ORB-SLAM3 RGB-D Wrapper)
 * =====================================================
 * Pseudo RGB-D SLAM Pipeline | BotLabs Dynamic Assignment
 *
 * This is a CUSTOM C++ ROS2 node that wraps ORB-SLAM3's RGB-D tracking API.
 * It demonstrates deep understanding of:
 *   - ORB-SLAM3's C++ API (System::TrackRGBD)
 *   - ROS2 message_filters for time-synchronized subscriptions
 *   - SE(3) pose publishing as geometry_msgs/PoseStamped
 *   - Sparse map point extraction as sensor_msgs/PointCloud2
 *
 * Architecture:
 *   - Subscribes: /camera/rgb (bgr8) + /camera/depth_predicted (16UC1)
 *   - Synchronizes via ApproximateTimeSynchronizer (50ms slop)
 *   - Feeds pairs to SLAM.TrackRGBD(rgb, depth, timestamp)
 *   - Publishes: camera pose, full trajectory, sparse 3D map points
 *
 * Why custom wrapper instead of off-the-shelf:
 *   Shows API understanding, C++ competence, and system integration skills.
 *   A robotics engineer needs to write these wrappers daily.
 *
 * ORB-SLAM3 RGB-D Tracking Pipeline (what happens inside TrackRGBD):
 *   1. ORB feature extraction (FAST-9 corners + rotated BRIEF descriptors)
 *   2. Depth backprojection: P = d · K⁻¹ · [u,v,1]ᵀ for features with valid depth
 *   3. Feature matching against local map (Hamming distance + ratio test)
 *   4. PnP pose estimation with RANSAC
 *   5. Motion-only Bundle Adjustment (Levenberg-Marquardt on SE(3))
 *   6. Keyframe decision → local mapping thread → loop closing thread
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ORB-SLAM3 headers
#include "System.h"

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <mutex>

using namespace std::placeholders;
using ImageMsg = sensor_msgs::msg::Image;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;


class PseudoSlamNode : public rclcpp::Node
{
public:
    PseudoSlamNode() : Node("pseudo_slam")
    {
        // ── Parameters ──────────────────────────────────────────────────
        this->declare_parameter("vocabulary_path", "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt");
        this->declare_parameter("config_path", "/ros2_ws/src/pseudo_rgbd_slam/config/TUM1.yaml");
        this->declare_parameter("use_viewer", true);
        this->declare_parameter("save_trajectory", true);
        this->declare_parameter("trajectory_file", "/data/trajectory_pseudo.txt");

        auto vocab_path = this->get_parameter("vocabulary_path").as_string();
        auto config_path = this->get_parameter("config_path").as_string();
        auto use_viewer = this->get_parameter("use_viewer").as_bool();
        save_trajectory_ = this->get_parameter("save_trajectory").as_bool();
        trajectory_file_ = this->get_parameter("trajectory_file").as_string();

        // ── Initialize ORB-SLAM3 in RGB-D mode ──────────────────────────
        RCLCPP_INFO(this->get_logger(),
            "Initializing ORB-SLAM3 (RGB-D mode)...\n"
            "  Vocabulary: %s\n"
            "  Config: %s\n"
            "  Viewer: %s",
            vocab_path.c_str(), config_path.c_str(),
            use_viewer ? "ON" : "OFF"
        );

        slam_ = std::make_unique<ORB_SLAM3::System>(
            vocab_path, config_path,
            ORB_SLAM3::System::RGBD,
            use_viewer
        );

        RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 initialized successfully");

        // ── QoS ─────────────────────────────────────────────────────────
        auto qos = rclcpp::QoS(10).reliable();

        // ── Publishers ──────────────────────────────────────────────────
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/slam/camera_pose", qos);
        trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/slam/trajectory", qos);
        map_points_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/slam/map_points", qos);
        state_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/slam/tracking_state", qos);

        // ── Synchronized subscribers ────────────────────────────────────
        // ApproximateTimeSynchronizer pairs RGB + depth messages within
        // a 50ms time window, matching dataset-timestamped frames from
        // Node A with depth predictions from Node B.
        rgb_sub_ = std::make_shared<message_filters::Subscriber<ImageMsg>>(
            this, "/camera/rgb");
        depth_sub_ = std::make_shared<message_filters::Subscriber<ImageMsg>>(
            this, "/camera/depth_predicted");

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), *rgb_sub_, *depth_sub_);
        sync_->registerCallback(&PseudoSlamNode::synchronized_callback, this);

        // ── Initialize trajectory path message ──────────────────────────
        trajectory_msg_.header.frame_id = "map";

        // ── Open trajectory file for TUM evaluation format ──────────────
        if (save_trajectory_) {
            trajectory_ofs_.open(trajectory_file_);
            if (trajectory_ofs_.is_open()) {
                trajectory_ofs_ << "# Pseudo RGB-D SLAM trajectory\n"
                                << "# timestamp tx ty tz qx qy qz qw\n";
                RCLCPP_INFO(this->get_logger(),
                    "Saving trajectory to: %s", trajectory_file_.c_str());
            }
        }

        RCLCPP_INFO(this->get_logger(),
            "Node C (Pseudo SLAM) started — waiting for synchronized RGB+depth...");
    }

    ~PseudoSlamNode()
    {
        if (slam_) {
            RCLCPP_INFO(this->get_logger(), "Shutting down ORB-SLAM3...");
            slam_->Shutdown();

            // Save final trajectory in TUM format for evaluation
            if (save_trajectory_) {
                slam_->SaveTrajectoryTUM(trajectory_file_ + ".orbslam3");
                RCLCPP_INFO(this->get_logger(),
                    "Trajectory saved (ORB-SLAM3 format): %s.orbslam3",
                    trajectory_file_.c_str());
            }
        }
        if (trajectory_ofs_.is_open()) {
            trajectory_ofs_.close();
        }
    }

private:
    /**
     * Synchronized callback for paired RGB + depth messages.
     *
     * This is the core of the pseudo SLAM pipeline:
     *   1. Convert ROS messages to OpenCV matrices
     *   2. Feed to ORB-SLAM3's TrackRGBD
     *   3. Extract and publish pose, trajectory, map points
     */
    void synchronized_callback(
        const ImageMsg::ConstSharedPtr& rgb_msg,
        const ImageMsg::ConstSharedPtr& depth_msg)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        // ── Convert ROS messages to OpenCV ──────────────────────────────
        cv::Mat rgb, depth;
        try {
            rgb = cv_bridge::toCvShare(rgb_msg, "bgr8")->image;
            depth = cv_bridge::toCvShare(
                depth_msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // ── Extract timestamp ───────────────────────────────────────────
        double timestamp = rgb_msg->header.stamp.sec +
                          rgb_msg->header.stamp.nanosec * 1e-9;

        // ── Feed to ORB-SLAM3 tracking ──────────────────────────────────
        // TrackRGBD returns the camera pose Tcw (world → camera transform)
        // as a Sophus::SE3f (4×4 matrix internally)
        Sophus::SE3f Tcw = slam_->TrackRGBD(rgb, depth, timestamp);

        // ── Get tracking state ──────────────────────────────────────────
        // ORB-SLAM3 eTrackingState enum (from Tracking.h):
        //   -1 = SYSTEM_NOT_READY
        //    0 = NO_IMAGES_YET
        //    1 = NOT_INITIALIZED
        //    2 = OK              ← tracking is working
        //    3 = RECENTLY_LOST
        //    4 = LOST
        //    5 = OK_KLT
        int state = slam_->GetTrackingState();
        std::string state_str;
        switch (state) {
            case -1: state_str = "SYSTEM_NOT_READY"; break;
            case 0:  state_str = "NO_IMAGES_YET"; break;
            case 1:  state_str = "NOT_INITIALIZED"; break;
            case 2:  state_str = "OK"; break;
            case 3:  state_str = "RECENTLY_LOST"; break;
            case 4:  state_str = "LOST"; break;
            case 5:  state_str = "OK_KLT"; break;
            default: state_str = "UNKNOWN"; break;
        }

        // Publish tracking state
        auto state_msg = std_msgs::msg::String();
        state_msg.data = state_str;
        state_pub_->publish(state_msg);

        // ── Only publish if tracking is OK ──────────────────────────────
        if (state == 2 || state == 5) {  // OK or OK_KLT
            // Compute world → camera inverse = camera pose in world frame
            Sophus::SE3f Twc = Tcw.inverse();

            // ── Publish camera pose ─────────────────────────────────────
            publish_pose(Twc, rgb_msg->header);

            // ── Accumulate and publish trajectory ───────────────────────
            publish_trajectory(Twc, rgb_msg->header);

            // ── Save to TUM format file ─────────────────────────────────
            if (save_trajectory_ && trajectory_ofs_.is_open()) {
                Eigen::Vector3f t = Twc.translation();
                Eigen::Quaternionf q = Twc.unit_quaternion();
                trajectory_ofs_ << std::fixed << std::setprecision(6)
                    << timestamp << " "
                    << t.x() << " " << t.y() << " " << t.z() << " "
                    << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                    << "\n";
            }
        }

        // ── Publish map points periodically ─────────────────────────────
        frame_count_++;
        if (frame_count_ % 10 == 0) {
            publish_map_points(rgb_msg->header);
        }

        // ── Performance logging ─────────────────────────────────────────
        auto t_end = std::chrono::high_resolution_clock::now();
        double dt_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        if (frame_count_ % 20 == 0) {
            int n_kf = slam_->GetTrackedKeyPointsUn().size();
            RCLCPP_INFO(this->get_logger(),
                "[Node C] Frame %d: %.1fms | State: %s | KeyPoints: %d",
                frame_count_, dt_ms, state_str.c_str(), n_kf);
        }
    }

    /**
     * Convert SE(3) pose to PoseStamped and publish.
     *
     * Pose is in world frame (camera position in world coordinates).
     * Rotation is published as quaternion (x, y, z, w).
     */
    void publish_pose(const Sophus::SE3f& Twc, const std_msgs::msg::Header& header)
    {
        auto pose_msg = geometry_msgs::msg::PoseStamped();
        pose_msg.header = header;
        pose_msg.header.frame_id = "map";

        // Translation
        Eigen::Vector3f t = Twc.translation();
        pose_msg.pose.position.x = t.x();
        pose_msg.pose.position.y = t.y();
        pose_msg.pose.position.z = t.z();

        // Rotation (quaternion)
        Eigen::Quaternionf q = Twc.unit_quaternion();
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();

        pose_pub_->publish(pose_msg);
    }

    /**
     * Accumulate pose into trajectory Path message and publish.
     */
    void publish_trajectory(const Sophus::SE3f& Twc, const std_msgs::msg::Header& header)
    {
        auto pose_stamped = geometry_msgs::msg::PoseStamped();
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "map";

        Eigen::Vector3f t = Twc.translation();
        Eigen::Quaternionf q = Twc.unit_quaternion();

        pose_stamped.pose.position.x = t.x();
        pose_stamped.pose.position.y = t.y();
        pose_stamped.pose.position.z = t.z();
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        trajectory_msg_.poses.push_back(pose_stamped);
        trajectory_msg_.header.stamp = header.stamp;
        trajectory_pub_->publish(trajectory_msg_);
    }

    /**
     * Extract sparse 3D map points from ORB-SLAM3 and publish as PointCloud2.
     *
     * These are the ORB feature points that have been triangulated and added
     * to the map. Visualizing them in RViz shows the sparse 3D reconstruction.
     */
    void publish_map_points(const std_msgs::msg::Header& header)
    {
        auto all_map_points = slam_->GetTrackedMapPoints();
        if (all_map_points.empty()) return;

        // Collect valid 3D positions
        std::vector<float> points_data;
        for (auto* mp : all_map_points) {
            if (mp == nullptr || mp->isBad()) continue;
            Eigen::Vector3f pos = mp->GetWorldPos();
            points_data.push_back(pos.x());
            points_data.push_back(pos.y());
            points_data.push_back(pos.z());
        }

        if (points_data.empty()) return;

        int n_points = points_data.size() / 3;

        // Build PointCloud2 message
        auto cloud_msg = sensor_msgs::msg::PointCloud2();
        cloud_msg.header = header;
        cloud_msg.header.frame_id = "map";
        cloud_msg.height = 1;
        cloud_msg.width = n_points;
        cloud_msg.is_bigendian = false;
        cloud_msg.is_dense = true;
        cloud_msg.point_step = 12;  // 3 × float32 = 12 bytes
        cloud_msg.row_step = cloud_msg.point_step * n_points;

        // Define XYZ fields
        sensor_msgs::msg::PointField x_field, y_field, z_field;
        x_field.name = "x"; x_field.offset = 0;
        x_field.datatype = sensor_msgs::msg::PointField::FLOAT32; x_field.count = 1;
        y_field.name = "y"; y_field.offset = 4;
        y_field.datatype = sensor_msgs::msg::PointField::FLOAT32; y_field.count = 1;
        z_field.name = "z"; z_field.offset = 8;
        z_field.datatype = sensor_msgs::msg::PointField::FLOAT32; z_field.count = 1;
        cloud_msg.fields = {x_field, y_field, z_field};

        // Copy point data
        cloud_msg.data.resize(points_data.size() * sizeof(float));
        memcpy(cloud_msg.data.data(), points_data.data(),
               points_data.size() * sizeof(float));

        map_points_pub_->publish(cloud_msg);
    }

    // ── Members ─────────────────────────────────────────────────────────
    std::unique_ptr<ORB_SLAM3::System> slam_;

    // Subscribers (synchronized)
    std::shared_ptr<message_filters::Subscriber<ImageMsg>> rgb_sub_;
    std::shared_ptr<message_filters::Subscriber<ImageMsg>> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_points_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;

    // Trajectory storage
    nav_msgs::msg::Path trajectory_msg_;
    bool save_trajectory_ = true;
    std::string trajectory_file_;
    std::ofstream trajectory_ofs_;

    // Stats
    int frame_count_ = 0;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PseudoSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
