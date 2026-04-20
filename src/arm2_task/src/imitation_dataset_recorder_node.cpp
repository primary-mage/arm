#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "rclcpp/rclcpp.hpp"
#include "robot_msgs/msg/robot_state.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

namespace fs = std::filesystem;

namespace
{

constexpr int kDof = 5;

std::string sanitize_task_name(std::string task_name)
{
    for (char & ch : task_name) {
        const bool valid = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                           (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        if (!valid) {
            ch = '_';
        }
    }
    if (task_name.empty()) {
        task_name = "unlabeled_task";
    }
    return task_name;
}

std::string current_timestamp_string()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now{};
#if defined(_WIN32)
    localtime_s(&tm_now, &now_c);
#else
    localtime_r(&now_c, &tm_now);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
    return oss.str();
}

std::string quote_json(const std::string & input)
{
    std::ostringstream oss;
    oss << '"';
    for (const char ch : input) {
        switch (ch) {
            case '\\':
                oss << "\\\\";
                break;
            case '"':
                oss << "\\\"";
                break;
            case '\n':
                oss << "\\n";
                break;
            default:
                oss << ch;
                break;
        }
    }
    oss << '"';
    return oss.str();
}

std::string trim_copy(const std::string & input)
{
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return std::string();
    }
    const auto last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1U);
}

bool is_color_encoding_supported(const std::string & encoding)
{
    return encoding == "rgb8" || encoding == "bgr8" || encoding == "bgra8" || encoding == "rgba8";
}

}  // namespace

class ImitationDatasetRecorderNode : public rclcpp::Node
{
public:
    ImitationDatasetRecorderNode() : Node("imitation_dataset_recorder_node")
    {
        dataset_root_ = this->declare_parameter(
            "imitation_dataset.dataset_root",
            std::string("dataset"));
        task_name_ = sanitize_task_name(
            this->declare_parameter("imitation_dataset.task_name", std::string("unnamed_task")));
        image_topic_ = this->declare_parameter(
            "imitation_dataset.image_topic",
            std::string("/camera/color/image_raw"));
        state_topic_ = this->declare_parameter(
            "imitation_dataset.state_topic",
            std::string("/arm2/_lowState/joint"));
        action_topic_ = this->declare_parameter(
            "imitation_dataset.action_topic",
            std::string("/joint_target_state"));
        event_topic_ = this->declare_parameter(
            "imitation_dataset.event_topic",
            std::string("/episode_event"));
        auto_start_ = this->declare_parameter("imitation_dataset.auto_start", false);
        output_fps_ = this->declare_parameter("imitation_dataset.output_fps", 30.0);
        write_mp4_ = this->declare_parameter("imitation_dataset.write_mp4", true);
        mp4_codec_ = this->declare_parameter("imitation_dataset.mp4_codec", std::string("mp4v"));
        overwrite_episode_ =
            this->declare_parameter("imitation_dataset.overwrite_episode", false);

        if (output_fps_ <= 0.0) {
            output_fps_ = 30.0;
        }

        prepare_dataset_root();

        auto image_qos = rclcpp::SensorDataQoS();
        auto state_qos = rclcpp::QoS(rclcpp::KeepLast(20)).best_effort().durability_volatile();
        auto action_qos = rclcpp::QoS(rclcpp::KeepLast(20)).reliable().durability_volatile();
        auto event_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            image_topic_,
            image_qos,
            std::bind(&ImitationDatasetRecorderNode::image_callback, this, std::placeholders::_1));
        state_sub_ = this->create_subscription<robot_msgs::msg::RobotState>(
            state_topic_,
            state_qos,
            std::bind(&ImitationDatasetRecorderNode::state_callback, this, std::placeholders::_1));
        action_sub_ = this->create_subscription<robot_msgs::msg::RobotState>(
            action_topic_,
            action_qos,
            std::bind(&ImitationDatasetRecorderNode::action_callback, this, std::placeholders::_1));
        event_sub_ = this->create_subscription<std_msgs::msg::String>(
            event_topic_,
            event_qos,
            std::bind(&ImitationDatasetRecorderNode::event_callback, this, std::placeholders::_1));

        if (auto_start_) {
            begin_episode(task_name_, "auto_start");
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Imitation dataset recorder started. root=%s image_topic=%s state_topic=%s action_topic=%s event_topic=%s auto_start=%s mp4=%s fps=%.1f",
            dataset_root_.c_str(),
            image_topic_.c_str(),
            state_topic_.c_str(),
            action_topic_.c_str(),
            event_topic_.c_str(),
            auto_start_ ? "true" : "false",
            write_mp4_ ? "true" : "false",
            output_fps_);
    }

    ~ImitationDatasetRecorderNode() override
    {
        finalize();
    }

    void finalize()
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        if (finalized_) {
            return;
        }

        finalized_ = true;
        if (recording_active_) {
            finalize_episode_locked("shutdown");
        }
    }

private:
    void prepare_dataset_root()
    {
        fs::path root_path(dataset_root_);
        if (!root_path.is_absolute()) {
            root_path = fs::current_path() / root_path;
        }

        fs::create_directories(root_path);
        dataset_root_ = root_path.string();
    }

    fs::path build_episode_directory_path(const std::string & episode_task_name) const
    {
        const std::string episode_name =
            "episode_" + current_timestamp_string() + "_" + sanitize_task_name(episode_task_name);
        return fs::path(dataset_root_) / episode_name;
    }

    void prepare_episode_directory(const std::string & episode_task_name)
    {
        const fs::path candidate_dir = build_episode_directory_path(episode_task_name);
        fs::path final_dir = candidate_dir;
        if (fs::exists(final_dir)) {
            if (!overwrite_episode_) {
                int suffix = 1;
                do {
                    final_dir = fs::path(candidate_dir.string() + "_" + std::to_string(suffix));
                    ++suffix;
                } while (fs::exists(final_dir));
            } else {
                fs::remove_all(final_dir);
            }
        }

        fs::create_directories(final_dir);
        episode_dir_ = final_dir.string();
    }

    void open_output_files()
    {
        rgb_timestamp_stream_.open(fs::path(episode_dir_) / "rgb_timestamps.csv", std::ios::out);
        state_stream_.open(fs::path(episode_dir_) / "states.csv", std::ios::out);
        action_stream_.open(fs::path(episode_dir_) / "actions.csv", std::ios::out);

        if (!rgb_timestamp_stream_.is_open() || !state_stream_.is_open() || !action_stream_.is_open()) {
            throw std::runtime_error("Failed to open one or more dataset output files.");
        }

        rgb_timestamp_stream_ << "frame_idx,t\n";
        state_stream_ << "t";
        action_stream_ << "t";
        for (int i = 0; i < kDof; ++i) {
            state_stream_ << ",q" << (i + 1);
            action_stream_ << ",q" << (i + 1);
        }
        for (int i = 0; i < kDof; ++i) {
            state_stream_ << ",dq" << (i + 1);
            action_stream_ << ",dq" << (i + 1);
        }
        for (int i = 0; i < kDof; ++i) {
            state_stream_ << ",ddq" << (i + 1);
            action_stream_ << ",ddq" << (i + 1);
        }
        for (int i = 0; i < kDof; ++i) {
            state_stream_ << ",tau" << (i + 1);
            action_stream_ << ",tau" << (i + 1);
        }
        state_stream_ << "\n";
        action_stream_ << "\n";

        write_metadata_file("recording");
    }

    void write_metadata_file(const std::string & status) const
    {
        std::ofstream meta_stream(fs::path(episode_dir_) / "meta.json", std::ios::out);
        if (!meta_stream.is_open()) {
            throw std::runtime_error("Failed to open meta.json for write.");
        }

        meta_stream << "{\n";
        meta_stream << "  \"task\": " << quote_json(current_episode_task_name_) << ",\n";
        meta_stream << "  \"image_topic\": " << quote_json(image_topic_) << ",\n";
        meta_stream << "  \"state_topic\": " << quote_json(state_topic_) << ",\n";
        meta_stream << "  \"action_topic\": " << quote_json(action_topic_) << ",\n";
        meta_stream << "  \"event_topic\": " << quote_json(event_topic_) << ",\n";
        meta_stream << "  \"status\": " << quote_json(status) << ",\n";
        meta_stream << "  \"close_reason\": " << quote_json(current_close_reason_) << ",\n";
        meta_stream << "  \"write_mp4\": " << (write_mp4_ ? "true" : "false") << ",\n";
        meta_stream << "  \"output_fps\": " << std::fixed << std::setprecision(3) << output_fps_
                    << ",\n";
        meta_stream << "  \"mp4_codec\": " << quote_json(mp4_codec_) << ",\n";
        meta_stream << "  \"frame_count\": " << frame_count_ << ",\n";
        meta_stream << "  \"state_count\": " << state_count_ << ",\n";
        meta_stream << "  \"action_count\": " << action_count_ << ",\n";
        meta_stream << "  \"start_time_sec\": " << std::fixed << std::setprecision(6)
                    << current_episode_start_time_sec_ << ",\n";
        meta_stream << "  \"end_time_sec\": " << std::fixed << std::setprecision(6)
                    << current_episode_end_time_sec_ << "\n";
        meta_stream << "}\n";
    }

    void reset_episode_counters_locked()
    {
        frame_count_ = 0U;
        state_count_ = 0U;
        action_count_ = 0U;
        current_episode_start_time_sec_ = this->now().seconds();
        current_episode_end_time_sec_ = 0.0;
        current_close_reason_.clear();
    }

    void close_streams_locked()
    {
        if (video_writer_.isOpened()) {
            video_writer_.release();
        }
        if (rgb_timestamp_stream_.is_open()) {
            rgb_timestamp_stream_.flush();
            rgb_timestamp_stream_.close();
        }
        if (state_stream_.is_open()) {
            state_stream_.flush();
            state_stream_.close();
        }
        if (action_stream_.is_open()) {
            action_stream_.flush();
            action_stream_.close();
        }
    }

    void begin_episode(const std::string & requested_task_name, const std::string & source)
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        if (recording_active_) {
            finalize_episode_locked("restart");
        }

        current_episode_task_name_ =
            sanitize_task_name(requested_task_name.empty() ? task_name_ : requested_task_name);
        prepare_episode_directory(current_episode_task_name_);
        reset_episode_counters_locked();
        open_output_files();
        recording_active_ = true;

        RCLCPP_INFO(
            this->get_logger(),
            "Episode recording started via %s. task=%s episode_dir=%s",
            source.c_str(),
            current_episode_task_name_.c_str(),
            episode_dir_.c_str());
    }

    void finalize_episode_locked(const std::string & close_reason)
    {
        current_episode_end_time_sec_ = this->now().seconds();
        current_close_reason_ = close_reason;
        write_metadata_file("closed");
        close_streams_locked();
        recording_active_ = false;

        RCLCPP_INFO(
            this->get_logger(),
            "Episode recording closed. reason=%s frames=%zu state_msgs=%zu action_msgs=%zu episode_dir=%s",
            close_reason.c_str(),
            frame_count_,
            state_count_,
            action_count_,
            episode_dir_.c_str());
    }

    void event_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        if (!msg) {
            return;
        }

        const std::string text = trim_copy(msg->data);
        if (text.empty()) {
            return;
        }

        std::string command = text;
        std::string argument;
        const auto separator_pos = text.find_first_of(": ");
        if (separator_pos != std::string::npos) {
            command = trim_copy(text.substr(0, separator_pos));
            argument = trim_copy(text.substr(separator_pos + 1U));
        }

        if (command == "start") {
            begin_episode(argument.empty() ? task_name_ : argument, "event");
            return;
        }

        std::lock_guard<std::mutex> lock(io_mutex_);
        if (!recording_active_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Received episode event '%s' while recorder is idle.",
                command.c_str());
            return;
        }

        if (command == "stop" || command == "success" || command == "fail") {
            finalize_episode_locked(command);
            return;
        }

        RCLCPP_WARN_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Unknown episode event: %s",
            text.c_str());
    }

    void state_callback(const robot_msgs::msg::RobotState::SharedPtr msg)
    {
        write_robot_state_row(msg, state_stream_, state_count_, "state");
    }

    void action_callback(const robot_msgs::msg::RobotState::SharedPtr msg)
    {
        write_robot_state_row(msg, action_stream_, action_count_, "action");
    }

    void write_robot_state_row(
        const robot_msgs::msg::RobotState::SharedPtr & msg,
        std::ofstream & stream,
        std::size_t & counter,
        const char * label)
    {
        if (!msg || msg->motor_state.size() < static_cast<std::size_t>(kDof)) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Dataset recorder ignored short %s message.",
                label);
            return;
        }

        std::lock_guard<std::mutex> lock(io_mutex_);
        if (!recording_active_) {
            return;
        }

        const double t = this->now().seconds();
        stream << std::fixed << std::setprecision(6) << t;
        for (int i = 0; i < kDof; ++i) {
            stream << "," << msg->motor_state[static_cast<std::size_t>(i)].q;
        }
        for (int i = 0; i < kDof; ++i) {
            stream << "," << msg->motor_state[static_cast<std::size_t>(i)].dq;
        }
        for (int i = 0; i < kDof; ++i) {
            stream << "," << msg->motor_state[static_cast<std::size_t>(i)].ddq;
        }
        for (int i = 0; i < kDof; ++i) {
            stream << "," << msg->motor_state[static_cast<std::size_t>(i)].tau_est;
        }
        stream << "\n";
        ++counter;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!msg) {
            return;
        }

        if (!is_color_encoding_supported(msg->encoding)) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "Dataset recorder only supports rgb8/bgr8/rgba8/bgra8 for MP4 output. Got %s.",
                msg->encoding.c_str());
            return;
        }

        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
        } catch (const cv_bridge::Exception & error) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Failed to convert image for dataset recorder: %s",
                error.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        if (frame.empty()) {
            return;
        }

        cv::Mat bgr_frame;
        if (msg->encoding == "rgb8") {
            cv::cvtColor(frame, bgr_frame, cv::COLOR_RGB2BGR);
        } else if (msg->encoding == "rgba8") {
            cv::cvtColor(frame, bgr_frame, cv::COLOR_RGBA2BGR);
        } else if (msg->encoding == "bgra8") {
            cv::cvtColor(frame, bgr_frame, cv::COLOR_BGRA2BGR);
        } else {
            bgr_frame = frame;
        }

        std::lock_guard<std::mutex> lock(io_mutex_);
        if (!recording_active_) {
            return;
        }

        const double t = this->now().seconds();
        if (write_mp4_) {
            ensure_video_writer_ready(bgr_frame);
            if (video_writer_.isOpened()) {
                video_writer_.write(bgr_frame);
            }
        }

        rgb_timestamp_stream_ << frame_count_ << "," << std::fixed << std::setprecision(6) << t
                              << "\n";
        ++frame_count_;
    }

    void ensure_video_writer_ready(const cv::Mat & frame)
    {
        if (video_writer_.isOpened()) {
            return;
        }

        const std::string output_path = (fs::path(episode_dir_) / "rgb.mp4").string();
        const int fourcc = resolve_fourcc(mp4_codec_);
        if (!video_writer_.open(output_path, fourcc, output_fps_, frame.size(), true)) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Failed to open MP4 writer at %s using codec %s",
                output_path.c_str(),
                mp4_codec_.c_str());
            return;
        }
    }

    int resolve_fourcc(const std::string & codec) const
    {
        if (codec.size() != 4U) {
            return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        }
        return cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
    }

    std::mutex io_mutex_;
    bool finalized_{false};
    bool recording_active_{false};
    std::size_t frame_count_{0U};
    std::size_t state_count_{0U};
    std::size_t action_count_{0U};
    double current_episode_start_time_sec_{0.0};
    double current_episode_end_time_sec_{0.0};

    std::string dataset_root_;
    std::string task_name_;
    std::string current_episode_task_name_;
    std::string current_close_reason_;
    std::string image_topic_;
    std::string state_topic_;
    std::string action_topic_;
    std::string event_topic_;
    std::string episode_dir_;
    bool auto_start_{false};
    double output_fps_{30.0};
    bool write_mp4_{true};
    std::string mp4_codec_{"mp4v"};
    bool overwrite_episode_{false};

    std::ofstream rgb_timestamp_stream_;
    std::ofstream state_stream_;
    std::ofstream action_stream_;
    cv::VideoWriter video_writer_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr state_sub_;
    rclcpp::Subscription<robot_msgs::msg::RobotState>::SharedPtr action_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr event_sub_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    std::shared_ptr<ImitationDatasetRecorderNode> node;
    int rc = 0;

    try {
        node = std::make_shared<ImitationDatasetRecorderNode>();
        rclcpp::spin(node);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "imitation_dataset_recorder_node failed: %s\n", error.what());
        rc = 1;
    }

    if (node) {
        node->finalize();
        node.reset();
    }

    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }

    return rc;
}
