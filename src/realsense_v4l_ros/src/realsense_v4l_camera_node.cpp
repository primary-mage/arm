#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <memory>
#include <sstream>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

using namespace std::chrono_literals;

namespace
{

constexpr char kDefaultFrameId[] = "camera_color_optical_frame";

int fourcc_from_string(const std::string & value)
{
    if (value.size() != 4U) {
        return 0;
    }
    return cv::VideoWriter::fourcc(value[0], value[1], value[2], value[3]);
}

std::string to_lower_copy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string trim_nul_terminated(const char * text, std::size_t size)
{
    std::size_t length = 0U;
    while (length < size && text[length] != '\0') {
        ++length;
    }
    return std::string(text, length);
}

std::string fourcc_to_string(__u32 pixel_format)
{
    std::string result(4, '\0');
    result[0] = static_cast<char>(pixel_format & 0xFF);
    result[1] = static_cast<char>((pixel_format >> 8) & 0xFF);
    result[2] = static_cast<char>((pixel_format >> 16) & 0xFF);
    result[3] = static_cast<char>((pixel_format >> 24) & 0xFF);
    return result;
}

struct VideoDeviceInfo
{
    std::string path;
    std::string card_name;
    std::string bus_info;
    std::string pixel_format;
    uint32_t width{0U};
    uint32_t height{0U};
    bool has_video_capture{false};
};

}  // namespace

class RealsenseV4LCameraNode : public rclcpp::Node
{
public:
    RealsenseV4LCameraNode() : Node("realsense_v4l_camera_node")
    {
        video_device_ = this->declare_parameter("video_device", std::string("/dev/video6"));
        target_camera_name_ =
            this->declare_parameter("target_camera_name", std::string("Intel(R) RealSense(TM)"));
        auto_select_device_ = this->declare_parameter("auto_select_device", true);
        preferred_pixel_formats_ = this->declare_parameter<std::vector<std::string>>(
            "preferred_pixel_formats",
            std::vector<std::string>{"YUYV", "MJPG", "RGB3", "BGR3"});
        image_topic_ = this->declare_parameter("image_topic", std::string("/camera/color/image_raw"));
        camera_info_topic_ =
            this->declare_parameter("camera_info_topic", std::string("/camera/color/camera_info"));
        frame_id_ = this->declare_parameter("frame_id", std::string(kDefaultFrameId));
        width_ = this->declare_parameter("width", 640);
        height_ = this->declare_parameter("height", 480);
        fps_ = this->declare_parameter("fps", 30.0);
        fourcc_ = this->declare_parameter("fourcc", std::string("YUYV"));
        convert_to_rgb_ = this->declare_parameter("convert_to_rgb", true);
        if (width_ <= 0) {
            width_ = 640;
        }
        if (height_ <= 0) {
            height_ = 480;
        }
        if (fps_ <= 0.0) {
            fps_ = 30.0;
        }

        auto qos = rclcpp::SensorDataQoS();
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(image_topic_, qos);
        camera_info_pub_ =
            this->create_publisher<sensor_msgs::msg::CameraInfo>(camera_info_topic_, qos);

        open_camera();

        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / fps_));
        capture_timer_ = this->create_wall_timer(period, std::bind(&RealsenseV4LCameraNode::capture_once, this));

        RCLCPP_INFO(
            this->get_logger(),
            "RealSense V4L camera publisher started. device=%s image_topic=%s camera_info_topic=%s frame_id=%s requested=%dx%d@%.1f fourcc=%s",
            video_device_.c_str(),
            image_topic_.c_str(),
            camera_info_topic_.c_str(),
            frame_id_.c_str(),
            width_,
            height_,
            fps_,
            fourcc_.c_str());
    }

private:
    static bool query_video_device(const std::string & path, VideoDeviceInfo & info)
    {
        const int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) {
            return false;
        }

        v4l2_capability capability{};
        if (ioctl(fd, VIDIOC_QUERYCAP, &capability) != 0) {
            ::close(fd);
            return false;
        }

        info.path = path;
        info.card_name = trim_nul_terminated(reinterpret_cast<char *>(capability.card), sizeof(capability.card));
        info.bus_info = trim_nul_terminated(reinterpret_cast<char *>(capability.bus_info), sizeof(capability.bus_info));

        const uint32_t device_caps =
            capability.device_caps != 0U ? capability.device_caps : capability.capabilities;
        info.has_video_capture =
            (device_caps & V4L2_CAP_VIDEO_CAPTURE) != 0U ||
            (device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) != 0U;

        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd, VIDIOC_G_FMT, &format) == 0) {
            info.pixel_format = fourcc_to_string(format.fmt.pix.pixelformat);
            info.width = format.fmt.pix.width;
            info.height = format.fmt.pix.height;
        }

        ::close(fd);
        return true;
    }

    std::vector<VideoDeviceInfo> enumerate_video_devices() const
    {
        std::vector<VideoDeviceInfo> devices;
        for (const auto & entry : std::filesystem::directory_iterator("/dev")) {
            const std::string filename = entry.path().filename().string();
            if (filename.rfind("video", 0) != 0) {
                continue;
            }

            VideoDeviceInfo info;
            if (query_video_device(entry.path().string(), info)) {
                devices.push_back(info);
            }
        }

        std::sort(
            devices.begin(),
            devices.end(),
            [](const VideoDeviceInfo & lhs, const VideoDeviceInfo & rhs) {
                return lhs.path < rhs.path;
            });
        return devices;
    }

    bool matches_target_camera_name(const VideoDeviceInfo & info) const
    {
        if (target_camera_name_.empty()) {
            return true;
        }
        return to_lower_copy(info.card_name).find(to_lower_copy(target_camera_name_)) != std::string::npos;
    }

    int preferred_pixel_format_rank(const std::string & pixel_format) const
    {
        for (std::size_t index = 0; index < preferred_pixel_formats_.size(); ++index) {
            if (pixel_format == preferred_pixel_formats_[index]) {
                return static_cast<int>(index);
            }
        }
        return static_cast<int>(preferred_pixel_formats_.size()) + 100;
    }

    void auto_select_video_device()
    {
        const auto devices = enumerate_video_devices();
        if (devices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No V4L video devices found under /dev/video*.");
            return;
        }

        std::ostringstream device_summary;
        device_summary << "Detected V4L devices:";
        for (const auto & device : devices) {
            device_summary << " [" << device.path << " card=" << device.card_name
                           << " format=" << device.pixel_format
                           << " size=" << device.width << "x" << device.height
                           << " capture=" << (device.has_video_capture ? "true" : "false") << "]";
        }
        RCLCPP_INFO(this->get_logger(), "%s", device_summary.str().c_str());

        std::vector<VideoDeviceInfo> candidates;
        for (const auto & device : devices) {
            if (!device.has_video_capture) {
                continue;
            }
            if (!matches_target_camera_name(device)) {
                continue;
            }
            candidates.push_back(device);
        }

        if (candidates.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "No V4L device matched target_camera_name='%s'. Falling back to configured video_device=%s",
                target_camera_name_.c_str(),
                video_device_.c_str());
            return;
        }

        std::sort(
            candidates.begin(),
            candidates.end(),
            [this](const VideoDeviceInfo & lhs, const VideoDeviceInfo & rhs) {
                const int lhs_rank = preferred_pixel_format_rank(lhs.pixel_format);
                const int rhs_rank = preferred_pixel_format_rank(rhs.pixel_format);
                if (lhs_rank != rhs_rank) {
                    return lhs_rank < rhs_rank;
                }

                const uint64_t lhs_area = static_cast<uint64_t>(lhs.width) * lhs.height;
                const uint64_t rhs_area = static_cast<uint64_t>(rhs.width) * rhs.height;
                if (lhs_area != rhs_area) {
                    return lhs_area > rhs_area;
                }

                return lhs.path < rhs.path;
            });

        video_device_ = candidates.front().path;
        RCLCPP_INFO(
            this->get_logger(),
            "Auto-selected V4L device %s for target_camera_name='%s' (card=%s format=%s size=%ux%u)",
            video_device_.c_str(),
            target_camera_name_.c_str(),
            candidates.front().card_name.c_str(),
            candidates.front().pixel_format.c_str(),
            candidates.front().width,
            candidates.front().height);
    }

    void open_camera()
    {
        if (auto_select_device_) {
            auto_select_video_device();
        }

        capture_.open(video_device_, cv::CAP_V4L2);
        if (!capture_.isOpened()) {
            throw std::runtime_error("Failed to open video device: " + video_device_);
        }

        capture_.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(width_));
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(height_));
        capture_.set(cv::CAP_PROP_FPS, fps_);

        const int fourcc_code = fourcc_from_string(fourcc_);
        if (fourcc_code != 0) {
            capture_.set(cv::CAP_PROP_FOURCC, static_cast<double>(fourcc_code));
        }

        width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double actual_fps = capture_.get(cv::CAP_PROP_FPS);
        if (actual_fps > 0.0) {
            fps_ = actual_fps;
        }
    }

    void capture_once()
    {
        cv::Mat frame;
        if (!capture_.read(frame) || frame.empty()) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Failed to capture a frame from %s",
                video_device_.c_str());
            return;
        }

        cv::Mat bgr_frame;
        if (frame.type() == CV_8UC1) {
            cv::cvtColor(frame, bgr_frame, cv::COLOR_GRAY2BGR);
        } else if (frame.type() == CV_8UC3) {
            bgr_frame = frame;
        } else if (frame.type() == CV_8UC2 && convert_to_rgb_) {
            cv::cvtColor(frame, bgr_frame, cv::COLOR_YUV2BGR_YUYV);
        } else if (frame.type() == CV_8UC4) {
            cv::cvtColor(frame, bgr_frame, cv::COLOR_BGRA2BGR);
        } else {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Unsupported OpenCV frame type %d from %s",
                frame.type(),
                video_device_.c_str());
            return;
        }

        const auto stamp = this->now();

        std_msgs::msg::Header header;
        header.stamp = stamp;
        header.frame_id = frame_id_;

        auto image_msg =
            cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, bgr_frame).toImageMsg();
        image_pub_->publish(*image_msg);

        sensor_msgs::msg::CameraInfo camera_info;
        camera_info.header = header;
        camera_info.height = static_cast<uint32_t>(bgr_frame.rows);
        camera_info.width = static_cast<uint32_t>(bgr_frame.cols);
        camera_info.distortion_model = "plumb_bob";
        camera_info.d.assign(5, 0.0);
        camera_info.k = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
        camera_info.r = {
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
        camera_info.p = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0};
        camera_info_pub_->publish(camera_info);
    }

    std::string video_device_;
    std::string target_camera_name_;
    bool auto_select_device_{true};
    std::vector<std::string> preferred_pixel_formats_;
    std::string image_topic_;
    std::string camera_info_topic_;
    std::string frame_id_;
    std::string fourcc_;
    int width_{640};
    int height_{480};
    double fps_{30.0};
    bool convert_to_rgb_{true};
    cv::VideoCapture capture_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    rclcpp::TimerBase::SharedPtr capture_timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    int rc = 0;

    try {
        rclcpp::spin(std::make_shared<RealsenseV4LCameraNode>());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "realsense_v4l_camera_node failed: %s\n", error.what());
        rc = 1;
    }

    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }
    return rc;
}
