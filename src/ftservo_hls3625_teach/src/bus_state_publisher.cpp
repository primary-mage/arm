#include <chrono>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include "ftservo_hls3625_teach/servo_bus.hpp"

namespace ftservo_hls3625_teach
{

class BusStatePublisher : public rclcpp::Node
{
public:
  BusStatePublisher()
  : Node("bus_state_publisher"),
    config_(declare_bus_config(*this)),
    bus_(config_)
  {
    position_filter_alpha_ = declare_parameter<double>("position_low_pass_alpha", 0.25);
    velocity_filter_alpha_ = declare_parameter<double>("velocity_low_pass_alpha", 0.25);
    angle_window_lower_ = declare_parameter<std::vector<double>>(
      "angle_window_lower",
      std::vector<double>{-kPi, -0.5, -3.0, -2.0, -kPi});
    angle_window_upper_ = declare_parameter<std::vector<double>>(
      "angle_window_upper",
      std::vector<double>{kPi, 3.5, 0.15, 1.57, kPi});
    position_filter_alpha_ = clamp_filter_alpha(position_filter_alpha_, "position_low_pass_alpha");
    velocity_filter_alpha_ = clamp_filter_alpha(velocity_filter_alpha_, "velocity_low_pass_alpha");
    normalize_angle_windows();
    filtered_positions_.assign(config_.ids.size(), 0.0);
    filtered_velocities_.assign(config_.ids.size(), 0.0);
    last_wrapped_positions_.assign(config_.ids.size(), 0.0);
    last_unwrapped_positions_.assign(config_.ids.size(), 0.0);

    bus_.ping_all(get_logger());

    publisher_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / config_.publish_hz));
    timer_ = create_wall_timer(period, std::bind(&BusStatePublisher::publish_once, this));

    RCLCPP_INFO(
      get_logger(),
      "Publishing %zu calibrated joints on /joint_states with low-pass filter alpha(pos)=%.2f alpha(vel)=%.2f",
      config_.ids.size(),
      position_filter_alpha_,
      velocity_filter_alpha_);
  }

private:
  static constexpr double kPi = 3.14159265358979323846;
  static constexpr double kTwoPi = 2.0 * kPi;

  static double apply_low_pass(double previous, double sample, double alpha)
  {
    return previous + alpha * (sample - previous);
  }

  static double normalize_angle_to_pi(double angle)
  {
    if (!std::isfinite(angle)) {
      return angle;
    }

    angle = std::fmod(angle + kPi, kTwoPi);
    if (angle < 0.0) {
      angle += kTwoPi;
    }
    return angle - kPi;
  }

  static double shortest_angular_distance(double from, double to)
  {
    return normalize_angle_to_pi(to - from);
  }

  static double unwrap_continuous_angle(double previous_wrapped, double previous_unwrapped, double wrapped)
  {
    if (!std::isfinite(previous_wrapped) || !std::isfinite(previous_unwrapped)) {
      return wrapped;
    }
    return previous_unwrapped + shortest_angular_distance(previous_wrapped, wrapped);
  }

  void normalize_angle_windows()
  {
    const std::vector<double> default_lower{-kPi, -0.5, -3.0, -2.0, -kPi};
    const std::vector<double> default_upper{kPi, 3.5, 0.15, 1.57, kPi};
    const auto expected_size = config_.ids.size();

    if (angle_window_lower_.size() != expected_size) {
      RCLCPP_WARN(
        get_logger(),
        "Parameter angle_window_lower size=%zu, expected=%zu. Falling back to defaults.",
        angle_window_lower_.size(),
        expected_size);
      angle_window_lower_ = default_lower;
    }

    if (angle_window_upper_.size() != expected_size) {
      RCLCPP_WARN(
        get_logger(),
        "Parameter angle_window_upper size=%zu, expected=%zu. Falling back to defaults.",
        angle_window_upper_.size(),
        expected_size);
      angle_window_upper_ = default_upper;
    }

    for (std::size_t i = 0; i < expected_size; ++i) {
      if (!std::isfinite(angle_window_lower_[i]) ||
          !std::isfinite(angle_window_upper_[i]) ||
          angle_window_lower_[i] >= angle_window_upper_[i])
      {
        RCLCPP_WARN(
          get_logger(),
          "Angle window for joint %zu is invalid. Resetting to default.",
          i + 1);
        angle_window_lower_[i] = default_lower[i];
        angle_window_upper_[i] = default_upper[i];
      }
    }
  }

  static double distance_to_interval(double value, double lower, double upper)
  {
    if (value < lower) {
      return lower - value;
    }
    if (value > upper) {
      return value - upper;
    }
    return 0.0;
  }

  double choose_windowed_angle(std::size_t joint_index, double continuous_angle, double previous_output) const
  {
    const double lower = angle_window_lower_[joint_index];
    const double upper = angle_window_upper_[joint_index];
    const double center = 0.5 * (lower + upper);

    double best_candidate = std::numeric_limits<double>::quiet_NaN();
    double best_rank0 = std::numeric_limits<double>::infinity();
    double best_rank1 = std::numeric_limits<double>::infinity();
    double best_rank2 = std::numeric_limits<double>::infinity();

    for (int turn = -2; turn <= 2; ++turn) {
      const double candidate = continuous_angle + static_cast<double>(turn) * kTwoPi;
      const double rank0 = distance_to_interval(candidate, lower, upper);
      const double rank1 = std::isfinite(previous_output) ? std::abs(candidate - previous_output)
                                                          : std::abs(candidate - center);
      const double rank2 = std::abs(candidate - center);
      const bool is_better =
        (rank0 < best_rank0) ||
        (rank0 == best_rank0 && rank1 < best_rank1) ||
        (rank0 == best_rank0 && rank1 == best_rank1 && rank2 < best_rank2);
      if (is_better) {
        best_candidate = candidate;
        best_rank0 = rank0;
        best_rank1 = rank1;
        best_rank2 = rank2;
      }
    }

    return std::clamp(best_candidate, lower, upper);
  }

  double clamp_filter_alpha(double alpha, const char * param_name)
  {
    if (!std::isfinite(alpha)) {
      RCLCPP_WARN(
        get_logger(),
        "Parameter %s is not finite; falling back to 1.0.",
        param_name);
      return 1.0;
    }
    if (alpha < 0.0 || alpha > 1.0) {
      RCLCPP_WARN(
        get_logger(),
        "Parameter %s=%.3f is outside [0, 1]; clamping.",
        param_name,
        alpha);
      return std::clamp(alpha, 0.0, 1.0);
    }
    return alpha;
  }

  void publish_once()
  {
    try {
      const auto states = bus_.read_states();

      if (states.size() != filtered_positions_.size()) {
        throw std::runtime_error("servo state size changed unexpectedly");
      }

      sensor_msgs::msg::JointState message;
      message.header.stamp = now();
      message.name = config_.joint_names;
      message.position.reserve(states.size());
      message.velocity.reserve(states.size());

      for (std::size_t i = 0; i < states.size(); ++i) {
        const auto & state = states[i];
        const double continuous_position = unwrap_initialized_ ?
          unwrap_continuous_angle(
            last_wrapped_positions_[i], last_unwrapped_positions_[i], state.position_rad) :
          state.position_rad;
        const double windowed_position = choose_windowed_angle(
          i,
          continuous_position,
          filter_initialized_ ? filtered_positions_[i] : std::numeric_limits<double>::quiet_NaN());

        if (!filter_initialized_) {
          filtered_positions_[i] = windowed_position;
          filtered_velocities_[i] = state.speed_rad_s;
        } else {
          filtered_positions_[i] = apply_low_pass(
            filtered_positions_[i], windowed_position, position_filter_alpha_);
          filtered_velocities_[i] = apply_low_pass(
            filtered_velocities_[i], state.speed_rad_s, velocity_filter_alpha_);
        }

        last_wrapped_positions_[i] = state.position_rad;
        last_unwrapped_positions_[i] = continuous_position;

        message.position.push_back(filtered_positions_[i]);
        message.velocity.push_back(filtered_velocities_[i]);
      }
      filter_initialized_ = true;
      unwrap_initialized_ = true;

      publisher_->publish(message);
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "State publish failed: %s",
        error.what());
    }
  }
  BusConfig config_;
  ServoBus bus_;
  bool filter_initialized_{false};
  double position_filter_alpha_{0.25};
  double velocity_filter_alpha_{0.25};
  std::vector<double> angle_window_lower_;
  std::vector<double> angle_window_upper_;
  std::vector<double> filtered_positions_;
  std::vector<double> filtered_velocities_;
  bool unwrap_initialized_{false};
  std::vector<double> last_wrapped_positions_;
  std::vector<double> last_unwrapped_positions_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace ftservo_hls3625_teach

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int rc = 0;

  try {
    auto node = std::make_shared<ftservo_hls3625_teach::BusStatePublisher>();
    rclcpp::spin(node);
  } catch (const std::exception & error) {
    fprintf(stderr, "bus_state_publisher failed: %s\n", error.what());
    rc = 1;
  }

  rclcpp::shutdown();
  return rc;
}
