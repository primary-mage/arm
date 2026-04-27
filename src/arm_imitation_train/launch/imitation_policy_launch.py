import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = "arm_imitation_train"
    pkg_share = get_package_share_directory(package_name)
    default_config_path = os.path.join(pkg_share, "config", "inference_config.yaml")

    params_path_arg = DeclareLaunchArgument(
        "params_path",
        default_value=default_config_path,
        description="Full path to the inference parameters file to use",
    )
    policy_executable_arg = DeclareLaunchArgument(
        "policy_executable",
        default_value="arm_imitation_policy",
        description="Executable name for the policy node",
    )

    policy_node = Node(
        package=package_name,
        executable=LaunchConfiguration("policy_executable"),
        name="arm_imitation_policy_node",
        output="screen",
        parameters=[LaunchConfiguration("params_path")],
        emulate_tty=True,
    )

    return LaunchDescription([params_path_arg, policy_executable_arg, policy_node])
