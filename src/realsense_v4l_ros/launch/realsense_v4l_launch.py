import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_name = "realsense_v4l_ros"
    pkg_share = get_package_share_directory(package_name)
    default_config_path = os.path.join(pkg_share, "config", "realsense_v4l.yaml")

    params_path_arg = DeclareLaunchArgument(
        "params_path",
        default_value=default_config_path,
        description="Full path to the ROS2 parameters file to use",
    )

    camera_node = Node(
        package=package_name,
        executable="realsense_v4l_camera_node",
        name="realsense_v4l_camera_node",
        output="screen",
        parameters=[LaunchConfiguration("params_path")],
        emulate_tty=True,
    )

    return LaunchDescription([
        params_path_arg,
        camera_node,
    ])
