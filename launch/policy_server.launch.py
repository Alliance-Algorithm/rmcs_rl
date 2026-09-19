from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value="",
                description=(
                    "Deployment YAML containing policy_server.ros__parameters; "
                    "usually the same file passed to rmcs_executor."
                ),
            ),
            Node(
                package="rmcs_rl",
                executable="policy_server",
                parameters=[params_file],
                output="screen",
                respawn=True,
                respawn_delay=1.0,
            ),
        ]
    )
