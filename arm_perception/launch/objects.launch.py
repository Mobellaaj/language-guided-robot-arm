"""All-color region detection and surface positions; no motion node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from arm_perception.platform_geometry import DEFAULTS


def generate_launch_description():
    geometry = {name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in DEFAULTS}
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=str(value)) for name, value in DEFAULTS.items()],
        DeclareLaunchArgument('plane_height', default_value='0.04',
                              description='Shared cube top-surface height in world, metres'),
        Node(package='arm_perception', executable='object_detector', output='screen',
             parameters=[{'use_sim_time': True, 'image_roi': [0, 0, 0, 0]}]),
        Node(package='arm_perception', executable='object_localizer', output='screen',
             parameters=[{**geometry, 'use_sim_time': True, 'plane_height': ParameterValue(
                 LaunchConfiguration('plane_height'), value_type=float)}]),
    ])
