"""Detect and localize three colors using one known cube-top plane."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    actions = [DeclareLaunchArgument(
        'plane_height', default_value='0.04',
        description='Shared cube top-surface height in world coordinates, metres')]
    for color in ('red', 'green', 'blue'):
        actions.extend([
            Node(
                package='arm_perception', executable='red_detector',
                name=f'{color}_detector', output='screen',
                parameters=[{'use_sim_time': True, 'color': color,
                             'image_topic': '/overhead_camera/image_raw',
                             # Front workspace in the fixed 640x480 overhead view;
                             # excludes the blue arm at its initial upright pose.
                             'image_roi': [0, 0, 640, 300]}],
            ),
            Node(
                package='arm_perception', executable='red_localizer',
                name=f'{color}_localizer', output='screen',
                parameters=[{
                    'use_sim_time': True, 'color': color,
                    'camera_info_topic': '/overhead_camera/camera_info',
                    'plane_frame': 'world', 'target_frame': 'base_link',
                    'plane_height': ParameterValue(
                        LaunchConfiguration('plane_height'), value_type=float),
                }],
            ),
        ])
    return LaunchDescription(actions)
