"""Overhead red detection and localization; start Gazebo separately."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('plane_height', default_value='0.04',
                              description='Object top-surface height in world coordinates, metres'),
        Node(
            package='arm_perception', executable='red_detector', output='screen',
            parameters=[{'use_sim_time': True, 'image_topic': '/overhead_camera/image_raw'}],
        ),
        Node(
            package='arm_perception', executable='red_localizer', output='screen',
            parameters=[{
                'use_sim_time': True,
                'camera_info_topic': '/overhead_camera/camera_info',
                'plane_frame': 'world',
                'target_frame': 'base_link',
                'plane_height': ParameterValue(LaunchConfiguration('plane_height'), value_type=float),
            }],
        ),
    ])
