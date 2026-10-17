from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    package_share = Path(get_package_share_directory('arm_description'))
    robot_description = xacro.process_file(
        str(package_share / 'urdf' / 'arm.urdf.xacro')
    ).toxml()

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='Show the joint angle sliders'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open the RViz view'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(LaunchConfiguration('gui')),
            output='screen',
        ),
        # A non-GUI publisher lets us verify the model without opening windows.
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            condition=UnlessCondition(LaunchConfiguration('gui')),
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', str(package_share / 'rviz' / 'arm.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ])
