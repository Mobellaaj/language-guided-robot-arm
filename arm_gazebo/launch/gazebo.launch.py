from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    package_share = Path(get_package_share_directory('arm_description'))
    gazebo_share = Path(get_package_share_directory('ros_gz_sim'))
    simulation_share = Path(get_package_share_directory('arm_gazebo'))
    robot_description = xacro.process_file(
        str(package_share / 'urdf' / 'arm.urdf.xacro'),
        mappings={'gazebo': 'true'},
    ).toxml()

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gazebo_share / 'launch' / 'gz_sim.launch.py')),
            launch_arguments={
                # No -r: start paused so the initial model can be inspected.
                'gz_args': f'{simulation_share / "worlds" / "arm_world.sdf"} -v 2',
                'on_exit_shutdown': 'true',
            }.items(),
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
            output='screen',
        ),
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-world', 'arm_world', '-name', 'learning_arm',
                       '-topic', '/robot_description', '-allow_renaming', 'false'],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': str(simulation_share / 'config' / 'gazebo_bridge.yaml'),
                         'use_sim_time': True}],
            output='screen',
        ),
    ])
