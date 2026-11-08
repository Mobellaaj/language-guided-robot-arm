"""Launch Gazebo, RViz and multiple-object perception together."""
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    gazebo = Path(get_package_share_directory('arm_gazebo'))
    perception = Path(get_package_share_directory('arm_perception'))
    # One geometry source: pass the actual world platform to perception and control.
    world = ET.parse(gazebo / 'worlds/arm_world.sdf')
    platform = world.find("world/model[@name='placement_platform']")
    pose = [float(v) for v in platform.findtext('pose').split()]
    size = [float(v) for v in platform.findtext('link/collision/geometry/box/size').split()]
    if any(abs(v) > 1e-8 for v in pose[3:]):
        raise ValueError('This exercise expects an axis-aligned placement platform')
    cube = [float(v) for v in world.findtext(
        "world/model[@name='red_block']/link/visual/geometry/box/size").split()]
    if max(cube)-min(cube) > 1e-8:
        raise ValueError('This exercise expects cubic blocks')
    geometry = dict(platform_x=pose[0], platform_y=pose[1], platform_size_x=size[0],
                    platform_size_y=size[1], platform_top=pose[2]+size[2]/2, cube_size=cube[0])
    floor = world.find("world/model[@name='floor_placement_area']")
    floor_pose = [float(v) for v in floor.findtext('pose').split()]
    floor_size = [float(v) for v in floor.findtext('link/visual/geometry/plane/size').split()]
    if any(abs(v) > 1e-8 for v in floor_pose[3:]):
        raise ValueError('The floor placement area must be axis-aligned.')
    geometry.update(floor_x=floor_pose[0], floor_y=floor_pose[1],
                    floor_size_x=floor_size[0], floor_size_y=floor_size[1])
    obstacle_boxes = []
    for obstacle in world.findall("world/model[@name='planning_obstacle']"):
        position = [float(v) for v in obstacle.findtext('pose').split()]
        dimensions = [float(v) for v in obstacle.findtext('link/collision/geometry/box/size').split()]
        if obstacle.findtext('static') != 'true':
            raise ValueError('The known planning obstacle must be static.')
        if any(abs(v) > 1e-8 for v in position[3:]):
            raise ValueError('The planning exercise expects an axis-aligned fixed obstacle.')
        obstacle_boxes.extend(position[:3]+dimensions)
    motion_parameters = {'use_sim_time': True, **geometry}
    if obstacle_boxes:
        motion_parameters['obstacle_boxes'] = obstacle_boxes
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        DeclareLaunchArgument('motion', default_value='true',
                              description='Start the idle target-selection and motion node'),
        DeclareLaunchArgument('plane_height', default_value='0.04',
                              description='Shared cube top-surface height in world, metres'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gazebo / 'launch/gazebo.launch.py')),
            launch_arguments={'rviz': LaunchConfiguration('rviz')}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(perception / 'launch/objects.launch.py')),
            launch_arguments={'plane_height': LaunchConfiguration('plane_height'),
                              **{name: str(value) for name,value in geometry.items()}}.items()),
        Node(package='arm_control', executable='move_to_color', output='screen',
             parameters=[motion_parameters],
             condition=IfCondition(LaunchConfiguration('motion'))),
    ])
