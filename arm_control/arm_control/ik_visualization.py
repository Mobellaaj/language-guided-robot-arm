"""Draw a candidate URDF pose as expiring RViz markers, without commanding joints."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xacro
from ament_index_python.packages import get_package_share_directory
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray


def rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew @ skew)


def origin_matrix(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = [float(v) for v in element.get('xyz', '0 0 0').split()]
        roll, pitch, yaw = [float(v) for v in element.get('rpy', '0 0 0').split()]
        result[:3, :3] = (rotation([0, 0, 1], yaw)
                          @ rotation([0, 1, 0], pitch)
                          @ rotation([1, 0, 0], roll))
    return result


def quaternion_from_matrix(matrix):
    """Return x,y,z,w, including stable handling of rotations near 180 degrees."""
    trace = np.trace(matrix)
    if trace > 0:
        s = math.sqrt(trace + 1.0)*2
        q = [(matrix[2, 1]-matrix[1, 2])/s,
             (matrix[0, 2]-matrix[2, 0])/s,
             (matrix[1, 0]-matrix[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(max(0.0, 1+matrix[i, i]-matrix[j, j]-matrix[k, k]))*2
        q = [0.0]*4
        q[i] = s/4
        q[j] = (matrix[j, i]+matrix[i, j])/s
        q[k] = (matrix[k, i]+matrix[i, k])/s
        q[3] = (matrix[k, j]-matrix[j, k])/s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


class PreviewModel:
    def __init__(self):
        share = Path(get_package_share_directory('arm_description'))
        self.robot = ET.fromstring(
            xacro.process_file(str(share/'urdf/arm.urdf.xacro')).toxml())
        children = {j.find('child').get('link') for j in self.robot.findall('joint')}
        self.root = next(link.get('name') for link in self.robot.findall('link')
                         if link.get('name') not in children)

    def link_poses(self, angles, gripper=0.035):
        # The preview uses an open gripper to make the working point visible.
        values = dict(zip(['base_rotation_joint', 'shoulder_joint',
                           'elbow_joint', 'wrist_pitch_joint'], angles))
        values['gripper_opening_joint'] = gripper
        poses = {self.root: np.eye(4)}
        remaining = list(self.robot.findall('joint'))
        while remaining:
            progress = False
            for joint in remaining[:]:
                parent = joint.find('parent').get('link')
                if parent not in poses:
                    continue
                value = values.get(joint.get('name'), 0.0)
                mimic = joint.find('mimic')
                if mimic is not None:
                    value = (values.get(mimic.get('joint'), 0.0)
                             * float(mimic.get('multiplier', '1'))
                             + float(mimic.get('offset', '0')))
                motion = np.eye(4)
                if joint.get('type') in ('revolute', 'continuous', 'prismatic'):
                    axis_element = joint.find('axis')
                    axis = [float(v) for v in (
                        axis_element.get('xyz') if axis_element is not None else '1 0 0').split()]
                    if joint.get('type') == 'prismatic':
                        motion[:3, 3] = np.array(axis)*value
                    else:
                        motion[:3, :3] = rotation(axis, value)
                poses[joint.find('child').get('link')] = (
                    poses[parent] @ origin_matrix(joint.find('origin')) @ motion)
                remaining.remove(joint)
                progress = True
            if not progress:
                raise ValueError('Could not resolve the URDF joint tree')
        base_inverse = np.linalg.inv(poses['base_link'])
        return {name: base_inverse @ pose for name, pose in poses.items()}

    def markers(self, angles, target, stamp):
        poses = self.link_poses(angles)
        # This independently checks the solver against URDF joint transforms.
        if not np.allclose(poses['tool0'][:3, 3], target, atol=1e-6, rtol=0):
            raise ValueError('IK target does not match URDF tool0; check solver dimensions')
        if not np.allclose(poses['tool0'][:3, 2], [0, 0, -1], atol=1e-6):
            raise ValueError('URDF tool0 is not pointing down')
        return self.pose_markers(angles, stamp)

    def pose_markers(self, angles, stamp, gripper=0.035):
        """Render arbitrary joint poses, including intermediate movement samples."""
        poses = self.link_poses(angles, gripper)
        output = MarkerArray()
        for link in self.robot.findall('link'):
            for visual in link.findall('visual'):
                geometry = visual.find('geometry')
                marker = Marker()
                marker.header.frame_id = 'base_link'
                marker.header.stamp = stamp
                marker.ns = 'ik_candidate'
                marker.id = len(output.markers)
                marker.action = Marker.ADD
                if geometry.find('box') is not None:
                    marker.type = Marker.CUBE
                    dimensions = [float(v) for v in geometry.find('box').get('size').split()]
                elif geometry.find('cylinder') is not None:
                    marker.type = Marker.CYLINDER
                    cylinder = geometry.find('cylinder')
                    dimensions = [2*float(cylinder.get('radius'))]*2 + [float(cylinder.get('length'))]
                elif geometry.find('sphere') is not None:
                    marker.type = Marker.SPHERE
                    dimensions = [2*float(geometry.find('sphere').get('radius'))]*3
                else:
                    raise ValueError('Preview currently supports box, cylinder, and sphere visuals')
                pose = poses[link.get('name')] @ origin_matrix(visual.find('origin'))
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, pose[:3, 3])
                q = quaternion_from_matrix(pose[:3, :3])
                marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = map(float, q)
                marker.scale.x, marker.scale.y, marker.scale.z = dimensions
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 0.9, 1.0, 0.35
                marker.lifetime = Duration(seconds=0.8).to_msg()
                output.markers.append(marker)
        return output

    @staticmethod
    def clear():
        marker = Marker()
        marker.action = Marker.DELETEALL
        return MarkerArray(markers=[marker])
