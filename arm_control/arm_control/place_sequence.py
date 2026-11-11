"""Collision-aware, single-layer placement into a camera-confirmed empty slot."""
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from arm_control.ik_solver import forward_kinematics, solve_downward_ik
from arm_control.pick_sequence import PickSequence, vertical_path, path_duration
from arm_control.motion_math import UNLOADED_SPEEDS, UNLOADED_MIN_DURATION
from arm_control.gripper_control import finger_motion_duration, POSITION_SETTLE_TIME


class PlaceSequence(PickSequence):
    operation = 'Place'

    @property
    def manager(self):
        return getattr(self.node, self.program.get('destination', 'platform'))

    def clear(self):
        if self.program is not None:
            self.manager.release_reservation(self.program['slot'])
        super().clear()

    def prepare(self, start, finger, destination='platform'):
        n = self.node
        manager = getattr(n, destination)
        if n.held is None:
            raise ValueError('No held-cube context. Complete a pick before placing.')
        held = dict(n.held)
        problem = n.gripper.hold_problem(held.get('fingers'), held['gap'])
        if problem:
            raise ValueError(problem)
        error = None
        for name in manager.available():
            center = n.layout.placement_slots(destination)[name]
            release_z = center[2]+n.layout.cube/2+held['offset']+0.006
            transfer_z = max(0.15, release_z+0.07, forward_kinematics(start)[2])
            placed = [(center[0], center[1], center[2]+n.layout.cube/2)]
            try:
                raised = vertical_path(start, transfer_z)
                goals = solve_downward_ik(center[0], center[1], transfer_z)
                goal = min(goals, key=lambda q: sum((a-b)**2 for a,b in zip(q,raised[-1])))
                transfer = n.planner.plan(raised[-1], goal, finger, n.objects,
                                          held=held, keep_down=True)
                lower = vertical_path(transfer[-1], release_z)
                retreat = vertical_path(lower[-1], transfer_z)
                home = n.planner.plan(retreat[-1], n.observation_joints, n.gripper.opened,
                                      n.objects, removed=held['source'], extra=placed)
                stages = [
                    dict(name='raise', path=raised, duration=path_duration(raised), finger=finger),
                    dict(name='transfer', path=transfer, duration=path_duration(transfer, minimum=8.0, tool_speed=0.03), finger=finger),
                    dict(name='lower', path=lower, duration=path_duration(lower), finger=finger, tolerance=0.004),
                    dict(name='open', pose=lower[-1], duration=finger_motion_duration(finger,n.gripper.opened)+POSITION_SETTLE_TIME,
                         finger=finger, end_finger=n.gripper.opened),
                    dict(name='retreat', path=retreat, duration=path_duration(retreat), finger=n.gripper.opened),
                    dict(name='observe', path=home, duration=path_duration(
                        home, minimum=UNLOADED_MIN_DURATION, tool_speed=None, joint_speeds=UNLOADED_SPEEDS), finger=n.gripper.opened),
                    dict(name='verify', pose=home[-1], duration=2.0, finger=n.gripper.opened,
                         end_finger=n.gripper.opened),
                ]
                self.validate_stages(stages, held, center)
                break
            except ValueError as exc:
                error = exc
        else:
            raise ValueError(f'No free-slot path passed collision planning: {error}')
        manager.reserve(name)
        self.program = dict(stages=stages, slot=name, held=held, center=center, destination=destination,
                            duration=sum(s['duration'] for s in stages))
        return self.program

    def validate_stages(self, stages, held, center):
        n = self.node
        placed = [(center[0], center[1], center[2]+n.layout.cube/2)]
        for stage in stages:
            carrying = stage['name'] in ('raise', 'transfer', 'lower', 'open')
            payload = held if carrying else None
            extra = [] if carrying else placed
            path = stage['path'] if 'path' in stage else [stage['pose']]
            n.planner.validate(path, stage['finger'], n.objects, held=payload,
                               removed=held['source'], extra=extra, transfer=stage['name'] == 'transfer')
            if stage['name'] == 'open':
                n.planner.validate_fingers(stage['pose'], stage['finger'], stage['end_finger'],
                                           n.objects, held=held)

    def check_current_scene(self):
        # A new observation may invalidate the route inspected in RViz.
        self.validate_stages(self.program['stages'], self.program['held'], self.program['center'])

    def start(self):
        self.closed_positions=self.program['held'].get('fingers',tuple(self.node.gripper.positions))
        self.closed_gap=self.program['held']['gap']
        self.active=True
        self.index=0
        self.begin_stage()

    def begin_stage(self):
        stage=self.program['stages'][self.index]
        if stage['name']=='verify':
            self.elapsed=0.0
            self.stable_since=None
            self.node.set_status('Place: waiting for visible confirmation of the released cube in the reserved slot.')
            return
        super().begin_stage()
        if stage['name']=='open':
            self.manager.released(self.program['slot'])
            self.node.held=None
            # Opening is a release attempt, not confirmation that the fingers are clear.
            self.node.grasp_uncertain=True

    def tick(self,dt):
        if not self.active or dt <= 0:
            return
        n=self.node
        stage=self.program['stages'][self.index]
        if stage['name']=='verify':
            problem=n.feedback_problem()
            if problem:
                self.fail(problem)
                return
            if max(abs(a-b) for a,b in zip(n.measured(),stage['pose'])) > 0.08:
                self.fail('The arm drifted away from the observation pose during verification.')
                return
            self.elapsed+=min(dt,0.1)
            try:
                confirmed=self.manager.verified(self.program['slot'],self.program['held']['color'])
            except ValueError:
                confirmed=False
            if confirmed:
                self.complete()
            elif self.elapsed>8.0:
                self.fail('Release was not visually confirmed. Slot remains reserved; '
                          f'inspect the scene, then use inspect {self.program["destination"]} once visible.')
            return
        if stage['name'] in ('raise','transfer','lower'):
            problem=n.gripper.hold_problem(self.closed_positions,self.closed_gap)
            if problem:
                n.grasp_uncertain=True
                self.fail(problem)
                return
        super().tick(dt)

    def complete(self):
        n=self.node
        name=self.program['slot']
        color=self.program['held']['color']
        destination = self.program['destination']
        self.manager.reserved.pop(name,None)
        center = self.program['center']
        n.planner.remember_placement(self.program['held']['source'],
                                     (center[0], center[1], center[2]+n.layout.cube/2))
        self.active=False
        n.clear_plan()
        n.set_status(f'Placement verified: {color} observed in {destination} {name}. Arm returned to observation. '
                     'The next placement will choose another visibly empty slot.')

    def decorate_preview(self,markers,q,stage):
        # Visualize the carried cube, then its intended resting location after release.
        released=stage['name'] in ('retreat','observe','verify')
        if released:
            x,y,z=self.program['center']
            z+=self.node.layout.cube/2
        else:
            x,y,z=forward_kinematics(q)
            z-=self.program['held']['offset']
        marker=Marker()
        marker.header.frame_id='base_link'
        marker.header.stamp=self.node.get_clock().now().to_msg()
        marker.ns='carried_block'
        marker.id=2000
        marker.type=Marker.CUBE
        marker.pose.position=Point(x=float(x),y=float(y),z=float(z))
        marker.pose.orientation.w=1.0
        marker.scale.x=marker.scale.y=marker.scale.z=self.node.layout.cube
        colors={'red':(1.,0.,0.),'green':(0.,1.,0.),'blue':(0.,.3,1.)}
        marker.color.r,marker.color.g,marker.color.b=colors[self.program['held']['color']]
        marker.color.a=.45
        from rclpy.duration import Duration
        marker.lifetime=Duration(seconds=.8).to_msg()
        markers.markers.append(marker)
