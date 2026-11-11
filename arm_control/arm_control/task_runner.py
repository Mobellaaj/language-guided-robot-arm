"""Coordinate validated tasks using the existing pick and place state machines."""
import json
import math

from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger

from arm_control.target_selection import AmbiguousTarget, choose_object
from arm_control.task_request import ClarificationRequired, resolve_action, validate_task
from arm_control.errors import PerceptionNotReady


class TaskRunner:
    def __init__(self, node):
        self.node = node
        self.state = 'idle'
        self.phase = ''
        self.request = None
        self.source = None
        self.deadline = None
        self.internal = False
        self.message = 'No task prepared.'
        self.last_status = None

    @property
    def active(self):
        return self.state in ('running', 'waiting_for_scene')

    @property
    def busy(self):
        return self.active and not self.internal

    @property
    def prepared(self):
        return self.state == 'prepared'

    def publish(self, state=None, message=None):
        if state is not None:
            self.state = state
        if message is not None:
            self.message = message
        n = self.node
        text = json.dumps(dict(state=self.state, phase=self.phase, task=self.request,
                               message=self.message, held_color=(n.held or {}).get('color'),
                               motion_active=bool(n.active or n.pick.active or n.place.active or n.gripper.active),
                               grasp_uncertain=n.grasp_uncertain))
        if text != self.last_status:
            n.task_status_pub.publish(String(data=text))
            self.last_status = text

    def report(self, state, message):
        self.publish(state, message)
        self.node.set_status(message)

    def discard_prepared(self):
        if self.prepared:
            self.request = None
            self.source = None
            self.phase = ''
            self.publish('idle', 'Previous task preview discarded.')

    def interrupted(self, state):
        # Called before clearing a motion; the next timer must not restart a task.
        self.state = state
        self.deadline = None
        self.internal = False

    def prepare(self, text, response):
        n = self.node
        if self.busy or n.active or n.pick.active or n.place.active or n.gripper.active:
            response.message = 'A task or movement is active. Stop it first.'
            return response
        n.clear_plan()
        n.clear_selection()
        self.request, self.source, self.phase = None, None, ''
        self.deadline = None
        try:
            task = validate_task(text)
            if n.grasp_uncertain:
                raise ValueError(n.grasp_recovery_message())
            task = resolve_action(task, n.held)
            problem = n.objects_problem()
            if problem:
                if n.objects is not None and n.objects.header.frame_id != 'base_link':
                    raise ValueError(problem)
                raise PerceptionNotReady(problem)
            if task['action'] in ('pick', 'pick_and_place'):
                query = task['object']
                color, point = choose_object(n.objects, query['color'], query['selector'],
                                             query['source'], n.layout)
                if task['action'] == 'pick_and_place':
                    getattr(n, task['destination']).available()
                reply = n.set_parameters_atomically([
                    Parameter('target_color', value=color),
                    Parameter('target_side', value=query['selector']),
                ])
                if not reply.successful:
                    raise ValueError(reply.reason)
                result = n.prepare_pick(None, Trigger.Response(), selected=point, source=query['source'])
                task['object']['color'] = color
                self.source = point
            else:
                getattr(n, task['destination']).available()
                result = n.prepare_place(None, Trigger.Response(), destination=task['destination'])
            if not result.success:
                raise ValueError(result.message)
        except PerceptionNotReady as error:
            response.retryable = True
            response.message = str(error)
            self.publish('not_ready', response.message)
            return response
        except (ClarificationRequired, AmbiguousTarget) as error:
            response.needs_clarification = True
            response.message = str(error)
            self.report('clarification_required', response.message)
            return response
        except ValueError as error:
            response.message = str(error)
            self.report('rejected', response.message)
            return response
        self.request = task
        self.phase = 'place' if task['action'] == 'place' else 'pick'
        response.success = True
        response.normalized_json = json.dumps(task)
        response.message = result.message
        if task['action'] == 'pick_and_place':
            response.message = ('Prepared pick_and_place. The current preview shows pickup. '
                                'One execute starts the task; placement is planned, checked and executed '
                                'automatically after pickup, using the measured grasp. '
                                'Execute within 60 simulation seconds.')
        if task['action'] == 'pick_and_place':
            self.report('prepared', response.message)
        else:
            self.publish('prepared', response.message)
        return response

    def check_start(self):
        if self.prepared and self.request['action'] == 'pick_and_place':
            getattr(self.node, self.request['destination']).available()

    def started(self):
        if self.prepared:
            self.report('running', f"Task started: {self.request['action']} ({self.phase}).")

    def fail(self, reason):
        self.node.stop_motion('Task stopped: '+reason)

    def tick(self):
        if not self.active:
            return
        n = self.node
        if n.pick.active or n.place.active or n.gripper.active:
            return
        if self.phase == 'pick':
            if n.held is None or n.grasp_uncertain:
                self.fail('Pickup did not finish with a usable held-object context.')
                return
            if self.request['action'] == 'pick':
                self.report('succeeded', 'Task completed: pick. Holding the closure target.')
                return
            if (n.held['color'] != self.request['object']['color']
                    or math.dist(n.held['source'], self.source) > .015):
                self.fail('The held-object context does not match the selected task target.')
                return
            self.phase = 'handoff'
            self.deadline = n.now_seconds()+3.0
            self.report('waiting_for_scene', f"Pickup completed. Checking the {self.request['destination']} before automatic placement.")
            return
        if self.phase == 'handoff':
            if n.held is None or n.grasp_uncertain:
                self.fail('Held-object context was lost before placement.')
                return
            problem = n.feedback_problem() or n.gripper.hold_problem(n.held['fingers'], n.held['gap'])
            if problem:
                n.grasp_uncertain = True
                self.fail(problem)
                return
            try:
                problem = n.objects_problem()
                if problem:
                    raise ValueError(problem)
                getattr(n, self.request['destination']).available()
            except ValueError as error:
                if n.now_seconds() >= self.deadline:
                    self.fail('Placement is not ready: '+str(error))
                return
            self.internal = True
            try:
                reply = n.prepare_place(None, Trigger.Response(), destination=self.request['destination'])
                if not reply.success:
                    self.fail(reply.message)
                    return
                self.phase = 'place'
                reply = n.execute(None, Trigger.Response())
                if not reply.success:
                    self.fail(reply.message)
                    return
                self.report('running', 'Task: executing the checked placement automatically.')
            finally:
                self.internal = False
            return
        if self.phase == 'place':
            if n.held is not None or n.grasp_uncertain or n.plan is not None:
                self.fail('Placement did not finish normally.')
                return
            self.report('succeeded', 'Task completed: placement verified; arm returned to observation.')
