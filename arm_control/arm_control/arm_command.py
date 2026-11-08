"""Text and JSON task client. Preparing a task never starts movement."""
import argparse
import json
from pathlib import Path
import re
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Trigger
from arm_interfaces.srv import PrepareTask


def parse_target(text):
    text = re.sub(r' from (?:the )?(?:floor|platform)$', '', text.removeprefix('the '))
    match = re.fullmatch(
        r'(?:(leftmost|rightmost|nearest|farthest) )?'
        r'(?:(red|green|blue)(?: (?:cube|block))?|(cube|block))'
        r'(?: (?:on|to) your (left|right))?', text)
    if not match or (match.group(1) and match.group(4)):
        raise ValueError('Specify a supported color/block and one selector.')
    rank, color, _, region = match.groups()
    return color, (region+'_of_robot' if region else rank or 'unique')


def parse_task(text):
    """Parse the supported pick/place phrases into the same JSON task contract."""
    text = ' '.join(text.lower().split())
    destination = re.search(r' (?:on|onto) (?:the )?(floor|platform)$', text)
    target = text[:destination.start()] if destination else text
    action = None
    if destination:
        combined = re.fullmatch(r'pick (.+?) and (?:place|put)(?: it)?', target)
        if combined:
            action, target = 'pick_and_place', combined.group(1)
        else:
            match = re.fullmatch(r'(pick and place|place|put)(?: (.*))?', target)
            if match:
                action = 'pick_and_place' if match.group(1) == 'pick and place' else 'place'
                target = match.group(2) or ''
    elif text == 'pick' or text.startswith('pick '):
        action, target = 'pick', text[4:].strip() or 'block'
    if action is None:
        return None
    task = {'action': action}
    if action != 'place' or target not in ('', 'it'):
        source = re.search(r' from (?:the )?(floor|platform)$', target)
        color, selector = parse_target(target)
        task['object'] = dict(color=color, selector=selector,
                              source=source.group(1) if source else 'floor')
    if destination:
        task['destination'] = destination.group(1)
    return task


def parse_command(text):
    """Manual commands remain explicit; task phrases go through JSON validation."""
    text = ' '.join(text.lower().split())
    gripper = re.fullmatch(r'(open|close) (?:the )?gripper', text)
    if gripper:
        return gripper.group(1), None, None
    if text in ('inspect platform', 'inspect floor'):
        return ('inspect' if text.endswith('platform') else 'inspect_floor'), None, None
    if text == 'stop':
        return 'cancel', None, None
    if text in ('observe', 'execute', 'cancel'):
        return text, None, None
    task = parse_task(text)
    if task:
        query = task.get('object', {})
        return task['action'], query.get('color'), query.get('selector')
    match = re.fullmatch(r'(move above|select) (.+)', text)
    if match:
        if re.search(r' from (?:the )?platform$', match.group(2)):
            raise ValueError('Use a pick task to select a platform cube.')
        color, selector = parse_target(match.group(2))
        if color is None:
            raise ValueError('Specify a color for select or move above.')
        return ('approach' if match.group(1) == 'move above' else 'select'), color, selector
    raise ValueError('Use pick, pick and place COLOR from SURFACE on SURFACE, place on floor/platform, '
                     'select COLOR, move above COLOR, inspect floor/platform, '
                     'open/close gripper, observe, execute, stop, or --json/--file.')


def await_response(node, future):
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done():
        raise RuntimeError('Response timed out. Check the motion node before retrying; '
                           'the request may have been accepted.')
    result = future.result()
    if result is None:
        raise RuntimeError('The motion node returned no response.')
    return result


def run_task(node, text):
    client = node.create_client(PrepareTask, '/arm/prepare_task')
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError('Task service unavailable. Restart the updated combined launch.')
    reply = await_response(node, client.call_async(PrepareTask.Request(task_json=text)))
    if reply.needs_clarification:
        raise RuntimeError('Clarification required: '+reply.message)
    if not reply.success:
        raise RuntimeError(reply.message)
    print('Task JSON: '+reply.normalized_json)
    print(reply.message)
    print('Prepared only: inspect Planned Motion, then run: ros2 run arm_control arm_command execute')


def run_command(node, action, color, side):
    if action in ('pick', 'place', 'pick_and_place'):
        task = {'action': action}
        if action != 'place' or color is not None or side not in (None, 'unique'):
            task['object'] = dict(color=color, selector=side or 'unique', source='floor')
        if action != 'pick':
            task['destination'] = 'platform'
        return run_task(node, json.dumps(task))
    if action in ('approach', 'select'):
        parameters = AsyncParameterClient(node, '/move_to_color')
        if not parameters.wait_for_services(timeout_sec=5.0):
            raise RuntimeError('Start the move_to_color node first.')
        reply = await_response(node, parameters.set_parameters_atomically([
            Parameter('target_color', value=color), Parameter('target_side', value=side)]))
        if not reply.result.successful:
            raise RuntimeError(reply.result.reason or 'Target selection request failed.')
    service = {'approach': 'prepare_move', 'inspect': 'inspect_platform', 'inspect_floor': 'inspect_floor',
               'select': 'select_target',
               'observe': 'prepare_observation', 'execute': 'execute_move', 'cancel': 'cancel_move',
               'open': 'open_gripper', 'close': 'close_gripper'}[action]
    client = node.create_client(Trigger, f'/arm/{service}')
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f'/arm/{service} is unavailable. Start the combined launch.')
    reply = await_response(node, client.call_async(Trigger.Request()))
    if not reply.success:
        raise RuntimeError(reply.message)
    print(reply.message)
    if action in ('approach', 'observe'):
        print('Prepared only: inspect Planned Motion, then run: ros2 run arm_control arm_command execute')


def main(args=None):
    argv = sys.argv if args is None else ['arm_command', *args]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='*', help='For example: "put the rightmost red block on the platform"')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--json', dest='task_json', help='Task JSON as one quoted string')
    inputs.add_argument('--file', type=Path, help='Read task JSON from a file')
    parsed = parser.parse_args(remove_ros_args(args=argv)[1:])
    if sum((bool(parsed.command), parsed.task_json is not None, parsed.file is not None)) != 1:
        parser.error('Provide a text command, --json, or --file.')
    try:
        text = parsed.file.read_text() if parsed.file is not None else parsed.task_json
        task = parse_task(' '.join(parsed.command)) if parsed.command else None
        if task is not None:
            text = json.dumps(task)
        command = parse_command(' '.join(parsed.command)) if parsed.command and task is None else None
    except (ValueError, OSError) as error:
        parser.error(str(error))
    rclpy.init(args=argv[1:])
    node = Node('arm_command')
    code = 0
    try:
        if command is not None:
            run_command(node, *command)
        else:
            run_task(node, text)
    except (RuntimeError, KeyboardInterrupt) as error:
        print(str(error) or 'Interrupted.', file=sys.stderr)
        code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if code:
        raise SystemExit(code)
