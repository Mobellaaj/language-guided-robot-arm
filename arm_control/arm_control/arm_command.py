"""Small text-command client; preparing a movement never executes it."""
import argparse
import re
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.utilities import remove_ros_args
from std_srvs.srv import Trigger


def parse_command(text):
    """Accept a small explicit vocabulary before introducing an LLM."""
    text = ' '.join(text.lower().split())
    if text in ('observe', 'execute', 'cancel'):
        return text, None
    match = re.fullmatch(r'move above (?:the )?(red|green|blue)(?: (?:cube|block))?', text)
    if match:
        return 'approach', match.group(1)
    raise ValueError('Use: move above red/green/blue, observe, execute, or cancel.')


def await_response(node, future):
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done():
        # The server may still finish a request after this client stops waiting.
        raise RuntimeError('Response timed out. Check the motion node before retrying; '
                           'the request may have been accepted.')
    result = future.result()
    if result is None:
        raise RuntimeError('The motion node returned no response.')
    return result


def run_command(node, action, color):
    if action == 'approach':
        parameters = AsyncParameterClient(node, '/move_to_color')
        if not parameters.wait_for_services(timeout_sec=5.0):
            raise RuntimeError('Start the move_to_color node first.')
        reply = await_response(node, parameters.set_parameters([
            Parameter('target_color', value=color),
        ]))
        if not reply.results or not all(result.successful for result in reply.results):
            reason = '; '.join(result.reason for result in reply.results if not result.successful)
            raise RuntimeError(reason or 'Target-color selection failed.')

    service = {
        'approach': 'prepare_move',
        'observe': 'prepare_observation',
        'execute': 'execute_move',
        'cancel': 'cancel_move',
    }[action]
    client = node.create_client(Trigger, f'/arm/{service}')
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f'/arm/{service} is unavailable. Start the move_to_color node.')
    reply = await_response(node, client.call_async(Trigger.Request()))
    if not reply.success:
        raise RuntimeError(reply.message)
    print(reply.message)
    if action in ('approach', 'observe'):
        print('Prepared only: inspect Planned Motion in RViz, then run: '
              'ros2 run arm_control arm_command execute')


def main(args=None):
    argv = sys.argv if args is None else ['arm_command', *args]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='+', help='For example: "move above blue"')
    parsed = parser.parse_args(remove_ros_args(args=argv)[1:])
    try:
        action, color = parse_command(' '.join(parsed.command))
    except ValueError as error:
        parser.error(str(error))
    rclpy.init(args=argv[1:])
    node = Node('arm_command')
    code = 0
    try:
        run_command(node, action, color)
    except (RuntimeError, KeyboardInterrupt) as error:
        print(str(error) or 'Interrupted.', file=sys.stderr)
        code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if code:
        raise SystemExit(code)
