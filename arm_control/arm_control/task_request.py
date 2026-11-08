"""Strict language-to-task boundary, independent of motion and model inference."""
import json
from functools import lru_cache
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from jsonschema import Draft202012Validator, ValidationError


class ClarificationRequired(ValueError):
    """The request needs a user choice before any movement can be prepared."""


@lru_cache(maxsize=1)
def task_schema():
    path = Path(get_package_share_directory('arm_control'))/'config/task.schema.json'
    return json.loads(path.read_text())


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON field: {key}')
        result[key] = value
    return result


def validate_task(text):
    try:
        task = json.loads(text, object_pairs_hook=unique_fields)
        Draft202012Validator(task_schema()).validate(task)
    except json.JSONDecodeError as error:
        raise ValueError(f'Invalid JSON at line {error.lineno}, column {error.colno}.') from error
    except ValidationError as error:
        field = '.'.join(map(str, error.absolute_path)) or 'task'
        raise ValueError(f'Invalid {field}: {error.message}') from error
    if task['action'] != 'place' or 'object' in task:
        task['object'] = {'color': None, 'selector': 'unique', 'source': 'floor',
                          **task.get('object', {})}
    return task


def resolve_action(task, held):
    """Resolve the meaning of 'place' using actual held-object context."""
    if task['action'] == 'place':
        if held is None:
            if 'object' not in task:
                raise ClarificationRequired('Nothing is held. Which block and source surface should I use?')
            task['action'] = 'pick_and_place'
        elif 'object' in task:
            query = task['object']
            if query['selector'] != 'unique':
                raise ClarificationRequired('Do you mean the held block or a block on the requested surface?')
            if query['color'] is not None and query['color'] != held['color']:
                raise ClarificationRequired(f"I am holding {held['color']}. Should I place that block first?")
    elif held is not None:
        raise ClarificationRequired(f"I am already holding {held['color']}. Should I place it first?")
    return task
