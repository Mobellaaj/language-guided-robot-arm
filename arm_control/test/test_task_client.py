"""Bounded preparation retries must never retry execution or RPC failures."""
from types import SimpleNamespace

import pytest
from arm_control import arm_command


def client(monkeypatch, replies):
    clock = [0.]
    calls = []
    responses = iter(replies)

    def call(request):
        calls.append(request.task_json)
        return next(responses)

    monkeypatch.setattr(arm_command.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(arm_command.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    monkeypatch.setattr(arm_command, 'await_response', lambda node, future: future)
    return SimpleNamespace(call_async=call), calls, clock


def reply(success=False, retryable=False):
    return SimpleNamespace(success=success, retryable=retryable)


def test_waits_for_readiness_then_prepares_once_without_executing(monkeypatch):
    ready = reply(success=True)
    service, calls, _ = client(monkeypatch, [reply(retryable=True), reply(retryable=True), ready])
    assert arm_command.prepare_task(None, service, '{}') is ready
    assert calls == ['{}']*3


def test_readiness_wait_has_a_wall_clock_deadline(monkeypatch):
    unavailable = reply(retryable=True)
    service, calls, clock = client(monkeypatch, [unavailable]*100)
    assert arm_command.prepare_task(None, service, '{}', readiness_timeout=.25) is unavailable
    assert clock[0] == pytest.approx(.25)
    assert len(calls) == 3


def test_invalid_or_ambiguous_requests_return_immediately(monkeypatch):
    rejected = reply()
    service, calls, clock = client(monkeypatch, [rejected])
    assert arm_command.prepare_task(None, service, '{}') is rejected
    assert len(calls) == 1 and clock[0] == 0.


def test_an_rpc_timeout_is_never_automatically_retried(monkeypatch):
    service, calls, _ = client(monkeypatch, [None])

    def timeout(node, future):
        raise RuntimeError('Response timed out; request may have been accepted.')

    monkeypatch.setattr(arm_command, 'await_response', timeout)
    with pytest.raises(RuntimeError, match='timed out'):
        arm_command.prepare_task(None, service, '{}')
    assert len(calls) == 1
