"""R1–R12 synthetic Core -> generated ROS message tests."""

import ast
from dataclasses import replace
from pathlib import Path

import pytest
from rclpy.serialization import serialize_message, deserialize_message

from amir_irt_mro.local_hold_state_estimator import LocalHoldStateEstimator
from amir_irt_mro.local_hold_mock import mock_inputs
from share_semantic_interfaces.msg import LocalHoldState as RosState
from share_semantic_transport.mapper import SemanticStream, from_message, to_message, UINT64_MAX


@pytest.fixture
def pipeline():
    estimator = LocalHoldStateEstimator()
    stream = SemanticStream('User1')

    def step(ms, **kwargs):
        return stream.encode(estimator.update(mock_inputs(ms * 1_000_000, **kwargs)))
    return step


def grasp(step):
    step(0)
    step(10, close=True)
    return step(210, close=True, selected_id=4)


def test_r1_open(pipeline):
    msg = pipeline(0)
    assert msg.valid and not msg.holds_object


def test_r2_hold(pipeline):
    assert grasp(pipeline).holds_object


def test_r3_candidate(pipeline):
    msg = grasp(pipeline)
    assert msg.grasp_candidate_valid and msg.grasp_candidate_object_id == 3


def test_r4_commit(pipeline):
    grasp(pipeline)
    assert not pipeline(600, close=True).held_object_valid
    msg = pipeline(610, close=True, selected_id=4)
    assert msg.held_object_valid and msg.held_object_id == 3


def test_r5_release(pipeline):
    grasp(pipeline)
    pipeline(610, close=True)
    pipeline(620)
    msg = pipeline(720)
    assert not msg.holds_object and not msg.held_object_valid


@pytest.mark.parametrize('failure', ['missing', 'stale'])
def test_r6_invalid_after_hold(failure):
    estimator, stream = LocalHoldStateEstimator(), SemanticStream('User1')
    for ms, close in [(0, False), (10, True), (210, True), (610, True)]:
        state = estimator.update(mock_inputs(ms * 1_000_000, close=close))
    assert stream.encode(state).holds_object
    data = mock_inputs(620_000_000, close=True)
    distance = None if failure == 'missing' else replace(data.bottle_ee_distance, timestamp_ns=0)
    msg = stream.encode(estimator.update(replace(data, bottle_ee_distance=distance)))
    assert not msg.valid and not msg.holds_object and not msg.held_object_valid
    assert msg.invalid_reason


def test_r7_optional_load(pipeline):
    msg = grasp(pipeline)
    assert msg.valid and msg.holds_object and not msg.load_detected_available


def test_r8_sequence(pipeline):
    assert [pipeline(ms).sequence for ms in (0, 10, 20)] == [1, 2, 3]


def test_r9_timestamp(pipeline):
    msg = pipeline(123)
    assert msg.source_timestamp_valid and msg.source_timestamp_ns == 123_000_000


@pytest.mark.parametrize('identity', ['User1', 'User2', 'User3'])
def test_r10_pair_identity(identity):
    state = LocalHoldStateEstimator().update(mock_inputs(0))
    assert SemanticStream(identity).encode(state).participant_id == identity


@pytest.mark.parametrize('kind', ['open', 'hold', 'invalid', 'bad_clock', 'no_id', 'load'])
def test_r11_cdr_round_trip(kind):
    estimator = LocalHoldStateEstimator()
    estimator.update(mock_inputs(0))
    estimator.update(mock_inputs(10_000_000, close=True))
    data = mock_inputs(210_000_000, close=kind != 'open',
                       selected_id=None if kind == 'no_id' else 3,
                       load=False if kind == 'load' else None)
    if kind == 'invalid':
        data = replace(data, gripper_actual_position=None)
    if kind == 'bad_clock':
        data = replace(data, timestamp_ns=float('nan'))
    state = estimator.update(data)
    msg = SemanticStream('User1').encode(state)
    received = deserialize_message(serialize_message(msg), RosState)
    assert received == msg
    assert from_message(received) == state


def test_r12_no_raw_hardware_dependencies():
    import share_semantic_transport.node as node
    source = Path(node.__file__).read_text()
    tree = ast.parse(source)
    subscriptions = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                     and n.attr == 'create_subscription']
    assert not subscriptions
    publishers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'create_publisher']
    assert len(publishers) == 1
    assert ast.unparse(publishers[0].args[0]) == 'RosLocalHoldState'
    assert ast.literal_eval(publishers[0].args[1]) == 'local_hold_state'
    for forbidden in ('/joint_states', '/gripper_state', '/grasp_command', '/identified_bottle',
                      '/palm_pose', '/odom', '/tf', 'base_link', 'tcp_link', 'ee_link',
                      'ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION'):
        assert forbidden not in source


def test_sequence_overflow_and_new_lifetime():
    stream = SemanticStream('User1')
    state = LocalHoldStateEstimator().update(mock_inputs(0))
    stream.sequence = UINT64_MAX - 1
    assert stream.encode(state).sequence == UINT64_MAX
    with pytest.raises(OverflowError):
        stream.encode(state)
    other = SemanticStream('User1')
    assert other.session_id != stream.session_id and other.encode(state).sequence == 1


@pytest.mark.parametrize('identity', ['Unassigned', 'user_1', '', 'User4', 1])
def test_unknown_identity_rejected(identity):
    with pytest.raises(ValueError):
        SemanticStream(identity)


def test_integer_limits_fail_explicitly():
    state = LocalHoldStateEstimator().update(mock_inputs(0))
    with pytest.raises(ValueError):
        to_message(replace(state, held_object_id=1 << 63), 'User1', 'session', 1)
    with pytest.raises(ValueError):
        to_message(replace(state, timestamp_ns=1 << 64), 'User1', 'session', 1)


def test_bad_parallel_arrays_rejected(pipeline):
    msg = pipeline(0)
    msg.input_age_valid = []
    with pytest.raises(ValueError):
        from_message(msg)
