import importlib.util
import importlib.machinery
import math
import sys
import types

import pytest


def _module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


if not _module_available("rclpy"):
    def _stub_module(name):
        module = sys.modules.setdefault(name, types.ModuleType(name))
        if module.__spec__ is None:
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return module

    rclpy_stub = _stub_module("rclpy")
    rclpy_stub.init = lambda *args, **kwargs: None
    rclpy_stub.spin = lambda *args, **kwargs: None
    rclpy_stub.shutdown = lambda *args, **kwargs: None
    rclpy_stub.ok = lambda *args, **kwargs: False

    rclpy_node_stub = _stub_module("rclpy.node")

    class Node:
        pass

    rclpy_node_stub.Node = Node

    rclpy_signals_stub = _stub_module("rclpy.signals")

    class SignalHandlerOptions:
        NO = object()

    rclpy_signals_stub.SignalHandlerOptions = SignalHandlerOptions

    rcl_interfaces_stub = _stub_module("rcl_interfaces")
    rcl_interfaces_msg_stub = _stub_module("rcl_interfaces.msg")

    class SetParametersResult:
        def __init__(self, successful=True, reason=""):
            self.successful = successful
            self.reason = reason

    rcl_interfaces_msg_stub.SetParametersResult = SetParametersResult
    rcl_interfaces_stub.msg = rcl_interfaces_msg_stub

    geometry_stub = _stub_module("geometry_msgs")
    geometry_msg_stub = _stub_module("geometry_msgs.msg")

    class Vector3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Quaternion:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.w = 1.0

    class Pose:
        def __init__(self):
            self.position = Vector3()
            self.orientation = Quaternion()

    class Header:
        def __init__(self):
            self.frame_id = ""

    class PoseStamped:
        def __init__(self):
            self.header = Header()
            self.pose = Pose()

    class Twist:
        def __init__(self):
            self.linear = Vector3()
            self.angular = Vector3()

    geometry_msg_stub.PoseStamped = PoseStamped
    geometry_msg_stub.Twist = Twist
    geometry_stub.msg = geometry_msg_stub

    std_msgs_stub = _stub_module("std_msgs")
    std_msgs_msg_stub = _stub_module("std_msgs.msg")

    class Float32MultiArray:
        def __init__(self, data=None):
            self.data = [] if data is None else list(data)

    class String:
        def __init__(self, data=""):
            self.data = data

    std_msgs_msg_stub.Float32MultiArray = Float32MultiArray
    std_msgs_msg_stub.String = String
    std_msgs_stub.msg = std_msgs_msg_stub

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, String

from amir_irt_mro.amir_multi_user_input_router import (
    MultiUserInputRouterCore,
    control_target_for_channel,
)


def make_core():
    return MultiUserInputRouterCore(
        user_ids=["user_1", "user_2", "user_3"],
        active_user_id="user_1",
        timeout_sec=0.3,
    )


def make_robot_target_core(**kwargs):
    params = {
        "user_ids": ["user_1", "user_2", "user_3"],
        "active_user_id": "user_1",
        "timeout_sec": 0.3,
        "routing_mode": "robot_target",
        "target_timeout_sec": 2.5,
    }
    params.update(kwargs)
    return MultiUserInputRouterCore(**params)


def make_single_user_core(**kwargs):
    params = {
        "user_ids": ["user_1", "user_2", "user_3"],
        "active_user_id": "user_1",
        "single_user_id": "user_1",
        "timeout_sec": 0.3,
        "routing_mode": "single_user_dual",
        "target_timeout_sec": 2.5,
    }
    params.update(kwargs)
    return MultiUserInputRouterCore(**params)


def make_mecanum(data):
    msg = Float32MultiArray()
    msg.data = list(data)
    return msg


def make_target(target):
    return String(data=target)


def make_pose():
    msg = PoseStamped()
    msg.header.frame_id = "unity"
    msg.pose.position.x = 1.0
    msg.pose.position.y = 2.0
    msg.pose.position.z = 3.0
    msg.pose.orientation.x = 0.1
    msg.pose.orientation.y = 0.2
    msg.pose.orientation.z = 0.3
    msg.pose.orientation.w = 0.9
    return msg


def test_active_user_input_is_forwarded():
    core = make_core()
    msg = make_mecanum([1.0, 0.01, -0.02, 0.10])

    decision = core.route_message(
        "user_1", "right_hand_mecanum_input", msg, now_sec=1.0)

    assert decision.accepted
    assert decision.channel == "right_hand_mecanum_input"
    assert decision.message is msg


def test_inactive_user_input_is_discarded():
    core = make_core()
    msg = make_mecanum([1.0, 0.01, -0.02, 0.10])

    decision = core.route_message(
        "user_2", "right_hand_mecanum_input", msg, now_sec=1.0)

    assert not decision.accepted
    assert "inactive user" in decision.reason
    assert not decision.stop_required


def test_timeout_requests_rover_stop_once():
    core = make_core()
    msg = make_mecanum([1.0, 0.01, -0.02, 0.10])
    assert core.route_message(
        "user_1", "right_hand_mecanum_input", msg, now_sec=1.0).accepted

    before_timeout = core.check_timeout(now_sec=1.29)
    at_timeout = core.check_timeout(now_sec=1.31)
    after_stop = core.check_timeout(now_sec=1.62)

    assert not before_timeout.stop_required
    assert at_timeout.stop_required
    assert at_timeout.stop_reason == "input_timeout"
    assert not after_stop.stop_required
    assert list(core.make_stop_message().data) == [0.0, 0.0, 0.0, 0.0]


def test_invalid_mecanum_array_length_is_rejected():
    core = make_core()
    msg = make_mecanum([1.0, 0.01, -0.02])

    decision = core.route_message(
        "user_1", "right_hand_mecanum_input", msg, now_sec=1.0)

    assert not decision.accepted
    assert "expected at least 4" in decision.reason
    assert decision.stop_required


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_mecanum_nan_and_inf_are_rejected(bad_value):
    core = make_core()
    msg = make_mecanum([1.0, bad_value, 0.0, 0.0])

    decision = core.route_message(
        "user_1", "right_hand_mecanum_input", msg, now_sec=1.0)

    assert not decision.accepted
    assert "NaN or inf" in decision.reason
    assert decision.stop_required


def test_active_user_switch_requests_rover_stop():
    core = make_core()
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.01, 0.0, 0.0]),
        now_sec=1.0,
    ).accepted

    decision = core.set_active_user("user_2", now_sec=1.1)

    assert decision.accepted
    assert decision.stop_required
    assert decision.stop_reason == "active_user_changed"
    assert core.active_user_id == "user_2"


def test_palm_pose_is_forwarded_without_transform():
    core = make_core()
    msg = make_pose()

    decision = core.route_message(
        "user_1", "palm_pose_world", msg, now_sec=1.0)

    assert decision.accepted
    assert decision.message is msg
    assert decision.message.pose.position.x == pytest.approx(1.0)
    assert decision.message.pose.position.y == pytest.approx(2.0)
    assert decision.message.pose.position.z == pytest.approx(3.0)
    assert decision.message.pose.orientation.w == pytest.approx(0.9)


def test_gripper_message_is_forwarded_without_transform():
    core = make_core()
    msg = String(data="Open")

    decision = core.route_message(
        "user_1", "gripper_cmd", msg, now_sec=1.0)

    assert decision.accepted
    assert decision.message is msg
    assert decision.message.data == "Open"


def test_invalid_user_id_is_rejected():
    core = make_core()

    decision = core.set_active_user("user/4", now_sec=1.0)

    assert not decision.accepted
    assert "not configured" in decision.reason


def test_control_target_for_channel_classifies_router_outputs():
    assert control_target_for_channel("palm_pose_world") == "amir"
    assert control_target_for_channel("gripper_cmd") == "amir"
    assert control_target_for_channel("right_hand_mecanum_input") == "rover"
    assert control_target_for_channel("robot_target") is None


def test_robot_target_routes_user1_amir_and_user2_rover_simultaneously():
    core = make_robot_target_core()
    assert core.route_message(
        "user_1", "robot_target", make_target("amir"), now_sec=1.0).accepted
    assert core.route_message(
        "user_2", "robot_target", make_target("rover"), now_sec=1.1).accepted

    pose = make_pose()
    rover = make_mecanum([1.0, 0.02, -0.01, 0.10])
    amir_decision = core.route_message(
        "user_1", "palm_pose_world", pose, now_sec=1.2)
    rover_decision = core.route_message(
        "user_2", "right_hand_mecanum_input", rover, now_sec=1.2)

    assert amir_decision.accepted
    assert amir_decision.message is pose
    assert rover_decision.accepted
    assert rover_decision.message is rover


def test_both_users_amir_auto_discards_both():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    core.route_message("user_2", "robot_target", make_target("amir"), now_sec=1.1)

    user1 = core.route_message(
        "user_1", "palm_pose_world", make_pose(), now_sec=1.2)
    user2 = core.route_message(
        "user_2", "palm_pose_world", make_pose(), now_sec=1.2)

    assert not user1.accepted
    assert "conflict" in user1.reason
    assert not user2.accepted
    assert "conflict" in user2.reason


def test_explicit_amir_user_selects_user1_during_conflict():
    core = make_robot_target_core(amir_active_user_id="user_1")
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    core.route_message("user_2", "robot_target", make_target("amir"), now_sec=1.1)

    user1 = core.route_message(
        "user_1", "palm_pose_world", make_pose(), now_sec=1.2)
    user2 = core.route_message(
        "user_2", "palm_pose_world", make_pose(), now_sec=1.2)

    assert user1.accepted
    assert not user2.accepted
    assert 'selected="user_1"' in user2.reason


def test_target_change_does_not_change_user_topic_contract():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    amir_pose = core.route_message(
        "user_1", "palm_pose_world", make_pose(), now_sec=1.1)
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.2)
    rover_input = core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.3,
    )

    assert amir_pose.accepted
    assert rover_input.accepted


def test_stale_target_is_rejected():
    core = make_robot_target_core(target_timeout_sec=2.5)
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    timeout = core.check_timeouts(now_sec=3.6)
    decision = core.route_message(
        "user_1", "palm_pose_world", make_pose(), now_sec=3.7)

    assert timeout.events
    assert not decision.accepted
    assert "missing or stale" in decision.reason


def test_invalid_target_is_rejected():
    core = make_robot_target_core()

    decision = core.route_message(
        "user_1", "robot_target", make_target(" excavator "), now_sec=1.0)

    assert not decision.accepted
    assert "robot_target must be one of" in decision.reason


def test_observer_input_is_rejected():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("observer"), now_sec=1.0)

    decision = core.route_message(
        "user_1", "palm_pose_world", make_pose(), now_sec=1.1)

    assert not decision.accepted
    assert "observer" in decision.reason


def test_rover_target_leave_requests_stop():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.0)
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.1,
    ).accepted

    decision = core.route_message(
        "user_1", "robot_target", make_target("amir"), now_sec=1.2)

    assert decision.stop_required
    assert decision.stop_reason == "rover_target_changed"


def test_rover_target_timeout_requests_stop():
    core = make_robot_target_core(target_timeout_sec=0.5, timeout_sec=10.0)
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.0)
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.1,
    ).accepted

    decision = core.check_timeouts(now_sec=1.6)

    assert decision.stop_required
    assert decision.stop_reason == "target_timeout"


def test_robot_target_mecanum_timeout_requests_stop():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.0)
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.1,
    ).accepted

    decision = core.check_timeouts(now_sec=1.5)

    assert decision.stop_required
    assert decision.stop_reason == "input_timeout"


def test_rover_authority_change_requests_stop():
    core = make_robot_target_core(rover_active_user_id="user_1")
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.0)
    core.route_message("user_2", "robot_target", make_target("rover"), now_sec=1.1)
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.2,
    ).accepted

    decision = core.set_robot_active_user("rover", "user_2", now_sec=1.3)

    assert decision.stop_required
    assert decision.stop_reason == "rover_active_user_changed"


def test_rover_conflict_requests_stop():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.0)
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.1,
    ).accepted

    decision = core.route_message(
        "user_2", "robot_target", make_target("rover"), now_sec=1.2)

    assert decision.stop_required
    assert decision.stop_reason == "rover_target_conflict"


def test_drone_without_consumer_is_not_forwarded():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("drone"), now_sec=1.0)

    decision = core.route_message(
        "user_1", "palm_pose_world", make_pose(), now_sec=1.1)

    assert not decision.accepted
    assert "drone target has no configured consumer" in decision.reason


def test_robot_target_pose_is_forwarded_without_transform():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    msg = make_pose()

    decision = core.route_message(
        "user_1", "palm_pose_hmd_relative", msg, now_sec=1.1)

    assert decision.accepted
    assert decision.message is msg
    assert decision.message.pose.position.x == pytest.approx(1.0)
    assert decision.message.pose.position.y == pytest.approx(2.0)
    assert decision.message.pose.position.z == pytest.approx(3.0)


def test_robot_target_gripper_string_is_forwarded_without_transform():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    msg = String(data="airtap-close")

    decision = core.route_message(
        "user_1", "gripper_cmd", msg, now_sec=1.1)

    assert decision.accepted
    assert decision.message is msg
    assert decision.message.data == "airtap-close"


def test_robot_target_mecanum_accepts_four_and_five_fields():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.0)
    four = make_mecanum([1.0, 0.02, -0.01, 0.10])
    five = make_mecanum([1.0, 0.02, -0.01, 0.10, 123.456])

    first = core.route_message(
        "user_1", "right_hand_mecanum_input", four, now_sec=1.1)
    second = core.route_message(
        "user_1", "right_hand_mecanum_input", five, now_sec=1.2)

    assert first.accepted
    assert second.accepted
    assert list(second.message.data) == pytest.approx(
        [1.0, 0.02, -0.01, 0.10, 123.456])


def test_robot_target_amir_rejects_right_hand_mecanum_input():
    core = make_robot_target_core()
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)

    decision = core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.1,
    )

    assert not decision.accepted
    assert 'channel "right_hand_mecanum_input" is not routed for AMIR target' in (
        decision.reason)


def test_single_user_dual_routes_palm_to_amir_when_target_is_amir():
    core = make_single_user_core()
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    msg = make_pose()

    decision = core.route_message(
        "user_1", "palm_pose_world", msg, now_sec=1.1)

    assert decision.accepted
    assert decision.channel == "palm_pose_world"
    assert decision.message is msg


def test_single_user_dual_routes_mecanum_to_rover_when_target_is_amir():
    core = make_single_user_core()
    core.route_message("user_1", "robot_target", make_target("amir"), now_sec=1.0)
    msg = make_mecanum([1.0, 0.02, 0.0, 0.0])

    decision = core.route_message(
        "user_1", "right_hand_mecanum_input", msg, now_sec=1.1)

    assert decision.accepted
    assert decision.channel == "right_hand_mecanum_input"
    assert decision.message is msg


def test_single_user_dual_routes_palm_to_amir_when_target_is_rover():
    core = make_single_user_core()
    core.route_message("user_1", "robot_target", make_target("rover"), now_sec=1.0)
    msg = make_pose()

    decision = core.route_message(
        "user_1", "palm_pose_hmd_relative", msg, now_sec=1.1)

    assert decision.accepted
    assert decision.channel == "palm_pose_hmd_relative"
    assert decision.message is msg


def test_single_user_dual_routes_gripper_to_amir_output():
    core = make_single_user_core()
    msg = String(data="close")

    decision = core.route_message(
        "user_1", "gripper_cmd", msg, now_sec=1.0)

    assert decision.accepted
    assert decision.channel == "gripper_cmd"
    assert decision.message is msg
    assert decision.message.data == "close"


@pytest.mark.parametrize(
    "channel,make_msg",
    (
        ("palm_pose_world", make_pose),
        ("gripper_cmd", lambda: String(data="open")),
        ("right_hand_mecanum_input", lambda: make_mecanum([1.0, 0.02, 0.0, 0.0])),
    ),
)
def test_single_user_dual_rejects_non_single_user_inputs(channel, make_msg):
    core = make_single_user_core(single_user_id="user_1")

    decision = core.route_message(
        "user_2", channel, make_msg(), now_sec=1.0)

    assert not decision.accepted
    assert 'single_user_id="user_1"' in decision.reason


def test_single_user_dual_mecanum_timeout_requests_stop_once():
    core = make_single_user_core()
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.0,
    ).accepted

    before_timeout = core.check_timeout(now_sec=1.29)
    at_timeout = core.check_timeout(now_sec=1.31)
    after_stop = core.check_timeout(now_sec=1.62)

    assert not before_timeout.stop_required
    assert at_timeout.stop_required
    assert at_timeout.stop_reason == "input_timeout"
    assert not after_stop.stop_required


def test_single_user_dual_single_user_change_requests_stop():
    core = make_single_user_core(single_user_id="user_1")
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.0,
    ).accepted

    decision = core.set_single_user("user_2", now_sec=1.1)
    old_user = core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.2,
    )

    assert decision.accepted
    assert decision.stop_required
    assert decision.stop_reason == "single_user_changed"
    assert not old_user.accepted
    assert 'single_user_id="user_2"' in old_user.reason


def test_single_user_dual_routing_mode_leave_requests_stop():
    core = make_single_user_core()
    assert core.route_message(
        "user_1",
        "right_hand_mecanum_input",
        make_mecanum([1.0, 0.02, 0.0, 0.0]),
        now_sec=1.0,
    ).accepted

    decision = core.set_routing_mode("exclusive_user", now_sec=1.1)

    assert decision.accepted
    assert decision.stop_required
    assert decision.stop_reason == "routing_mode_changed"
