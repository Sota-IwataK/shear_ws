import math
import importlib.util
import sys
import types

import pytest

if importlib.util.find_spec("rclpy") is None:
    rclpy_stub = types.ModuleType("rclpy")
    rclpy_stub.init = lambda *args, **kwargs: None
    rclpy_stub.spin = lambda *args, **kwargs: None
    rclpy_stub.shutdown = lambda *args, **kwargs: None

    rclpy_node_stub = types.ModuleType("rclpy.node")

    class Node:
        pass

    rclpy_node_stub.Node = Node

    rclpy_signals_stub = types.ModuleType("rclpy.signals")

    class SignalHandlerOptions:
        NO = object()

    rclpy_signals_stub.SignalHandlerOptions = SignalHandlerOptions

    geometry_stub = types.ModuleType("geometry_msgs")
    geometry_msg_stub = types.ModuleType("geometry_msgs.msg")

    class Vector3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:
        def __init__(self):
            self.linear = Vector3()
            self.angular = Vector3()

    geometry_msg_stub.Twist = Twist
    geometry_stub.msg = geometry_msg_stub

    std_msgs_stub = types.ModuleType("std_msgs")
    std_msgs_msg_stub = types.ModuleType("std_msgs.msg")

    class Float32MultiArray:
        def __init__(self):
            self.data = []

    std_msgs_msg_stub.Float32MultiArray = Float32MultiArray
    std_msgs_stub.msg = std_msgs_msg_stub

    sys.modules["rclpy"] = rclpy_stub
    sys.modules["rclpy.node"] = rclpy_node_stub
    sys.modules["rclpy.signals"] = rclpy_signals_stub
    sys.modules["geometry_msgs"] = geometry_stub
    sys.modules["geometry_msgs.msg"] = geometry_msg_stub
    sys.modules["std_msgs"] = std_msgs_stub
    sys.modules["std_msgs.msg"] = std_msgs_msg_stub

from amir_irt_mro.amir_right_hand_mecanum_node import (
    AmirRightHandMecanumNode,
    parse_mecanum_input,
)


def test_parse_mecanum_input_accepts_four_field_unity_contract():
    parsed = parse_mecanum_input([1.0, 0.01, -0.02, 0.10])

    assert parsed.active is True
    assert parsed.active_value == pytest.approx(1.0)
    assert parsed.delta_x == pytest.approx(0.01)
    assert parsed.delta_z == pytest.approx(-0.02)
    assert parsed.roll_delta == pytest.approx(0.10)


def test_parse_mecanum_input_accepts_optional_timestamp_field():
    parsed = parse_mecanum_input([0.0, 0.01, -0.02, 0.10, 123.456])

    assert parsed.active is False
    assert parsed.delta_x == pytest.approx(0.01)
    assert parsed.delta_z == pytest.approx(-0.02)
    assert parsed.roll_delta == pytest.approx(0.10)


def test_parse_mecanum_input_rejects_short_message():
    with pytest.raises(ValueError, match="expected at least 4"):
        parse_mecanum_input([1.0, 0.01, -0.02])


def test_parse_mecanum_input_rejects_nan_or_inf():
    with pytest.raises(ValueError, match="NaN or inf"):
        parse_mecanum_input([1.0, math.nan, 0.0, 0.0])


def test_deadband_suppression_only_flags_nonzero_small_values():
    assert AmirRightHandMecanumNode.is_deadband_suppressed(0.01, 0.02)
    assert not AmirRightHandMecanumNode.is_deadband_suppressed(0.0, 0.02)
    assert not AmirRightHandMecanumNode.is_deadband_suppressed(0.03, 0.02)
