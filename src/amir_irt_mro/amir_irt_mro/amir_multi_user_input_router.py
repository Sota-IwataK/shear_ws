#!/usr/bin/env python3
"""Route Unity Shared-mode user inputs to existing AMIR control topics."""

import math
import re
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from amir_irt_mro.amir_right_hand_mecanum_node import parse_mecanum_input


DEFAULT_USER_IDS = ("user_1", "user_2", "user_3")
ROUTING_MODES = ("exclusive_user", "robot_target", "single_user_dual")
ROBOT_TARGETS = ("amir", "rover", "drone", "observer")
CONTROL_AUTH_SPECIAL_VALUES = ("auto", "none")
USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")

AMIR_CHANNELS = {
    "palm_pose",
    "palm_pose_world",
    "palm_pose_control_center_world",
    "palm_pose_hmd_relative",
    "gripper_cmd",
}
ROVER_CHANNELS = {"right_hand_mecanum_input"}
POSE_CHANNELS = {
    "palm_pose",
    "palm_pose_world",
    "palm_pose_control_center_world",
    "palm_pose_hmd_relative",
}


@dataclass(frozen=True)
class InputChannelSpec:
    name: str
    input_suffix: str
    message_type: object
    ros_type: str
    output_parameter: str = ""
    default_output_topic: str = ""
    qos_depth: int = 10


CONTROL_CHANNEL_SPECS = (
    InputChannelSpec(
        name="palm_pose",
        input_suffix="palm_pose",
        message_type=PoseStamped,
        ros_type="geometry_msgs/msg/PoseStamped",
        output_parameter="palm_pose_output_topic",
        default_output_topic="/palm_pose",
        qos_depth=50,
    ),
    InputChannelSpec(
        name="palm_pose_world",
        input_suffix="palm_pose_world",
        message_type=PoseStamped,
        ros_type="geometry_msgs/msg/PoseStamped",
        output_parameter="palm_pose_world_output_topic",
        default_output_topic="/palm_pose_world",
        qos_depth=50,
    ),
    InputChannelSpec(
        name="palm_pose_control_center_world",
        input_suffix="palm_pose_control_center_world",
        message_type=PoseStamped,
        ros_type="geometry_msgs/msg/PoseStamped",
        output_parameter="palm_pose_control_center_world_output_topic",
        default_output_topic="/palm_pose_control_center_world",
        qos_depth=10,
    ),
    InputChannelSpec(
        name="palm_pose_hmd_relative",
        input_suffix="palm_pose_hmd_relative",
        message_type=PoseStamped,
        ros_type="geometry_msgs/msg/PoseStamped",
        output_parameter="palm_pose_hmd_relative_output_topic",
        default_output_topic="/palm_pose_hmd_relative",
        qos_depth=50,
    ),
    InputChannelSpec(
        name="right_hand_mecanum_input",
        input_suffix="right_hand_mecanum_input",
        message_type=Float32MultiArray,
        ros_type="std_msgs/msg/Float32MultiArray",
        output_parameter="right_hand_mecanum_output_topic",
        default_output_topic="/amir/right_hand_mecanum_input",
        qos_depth=10,
    ),
    InputChannelSpec(
        name="gripper_cmd",
        input_suffix="gripper_cmd",
        message_type=String,
        ros_type="std_msgs/msg/String",
        output_parameter="gripper_cmd_output_topic",
        default_output_topic="/grasp_command",
        qos_depth=10,
    ),
)

ROBOT_TARGET_SPEC = InputChannelSpec(
    name="robot_target",
    input_suffix="robot_target",
    message_type=String,
    ros_type="std_msgs/msg/String",
    qos_depth=10,
)
USER_INPUT_SPECS = CONTROL_CHANNEL_SPECS + (ROBOT_TARGET_SPEC,)
CONTROL_CHANNEL_BY_NAME = {spec.name: spec for spec in CONTROL_CHANNEL_SPECS}
USER_INPUT_BY_NAME = {spec.name: spec for spec in USER_INPUT_SPECS}


@dataclass(frozen=True)
class RouteDecision:
    accepted: bool
    channel: str = ""
    message: object = None
    reason: str = ""
    stop_required: bool = False
    stop_reason: str = ""
    events: tuple = field(default_factory=tuple)


def normalize_user_ids(user_ids):
    normalized = []
    values = user_ids
    if isinstance(user_ids, str):
        values = [value.strip() for value in user_ids.split(",")]
    for raw_user_id in values:
        user_id = str(raw_user_id).strip()
        if not user_id:
            continue
        if not USER_ID_PATTERN.match(user_id):
            raise ValueError(f'invalid user_id "{user_id}"')
        if user_id in normalized:
            raise ValueError(f'duplicate user_id "{user_id}"')
        normalized.append(user_id)
    if not normalized:
        raise ValueError("user_ids must contain at least one valid user id")
    return tuple(normalized)


def validate_user_id(user_id, allowed_user_ids):
    if str(user_id) not in allowed_user_ids:
        return f'user_id "{user_id}" is not configured'
    if not USER_ID_PATTERN.match(str(user_id)):
        return f'user_id "{user_id}" has an invalid format'
    return None


def validate_routing_mode(routing_mode):
    value = str(routing_mode).strip().lower()
    if value not in ROUTING_MODES:
        raise ValueError(
            f'routing_mode must be one of {ROUTING_MODES}; got "{routing_mode}"')
    return value


def validate_control_user_value(value, allowed_user_ids, parameter_name):
    normalized = str(value).strip()
    lower_value = normalized.lower()
    if lower_value in CONTROL_AUTH_SPECIAL_VALUES:
        return lower_value
    reason = validate_user_id(normalized, allowed_user_ids)
    if reason is not None:
        raise ValueError(f"{parameter_name}: {reason}")
    return normalized


def normalize_robot_target(target):
    normalized = str(target).strip().lower()
    if normalized not in ROBOT_TARGETS:
        raise ValueError(
            f'robot_target must be one of {ROBOT_TARGETS}; got "{target}"')
    return normalized


def control_target_for_channel(channel):
    if channel in AMIR_CHANNELS:
        return "amir"
    if channel in ROVER_CHANNELS:
        return "rover"
    return None


def is_channel_routed_for_target(channel, target):
    return control_target_for_channel(channel) == target


def is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_pose_stamped(msg):
    fields = (
        ("pose.position.x", msg.pose.position.x),
        ("pose.position.y", msg.pose.position.y),
        ("pose.position.z", msg.pose.position.z),
        ("pose.orientation.x", msg.pose.orientation.x),
        ("pose.orientation.y", msg.pose.orientation.y),
        ("pose.orientation.z", msg.pose.orientation.z),
        ("pose.orientation.w", msg.pose.orientation.w),
    )
    for field_name, value in fields:
        if not is_finite_number(value):
            return f"{field_name} is not finite"
    return None


def validate_mecanum_message(msg):
    data = list(msg.data)
    try:
        parse_mecanum_input(data)
    except ValueError as exc:
        return str(exc)
    for index, value in enumerate(data):
        if not is_finite_number(value):
            return f"right_hand_mecanum_input data[{index}] is not finite"
    return None


def validate_gripper_message(msg):
    if not hasattr(msg, "data"):
        return "gripper_cmd message has no data field"
    return None


def validate_message(channel, msg):
    if channel in POSE_CHANNELS:
        return validate_pose_stamped(msg)
    if channel == "right_hand_mecanum_input":
        return validate_mecanum_message(msg)
    if channel == "gripper_cmd":
        return validate_gripper_message(msg)
    return f'unknown input channel "{channel}"'


def validate_mecanum_stop_data(stop_data):
    data = list(stop_data)
    if len(data) < 4:
        raise ValueError("mecanum_stop_data must contain at least 4 values")
    for index, value in enumerate(data):
        if not is_finite_number(value):
            raise ValueError(f"mecanum_stop_data[{index}] must be finite")
    return [float(value) for value in data]


def make_mecanum_stop_message(stop_data):
    msg = Float32MultiArray()
    msg.data = validate_mecanum_stop_data(stop_data)
    return msg


class MultiUserInputRouterCore:
    """Pure routing policy for unit tests and the ROS node wrapper."""

    def __init__(
        self,
        user_ids,
        active_user_id,
        timeout_sec,
        routing_mode="exclusive_user",
        target_timeout_sec=2.5,
        amir_active_user_id="auto",
        rover_active_user_id="auto",
        drone_active_user_id="auto",
        single_user_id=DEFAULT_USER_IDS[0],
        mecanum_stop_data=None,
        drone_consumer_available=False,
    ):
        self.user_ids = normalize_user_ids(user_ids)
        active_user_id = str(active_user_id).strip()
        reason = validate_user_id(active_user_id, self.user_ids)
        if reason is not None:
            raise ValueError(reason)
        self.active_user_id = active_user_id
        single_user_id = str(single_user_id).strip()
        reason = validate_user_id(single_user_id, self.user_ids)
        if reason is not None:
            raise ValueError(f"single_user_id: {reason}")
        self.single_user_id = single_user_id
        self.routing_mode = validate_routing_mode(routing_mode)
        self.timeout_sec = max(float(timeout_sec), 0.0)
        self.target_timeout_sec = max(float(target_timeout_sec), 0.0)
        self.robot_active_user_ids = {
            "amir": validate_control_user_value(
                amir_active_user_id, self.user_ids, "amir_active_user_id"),
            "rover": validate_control_user_value(
                rover_active_user_id, self.user_ids, "rover_active_user_id"),
            "drone": validate_control_user_value(
                drone_active_user_id, self.user_ids, "drone_active_user_id"),
        }
        self.mecanum_stop_data = validate_mecanum_stop_data(
            mecanum_stop_data if mecanum_stop_data is not None
            else [0.0, 0.0, 0.0, 0.0])
        self.drone_consumer_available = bool(drone_consumer_available)

        self.target_by_user = {user_id: None for user_id in self.user_ids}
        self.target_time_by_user = {user_id: None for user_id in self.user_ids}
        self.target_fresh_by_user = {user_id: False for user_id in self.user_ids}
        self.last_selected_users = {target: None for target in ("amir", "rover", "drone")}
        self.last_conflicts = {target: False for target in ("amir", "rover", "drone")}
        self.last_mecanum_input_time = None
        self.mecanum_stop_sent = True

    def set_timeout_sec(self, timeout_sec):
        self.timeout_sec = max(float(timeout_sec), 0.0)

    def set_target_timeout_sec(self, timeout_sec, now_sec):
        self.target_timeout_sec = max(float(timeout_sec), 0.0)
        return self.check_timeouts(now_sec)

    def set_active_user(self, user_id, now_sec):
        user_id = str(user_id).strip()
        reason = validate_user_id(user_id, self.user_ids)
        if reason is not None:
            return RouteDecision(accepted=False, reason=reason)
        if user_id == self.active_user_id:
            return RouteDecision(accepted=True)
        self.active_user_id = user_id
        if self.routing_mode == "exclusive_user":
            return self._make_stop_decision(
                "active_user_changed",
                events=(f'active_user_id changed to "{user_id}"',),
            )
        return RouteDecision(
            accepted=True,
            events=(f'active_user_id changed to "{user_id}"',),
        )

    def set_single_user(self, user_id, now_sec):
        user_id = str(user_id).strip()
        reason = validate_user_id(user_id, self.user_ids)
        if reason is not None:
            return RouteDecision(
                accepted=False,
                reason=f"single_user_id: {reason}",
            )
        if user_id == self.single_user_id:
            return RouteDecision(accepted=True)
        self.single_user_id = user_id
        events = (f'single_user_id changed to "{user_id}"',)
        if self.routing_mode == "single_user_dual":
            return self._make_stop_decision(
                "single_user_changed",
                events=events,
            )
        return RouteDecision(accepted=True, events=events)

    def set_routing_mode(self, routing_mode, now_sec):
        routing_mode = validate_routing_mode(routing_mode)
        if routing_mode == self.routing_mode:
            return RouteDecision(accepted=True)
        self.routing_mode = routing_mode
        events = (f'routing_mode changed to "{routing_mode}"',)
        decision = self._reconcile_authority(now_sec, events=events)
        return self._merge_stop(decision, "routing_mode_changed", events)

    def set_robot_active_user(self, robot_target, value, now_sec):
        if robot_target not in self.robot_active_user_ids:
            return RouteDecision(
                accepted=False,
                reason=f'unknown robot target "{robot_target}"',
            )
        normalized = validate_control_user_value(
            value, self.user_ids, f"{robot_target}_active_user_id")
        if normalized == self.robot_active_user_ids[robot_target]:
            return RouteDecision(accepted=True)
        self.robot_active_user_ids[robot_target] = normalized
        events = (f'{robot_target}_active_user_id changed to "{normalized}"',)
        decision = self._reconcile_authority(now_sec, events=events)
        if robot_target == "rover" and normalized == "none":
            return self._merge_stop(decision, "rover_active_user_id_none", events)
        return decision

    def update_robot_target(self, user_id, msg, now_sec):
        user_id = str(user_id).strip()
        reason = validate_user_id(user_id, self.user_ids)
        if reason is not None:
            return RouteDecision(accepted=False, reason=reason)

        old_target = self.target_by_user[user_id]
        old_fresh = self.target_fresh_by_user[user_id]
        old_rover_user = self.last_selected_users["rover"]
        raw_target = getattr(msg, "data", "")

        try:
            target = normalize_robot_target(raw_target)
        except ValueError as exc:
            self.target_by_user[user_id] = None
            self.target_time_by_user[user_id] = None
            self.target_fresh_by_user[user_id] = False
            decision = self._reconcile_authority(
                now_sec,
                events=(f'invalid robot_target from {user_id}: "{raw_target}"',),
            )
            if old_target == "rover" and old_fresh and old_rover_user == user_id:
                decision = self._merge_stop(decision, "rover_target_invalid", decision.events)
            return RouteDecision(
                accepted=False,
                reason=str(exc),
                stop_required=decision.stop_required,
                stop_reason=decision.stop_reason,
                events=decision.events,
            )

        events = []
        if old_target != target:
            events.append(f'{user_id} robot_target changed: {old_target} -> {target}')
        elif not old_fresh:
            events.append(f'{user_id} robot_target fresh again: {target}')

        self.target_by_user[user_id] = target
        self.target_time_by_user[user_id] = float(now_sec)
        self.target_fresh_by_user[user_id] = True
        decision = self._reconcile_authority(now_sec, events=tuple(events))
        if old_target == "rover" and target != "rover" and old_rover_user == user_id:
            return self._merge_stop(decision, "rover_target_changed", decision.events)
        return decision

    def route_message(self, user_id, channel, msg, now_sec):
        if channel == "robot_target":
            return self.update_robot_target(user_id, msg, now_sec)

        user_id = str(user_id).strip()
        reason = validate_user_id(user_id, self.user_ids)
        if reason is not None:
            return RouteDecision(accepted=False, reason=reason)
        if channel not in CONTROL_CHANNEL_BY_NAME:
            return RouteDecision(
                accepted=False,
                reason=f'unknown input channel "{channel}"',
            )

        if self.routing_mode == "exclusive_user":
            return self._route_exclusive_user(user_id, channel, msg, now_sec)
        if self.routing_mode == "robot_target":
            return self._route_robot_target(user_id, channel, msg, now_sec)
        return self._route_single_user_dual(user_id, channel, msg, now_sec)

    def _route_exclusive_user(self, user_id, channel, msg, now_sec):
        if user_id != self.active_user_id:
            return RouteDecision(
                accepted=False,
                reason=(
                    f'inactive user "{user_id}" '
                    f'(active_user_id="{self.active_user_id}")'
                ),
            )
        return self._validate_and_accept(channel, msg, now_sec)

    def _route_robot_target(self, user_id, channel, msg, now_sec):
        self.check_timeouts(now_sec)
        target = self.fresh_target_for_user(user_id, now_sec)
        if target is None:
            return RouteDecision(
                accepted=False,
                reason=f'robot_target for "{user_id}" is missing or stale',
            )
        if target == "observer":
            return RouteDecision(
                accepted=False,
                reason=f'user "{user_id}" target is observer',
            )
        if target == "drone":
            return RouteDecision(
                accepted=False,
                reason=(
                    "drone target has no configured consumer/output topic "
                    "in this router"
                ),
            )

        selected_user = self.selected_user_for_target(target, now_sec)
        if selected_user is None:
            conflict = self.has_conflict(target, now_sec)
            reason = "conflict" if conflict else "no active user"
            return RouteDecision(
                accepted=False,
                reason=f'{target} has {reason}',
            )
        if selected_user != user_id:
            return RouteDecision(
                accepted=False,
                reason=(
                    f'user "{user_id}" is not selected for {target} '
                    f'(selected="{selected_user}")'
                ),
            )

        if target == "amir" and not is_channel_routed_for_target(channel, "amir"):
            return RouteDecision(
                accepted=False,
                reason=f'channel "{channel}" is not routed for AMIR target',
            )
        if target == "rover" and not is_channel_routed_for_target(channel, "rover"):
            return RouteDecision(
                accepted=False,
                reason=f'channel "{channel}" is not routed for Rover target',
            )
        return self._validate_and_accept(channel, msg, now_sec)

    def _route_single_user_dual(self, user_id, channel, msg, now_sec):
        if user_id != self.single_user_id:
            return RouteDecision(
                accepted=False,
                reason=(
                    f'inactive user "{user_id}" '
                    f'(single_user_id="{self.single_user_id}")'
                ),
            )
        target = control_target_for_channel(channel)
        if target == "amir" or target == "rover":
            return self._validate_and_accept(channel, msg, now_sec)
        return RouteDecision(
            accepted=False,
            reason=f'channel "{channel}" is not routed in single_user_dual',
        )

    def _validate_and_accept(self, channel, msg, now_sec):
        reason = validate_message(channel, msg)
        if reason is not None:
            stop_required = channel == "right_hand_mecanum_input"
            if stop_required:
                self.last_mecanum_input_time = None
                self.mecanum_stop_sent = True
            return RouteDecision(
                accepted=False,
                reason=reason,
                stop_required=stop_required,
                stop_reason="invalid_mecanum_input",
            )

        if channel == "right_hand_mecanum_input":
            self.last_mecanum_input_time = float(now_sec)
            self.mecanum_stop_sent = False

        return RouteDecision(
            accepted=True,
            channel=channel,
            message=msg,
        )

    def check_timeouts(self, now_sec):
        events = []
        timeout_rover_user = False
        for user_id in self.user_ids:
            was_fresh = self.target_fresh_by_user[user_id]
            if was_fresh and not self.is_target_fresh(user_id, now_sec):
                self.target_fresh_by_user[user_id] = False
                target = self.target_by_user[user_id]
                events.append(f'{user_id} robot_target timed out: {target}')
                if target == "rover" and self.last_selected_users["rover"] == user_id:
                    timeout_rover_user = True

        decision = self._reconcile_authority(now_sec, events=tuple(events))
        if timeout_rover_user:
            decision = self._merge_stop(decision, "target_timeout", decision.events)

        mecanum_decision = self._check_mecanum_timeout(now_sec)
        if mecanum_decision.stop_required:
            decision = self._merge_stop(
                decision,
                mecanum_decision.stop_reason,
                decision.events + mecanum_decision.events,
            )
        elif mecanum_decision.events:
            decision = RouteDecision(
                accepted=True,
                stop_required=decision.stop_required,
                stop_reason=decision.stop_reason,
                events=decision.events + mecanum_decision.events,
            )
        return decision

    def check_timeout(self, now_sec):
        return self.check_timeouts(now_sec)

    def _check_mecanum_timeout(self, now_sec):
        if self.timeout_sec <= 0.0:
            return RouteDecision(accepted=True)
        if self.last_mecanum_input_time is None:
            return RouteDecision(accepted=True)
        elapsed_sec = float(now_sec) - self.last_mecanum_input_time
        if elapsed_sec < self.timeout_sec:
            return RouteDecision(accepted=True)
        if self.mecanum_stop_sent:
            return RouteDecision(accepted=True)
        return self._make_stop_decision(
            "input_timeout",
            events=("right_hand_mecanum_input timed out",),
        )

    def _reconcile_authority(self, now_sec, events=()):
        selected_users = self.selected_users_by_target(now_sec)
        conflicts = {
            target: self.has_conflict(target, now_sec)
            for target in ("amir", "rover", "drone")
        }

        event_list = list(events)
        for target in ("amir", "rover", "drone"):
            old_user = self.last_selected_users[target]
            new_user = selected_users[target]
            if old_user != new_user:
                event_list.append(
                    f'{target} selected user changed: {old_user} -> {new_user}')
            if conflicts[target] and not self.last_conflicts[target]:
                event_list.append(f'{target} target conflict detected')
            if not conflicts[target] and self.last_conflicts[target]:
                event_list.append(f'{target} target conflict cleared')

        stop_required = False
        stop_reason = ""
        if selected_users["rover"] != self.last_selected_users["rover"]:
            stop_required = True
            stop_reason = "rover_active_user_changed"
        if conflicts["rover"] and not self.last_conflicts["rover"]:
            stop_required = True
            stop_reason = "rover_target_conflict"

        self.last_selected_users = selected_users
        self.last_conflicts = conflicts
        if stop_required:
            return self._make_stop_decision(stop_reason, tuple(event_list))
        return RouteDecision(accepted=True, events=tuple(event_list))

    def selected_users_by_target(self, now_sec):
        return {
            target: self.selected_user_for_target(target, now_sec)
            for target in ("amir", "rover", "drone")
        }

    def selected_user_for_target(self, target, now_sec):
        if self.routing_mode != "robot_target":
            return None
        setting = self.robot_active_user_ids[target]
        if setting == "none":
            return None
        if setting == "auto":
            candidates = self.fresh_users_for_target(target, now_sec)
            return candidates[0] if len(candidates) == 1 else None
        if self.fresh_target_for_user(setting, now_sec) == target:
            return setting
        return None

    def has_conflict(self, target, now_sec):
        if self.routing_mode != "robot_target":
            return False
        if self.robot_active_user_ids[target] != "auto":
            return False
        return len(self.fresh_users_for_target(target, now_sec)) > 1

    def fresh_users_for_target(self, target, now_sec):
        return [
            user_id
            for user_id in self.user_ids
            if self.fresh_target_for_user(user_id, now_sec) == target
        ]

    def fresh_target_for_user(self, user_id, now_sec):
        if not self.is_target_fresh(user_id, now_sec):
            return None
        return self.target_by_user[user_id]

    def is_target_fresh(self, user_id, now_sec):
        if not self.target_fresh_by_user[user_id]:
            return False
        target_time = self.target_time_by_user[user_id]
        if target_time is None:
            return False
        if self.target_timeout_sec <= 0.0:
            return True
        return (float(now_sec) - target_time) <= self.target_timeout_sec

    def _make_stop_decision(self, reason, events=()):
        self.last_mecanum_input_time = None
        self.mecanum_stop_sent = True
        return RouteDecision(
            accepted=True,
            stop_required=True,
            stop_reason=reason,
            events=tuple(events),
        )

    def _merge_stop(self, decision, reason, events=()):
        self.last_mecanum_input_time = None
        self.mecanum_stop_sent = True
        return RouteDecision(
            accepted=decision.accepted,
            channel=decision.channel,
            message=decision.message,
            reason=decision.reason,
            stop_required=True,
            stop_reason=reason,
            events=tuple(events) or decision.events,
        )

    def make_stop_message(self):
        return make_mecanum_stop_message(self.mecanum_stop_data)


class AmirMultiUserInputRouter(Node):
    def __init__(self):
        super().__init__("amir_multi_user_input_router")

        self.declare_parameter("user_ids", list(DEFAULT_USER_IDS))
        self.declare_parameter("routing_mode", "exclusive_user")
        self.declare_parameter("active_user_id", DEFAULT_USER_IDS[0])
        self.declare_parameter("single_user_id", DEFAULT_USER_IDS[0])
        self.declare_parameter("amir_active_user_id", "auto")
        self.declare_parameter("rover_active_user_id", "auto")
        self.declare_parameter("drone_active_user_id", "auto")
        self.declare_parameter("user_topic_prefix", "/share/users")
        self.declare_parameter("subscribe_only_existing_topics", True)
        self.declare_parameter("topic_discovery_period_sec", 0.5)
        self.declare_parameter("timeout_sec", 0.3)
        self.declare_parameter("target_timeout_sec", 2.5)
        self.declare_parameter("log_throttle_sec", 1.0)
        self.declare_parameter("mecanum_stop_data", [0.0, 0.0, 0.0, 0.0])
        for spec in CONTROL_CHANNEL_SPECS:
            self.declare_parameter(
                spec.output_parameter, spec.default_output_topic)

        user_ids = normalize_user_ids(self.get_parameter("user_ids").value)
        self.log_throttle_sec = max(
            float(self.get_parameter("log_throttle_sec").value), 0.0)
        self.user_topic_prefix = self._normalize_topic_prefix(
            self.get_parameter("user_topic_prefix").value)
        self.subscribe_only_existing_topics = bool(
            self.get_parameter("subscribe_only_existing_topics").value)
        self.topic_discovery_period_sec = max(
            float(self.get_parameter("topic_discovery_period_sec").value),
            0.1,
        )
        self.core = MultiUserInputRouterCore(
            user_ids=user_ids,
            active_user_id=str(self.get_parameter("active_user_id").value).strip(),
            timeout_sec=float(self.get_parameter("timeout_sec").value),
            routing_mode=self.get_parameter("routing_mode").value,
            single_user_id=str(
                self.get_parameter("single_user_id").value).strip(),
            target_timeout_sec=float(
                self.get_parameter("target_timeout_sec").value),
            amir_active_user_id=self.get_parameter(
                "amir_active_user_id").value,
            rover_active_user_id=self.get_parameter(
                "rover_active_user_id").value,
            drone_active_user_id=self.get_parameter(
                "drone_active_user_id").value,
            mecanum_stop_data=self.get_parameter("mecanum_stop_data").value,
            drone_consumer_available=False,
        )

        self.output_topics = {}
        self.output_publishers = {}
        for spec in CONTROL_CHANNEL_SPECS:
            output_topic = self._topic_parameter(
                spec.output_parameter, spec.default_output_topic)
            self.output_topics[spec.name] = output_topic
            self.output_publishers[spec.name] = self.create_publisher(
                spec.message_type, output_topic, spec.qos_depth)

        self.input_subscriptions = {}
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.discovery_timer = self.create_timer(
            self.topic_discovery_period_sec, self.discover_user_topics)
        timeout_timer_period = min(
            max(min(self.core.timeout_sec, self.core.target_timeout_sec) / 2.0, 0.05),
            0.2,
        )
        self.timeout_timer = self.create_timer(
            timeout_timer_period, self.check_timeout)
        self.discover_user_topics()

        self.get_logger().info(
            "AmirMultiUserInputRouter started "
            f"user_ids={list(self.core.user_ids)} "
            f'routing_mode="{self.core.routing_mode}" '
            f'active_user_id="{self.core.active_user_id}" '
            f'single_user_id="{self.core.single_user_id}" '
            f"robot_active_user_ids={self.core.robot_active_user_ids} "
            f"user_topic_prefix={self.user_topic_prefix} "
            f"subscribe_only_existing_topics="
            f"{self.subscribe_only_existing_topics} "
            f"timeout_sec={self.core.timeout_sec:.3f} "
            f"target_timeout_sec={self.core.target_timeout_sec:.3f} "
            f"outputs={self.output_topics}"
        )

    def _on_set_parameters(self, parameters):
        updates = {}
        try:
            for parameter in parameters:
                name = parameter.name
                value = parameter.value
                if name == "user_ids":
                    raise ValueError(
                        "user_ids cannot be changed while router is running")
                if name == "routing_mode":
                    updates[name] = validate_routing_mode(value)
                elif name == "active_user_id":
                    user_id = str(value).strip()
                    reason = validate_user_id(user_id, self.core.user_ids)
                    if reason is not None:
                        raise ValueError(reason)
                    updates[name] = user_id
                elif name == "single_user_id":
                    user_id = str(value).strip()
                    reason = validate_user_id(user_id, self.core.user_ids)
                    if reason is not None:
                        raise ValueError(f"single_user_id: {reason}")
                    updates[name] = user_id
                elif name in (
                    "amir_active_user_id",
                    "rover_active_user_id",
                    "drone_active_user_id",
                ):
                    robot_target = name.replace("_active_user_id", "")
                    updates[name] = validate_control_user_value(
                        value, self.core.user_ids, name)
                    updates[f"{name}__target"] = robot_target
                elif name == "timeout_sec":
                    updates[name] = max(float(value), 0.0)
                elif name == "target_timeout_sec":
                    updates[name] = max(float(value), 0.0)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        now_sec = self._now_sec()
        for name, value in updates.items():
            if name.endswith("__target"):
                continue
            if name == "routing_mode":
                self._handle_decision(
                    self.core.set_routing_mode(value, now_sec))
            elif name == "active_user_id":
                self._handle_decision(
                    self.core.set_active_user(value, now_sec))
            elif name == "single_user_id":
                self._handle_decision(
                    self.core.set_single_user(value, now_sec))
            elif name in (
                "amir_active_user_id",
                "rover_active_user_id",
                "drone_active_user_id",
            ):
                robot_target = updates[f"{name}__target"]
                self._handle_decision(
                    self.core.set_robot_active_user(
                        robot_target, value, now_sec))
            elif name == "timeout_sec":
                self.core.set_timeout_sec(value)
                self.get_logger().info(
                    f"timeout_sec updated to {self.core.timeout_sec:.3f}")
            elif name == "target_timeout_sec":
                self._handle_decision(
                    self.core.set_target_timeout_sec(value, now_sec))
                self.get_logger().info(
                    "target_timeout_sec updated to "
                    f"{self.core.target_timeout_sec:.3f}")
        return SetParametersResult(successful=True)

    def discover_user_topics(self):
        graph_topics = {}
        if self.subscribe_only_existing_topics:
            graph_topics = {
                name: types
                for name, types in self.get_topic_names_and_types()
            }

        for user_id in self.core.user_ids:
            for spec in USER_INPUT_SPECS:
                key = (user_id, spec.name)
                if key in self.input_subscriptions:
                    continue
                topic_name = self.user_input_topic(user_id, spec)
                if self.subscribe_only_existing_topics:
                    topic_types = graph_topics.get(topic_name)
                    if topic_types is None:
                        continue
                    if spec.ros_type not in topic_types:
                        self.get_logger().warn(
                            "Refusing user input topic with unexpected type: "
                            f"{topic_name} types={topic_types} "
                            f"expected={spec.ros_type}",
                            throttle_duration_sec=self.log_throttle_sec,
                        )
                        continue
                self.input_subscriptions[key] = self.create_subscription(
                    spec.message_type,
                    topic_name,
                    self._make_input_callback(user_id, spec.name),
                    spec.qos_depth,
                )
                if spec.name == "robot_target":
                    output_text = "router state"
                else:
                    output_text = self.output_topics[spec.name]
                self.get_logger().info(
                    f"Subscribed user input {topic_name} "
                    f"({spec.ros_type}) -> {output_text}")

    def user_input_topic(self, user_id, spec):
        return (
            f"{self.user_topic_prefix}/"
            f"{user_id}/{spec.input_suffix.lstrip('/')}"
        )

    def _make_input_callback(self, user_id, channel):
        def callback(msg):
            self.handle_input(user_id, channel, msg)
        return callback

    def handle_input(self, user_id, channel, msg):
        decision = self.core.route_message(
            user_id=user_id,
            channel=channel,
            msg=msg,
            now_sec=self._now_sec(),
        )
        self._handle_decision(decision, user_id=user_id, channel=channel)
        if decision.accepted and decision.channel:
            self.output_publishers[decision.channel].publish(decision.message)

    def check_timeout(self):
        self._handle_decision(self.core.check_timeouts(self._now_sec()))

    def _handle_decision(self, decision, user_id=None, channel=None):
        for event in decision.events:
            self.get_logger().info(event)
        if not decision.accepted:
            prefix = "Rejected"
            if user_id is not None and channel is not None:
                prefix = f"Rejected {channel} from {user_id}"
            self.get_logger().warn(
                f"{prefix}: {decision.reason}",
                throttle_duration_sec=self.log_throttle_sec,
            )
        if decision.stop_required:
            self.publish_mecanum_stop(decision.stop_reason)

    def publish_mecanum_stop(self, reason):
        self.output_publishers["right_hand_mecanum_input"].publish(
            self.core.make_stop_message())
        self.get_logger().warn(
            f"Forwarded rover STOP on "
            f"{self.output_topics['right_hand_mecanum_input']} "
            f"reason={reason}",
            throttle_duration_sec=self.log_throttle_sec,
        )

    def _topic_parameter(self, parameter_name, default_topic):
        topic = str(self.get_parameter(parameter_name).value).strip()
        if topic.startswith("/"):
            return topic
        self.get_logger().warn(
            f'{parameter_name}="{topic}" is not absolute. '
            f"Using {default_topic}.")
        return default_topic

    @staticmethod
    def _normalize_topic_prefix(prefix):
        text = str(prefix).strip().rstrip("/")
        if not text:
            return "/share/users"
        if not text.startswith("/"):
            text = "/" + text
        return text

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AmirMultiUserInputRouter()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.publish_mecanum_stop("shutdown")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
