"""JointTrajectory message formatting helpers."""

import math

from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from amir_irt_mro.real_safety import (
    DEFAULT_ARM_JOINT_LIMITS,
    DEFAULT_ARM_JOINT_NAMES,
    validate_joint_positions,
    validate_joint_names,
)


def duration_from_seconds(value):
    sec = int(math.floor(max(float(value), 0.0)))
    nanosec = int(round((max(float(value), 0.0) - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def make_real_teleop_joint_trajectory(
        positions,
        time_from_start_sec,
        joint_names=DEFAULT_ARM_JOINT_NAMES):
    names = validate_joint_names(joint_names, DEFAULT_ARM_JOINT_NAMES)
    position_values = [float(value) for value in positions]
    if len(position_values) != len(names):
        raise ValueError(
            "Trajectory position count mismatch: "
            f"positions={len(position_values)}, joint_names={len(names)}")

    msg = JointTrajectory()
    msg.header.stamp.sec = 0
    msg.header.stamp.nanosec = 0
    msg.joint_names = list(names)

    point = JointTrajectoryPoint()
    point.positions = position_values
    point.time_from_start = duration_from_seconds(time_from_start_sec)
    msg.points = [point]
    return msg


def make_real_teleop_buffer_joint_trajectory(
        waypoints,
        times_from_start_sec,
        joint_names=DEFAULT_ARM_JOINT_NAMES):
    names = validate_joint_names(joint_names, DEFAULT_ARM_JOINT_NAMES)
    waypoint_values = [
        [float(value) for value in waypoint]
        for waypoint in waypoints
    ]
    time_values = [float(value) for value in times_from_start_sec]
    if not waypoint_values:
        raise ValueError("Trajectory buffer requires at least one waypoint")
    if len(waypoint_values) != len(time_values):
        raise ValueError(
            "Trajectory buffer waypoint/time count mismatch: "
            f"waypoints={len(waypoint_values)}, times={len(time_values)}")

    previous_time = 0.0
    msg = JointTrajectory()
    msg.header.stamp.sec = 0
    msg.header.stamp.nanosec = 0
    msg.joint_names = list(names)

    for idx, waypoint in enumerate(waypoint_values):
        if len(waypoint) != len(names):
            raise ValueError(
                "Trajectory buffer position count mismatch: "
                f"waypoint={idx}, positions={len(waypoint)}, "
                f"joint_names={len(names)}")
        time_value = time_values[idx]
        if time_value <= previous_time:
            raise ValueError(
                "Trajectory buffer times must be strictly increasing: "
                f"previous={previous_time}, current={time_value}")
        point = JointTrajectoryPoint()
        point.positions = waypoint
        point.time_from_start = duration_from_seconds(time_value)
        msg.points.append(point)
        previous_time = time_value
    return msg


def make_velocity_joint_trajectory(
        velocities,
        time_from_start_sec,
        joint_names=DEFAULT_ARM_JOINT_NAMES):
    names = validate_joint_names(joint_names, DEFAULT_ARM_JOINT_NAMES)
    velocity_values = [float(value) for value in velocities]
    if len(velocity_values) != len(names):
        raise ValueError(
            "Trajectory velocity count mismatch: "
            f"velocities={len(velocity_values)}, joint_names={len(names)}")

    msg = JointTrajectory()
    msg.header.stamp.sec = 0
    msg.header.stamp.nanosec = 0
    msg.joint_names = list(names)

    point = JointTrajectoryPoint()
    point.velocities = velocity_values
    point.time_from_start = duration_from_seconds(time_from_start_sec)
    msg.points = [point]
    return msg


def make_forward_position_command(
        positions,
        joint_names=DEFAULT_ARM_JOINT_NAMES,
        joint_limits=DEFAULT_ARM_JOINT_LIMITS):
    position_values = validate_joint_positions(
        positions, joint_names, joint_limits)
    msg = Float64MultiArray()
    msg.data = position_values
    return msg


def forward_position_log_fields(msg, mode, control_profile):
    return {
        "data": [float(value) for value in msg.data],
        "data_length": len(msg.data),
        "mode": str(mode),
        "control_profile": str(control_profile),
    }


def trajectory_log_fields(msg, mode, control_profile):
    point = msg.points[0]
    last_point = msg.points[-1]
    return {
        "header_stamp": {
            "sec": int(msg.header.stamp.sec),
            "nanosec": int(msg.header.stamp.nanosec),
        },
        "point_count": len(msg.points),
        "positions": [float(value) for value in point.positions],
        "first_waypoint": [float(value) for value in point.positions],
        "last_waypoint": [float(value) for value in last_point.positions],
        "positions_length": len(point.positions),
        "velocities": [float(value) for value in point.velocities],
        "velocities_length": len(point.velocities),
        "time_from_start": {
            "sec": int(point.time_from_start.sec),
            "nanosec": int(point.time_from_start.nanosec),
        },
        "last_time_from_start": {
            "sec": int(last_point.time_from_start.sec),
            "nanosec": int(last_point.time_from_start.nanosec),
        },
        "mode": str(mode),
        "control_profile": str(control_profile),
    }
