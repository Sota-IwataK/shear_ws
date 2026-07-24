#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relative palm-pose IK controller for Amir."""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from amir_irt_mro.real_safety import (
    DEFAULT_ARM_COMMAND_TOPIC,
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_ARM_STATE_TOPIC,
    DEFAULT_FORWARD_POSITION_COMMAND_TOPIC,
    DEFAULT_GRIPPER_COMMAND_TOPIC,
    clamp_joint_step,
    compute_relative_target,
    is_teleop_buffer_profile,
    is_teleop_control_profile,
    is_teleop_forward_position_profile,
    is_teleop_trajectory_profile,
    is_teleop_velocity_profile,
    is_target_within_workspace,
    integrate_joint_reference,
    limit_cartesian_command,
    limit_cartesian_velocity,
    limit_joint_servo_command,
    limit_joint_velocity_command,
    normalize_control_profile,
    normalize_command_unit,
    predict_joint_reference_buffer,
    reference_tracking_error,
    reorder_positions_from_joint_state,
    servo_stop_reason,
    should_invalidate_reference,
    should_reset_reference_for_tracking_error,
    teleop_servo_workspace_policy,
    validate_joint_names,
    workspace_axis_margins,
)
from amir_irt_mro.trajectory_format import (
    forward_position_log_fields,
    make_forward_position_command,
    make_real_teleop_buffer_joint_trajectory,
    make_real_teleop_joint_trajectory,
    make_velocity_joint_trajectory,
    trajectory_log_fields,
)

JOINT_NAMES = list(DEFAULT_ARM_JOINT_NAMES)

L1_OFFSET = 0.248
LINK_LENGTHS = np.array([0.310, 0.310, 0.148], dtype=np.float64)

DEFAULT_JOINTS = np.array([0.0, 1.3535, -2.5596, 0.5270, 0.0], dtype=np.float64)

JOINT_LIMITS = np.array([
    [-2.96706, 2.96706],
    [0.0, 2.356194],
    [-2.792527, 0.0],
    [-2.094395, 1.308997],
    [-2.75762, 2.75762],
], dtype=np.float64)

URDF_SOURCE_FILE = (
    "/home/sota/humbleble_sim_ws/install/amir_description/share/"
    "amir_description/urdf/amir_mecanum3_sim.xacro"
)
URDF_DETAIL_SOURCE_FILE = (
    "/home/sota/humbleble_sim_ws/install/amir_description/share/"
    "amir_description/urdf/amir_for_rover.xacro"
)
URDF_BASE_LINK = "link0_1"
URDF_TIP_LINK = "gripper_base_1"
URDF_JOINT_PARENTS = ["link0_1", "link1_1", "link2_1", "link3_1", "link4_1"]
URDF_JOINT_CHILDREN = ["link1_1", "link2_1", "link3_1", "link4_1", "gripper_base_1"]
URDF_JOINT_ORIGINS_XYZ = np.array([
    [-0.0, -0.001523, -0.097],
    [-0.0, 0.0, 0.097],
    [0.0, 0.31, -0.0],
    [-0.0, 0.31, 0.0],
    [0.0, 0.0405, 0.0],
], dtype=np.float64)
URDF_JOINT_ORIGINS_RPY = np.zeros((5, 3), dtype=np.float64)
URDF_JOINT_AXES = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)


def rpy_to_rot(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_to_rot(axis, angle):
    axis = np.array(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ], dtype=np.float64)


def make_transform(xyz, rpy):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_rot(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    transform[:3, 3] = np.array(xyz, dtype=np.float64)
    return transform


def transform_about_axis(axis, q):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axis_angle_to_rot(axis, float(q))
    return transform


def clamp_array(value, lower, upper):
    return np.minimum(np.maximum(value, lower), upper)


def seconds_now(node):
    return node.get_clock().now().nanoseconds * 1e-9


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def map_range_clamped(v, in_min, in_max, out_min, out_max):
    if abs(in_max - in_min) < 1e-9:
        return 0.5 * (out_min + out_max)
    t = (v - in_min) / (in_max - in_min)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)


def duration_from_seconds(value):
    sec = int(math.floor(max(value, 0.0)))
    nanosec = int(round((max(value, 0.0) - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return Duration(sec=sec, nanosec=nanosec)


def fk_xyz(joints):
    """Amir arm FK for Joint_1..Joint_5 position, base frame, meters."""
    j1, j2, j3, j4, _j5 = [float(v) for v in joints]
    th0 = math.pi / 2.0 - j2
    th1 = j3
    th2 = j4

    r = (LINK_LENGTHS[0] * math.sin(th0) +
         LINK_LENGTHS[1] * math.sin(th0 + th1) +
         LINK_LENGTHS[2] * math.sin(th0 + th1 + th2))
    z = (L1_OFFSET +
         LINK_LENGTHS[0] * math.cos(th0) +
         LINK_LENGTHS[1] * math.cos(th0 + th1) +
         LINK_LENGTHS[2] * math.cos(th0 + th1 + th2))
    x = -r * math.cos(j1)
    y = -r * math.sin(j1)
    return np.array([x, y, z], dtype=np.float64)


def fk_xyz_urdf_like(joints):
    transform = np.eye(4, dtype=np.float64)
    for idx, q in enumerate(joints):
        transform = (
            transform @
            make_transform(URDF_JOINT_ORIGINS_XYZ[idx], URDF_JOINT_ORIGINS_RPY[idx]) @
            transform_about_axis(URDF_JOINT_AXES[idx], q)
        )
    return transform[:3, 3].copy()


def numeric_jacobian(joints, fk_func=fk_xyz, eps=1e-5):
    base = fk_func(joints)
    jac = np.zeros((3, 5), dtype=np.float64)
    for idx in range(5):
        shifted = joints.copy()
        shifted[idx] += eps
        jac[:, idx] = (fk_func(shifted) - base) / eps
    return jac


def lm_ik(target_xyz, seed_joints, max_iters=60, fk_func=fk_xyz):
    joints = seed_joints.copy()
    damping = 2e-3

    for _ in range(max_iters):
        current = fk_func(joints)
        error = target_xyz - current
        if float(np.linalg.norm(error)) < 1e-4:
            break

        jac = numeric_jacobian(joints, fk_func=fk_func)
        lhs = jac.T @ jac + damping * np.eye(5)
        rhs = jac.T @ error
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(lhs) @ rhs

        delta = np.clip(delta, -0.08, 0.08)
        joints = clamp_array(joints + delta, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    final_error = float(np.linalg.norm(target_xyz - fk_func(joints)))
    return joints, final_error


class AmirPalmIkNode(Node):
    def __init__(self):
        super().__init__("amir_palm_ik_node")

        self.declare_parameter("mode", "sim")
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("kinematics_model", "simple_3link")
        self.declare_parameter("enable_kinematics_target_adapter", True)
        self.declare_parameter(
            "kinematics_target_adapter_mode", "control_to_urdf_y_forward")
        self.declare_parameter("palm_pose_topic", "/palm_pose_world")
        self.declare_parameter(
            "control_center_topic", "/palm_pose_control_center_world")
        self.declare_parameter("use_control_center", True)
        self.declare_parameter("require_control_center", True)
        self.declare_parameter("target_mapping_mode", "relative")
        self.declare_parameter("palm_motion_gain", 1.0)
        self.declare_parameter("scale_x", 0.6)
        self.declare_parameter("scale_y", 0.3)
        self.declare_parameter("scale_z", 0.6)
        self.declare_parameter("deadzone_m", 0.005)
        self.declare_parameter("max_delta_norm_m", 0.30)
        self.declare_parameter("max_target_step_m", 0.003)
        self.declare_parameter("lowpass_alpha", 0.25)
        self.declare_parameter("input_timeout_sec", 0.5)
        self.declare_parameter("target_timeout_sec", 0.5)
        self.declare_parameter("workspace_min", [0.12, -0.25, 0.05])
        self.declare_parameter("workspace_max", [0.45, 0.25, 0.45])
        self.declare_parameter("workspace_x_min", 0.12)
        self.declare_parameter("workspace_x_max", 0.45)
        self.declare_parameter("workspace_y_min", -0.25)
        self.declare_parameter("workspace_y_max", 0.25)
        self.declare_parameter("workspace_z_min", 0.05)
        self.declare_parameter("workspace_z_max", 0.45)
        self.declare_parameter("enable_reach_radius_limit", True)
        self.declare_parameter("reach_radius_max_m", 0.85)
        self.declare_parameter("reach_radius_margin_m", 0.03)
        self.declare_parameter("reach_radius_origin_x", 0.0)
        self.declare_parameter("reach_radius_origin_y", 0.0)
        self.declare_parameter("reach_radius_origin_z", 0.0)
        self.declare_parameter("trajectory_time_from_start_sec", 0.2)
        self.declare_parameter("trajectory_time_sec", 0.0)
        self.declare_parameter("command_unit", "rad")
        self.declare_parameter("arm_command_topic", DEFAULT_ARM_COMMAND_TOPIC)
        self.declare_parameter(
            "forward_position_command_topic",
            DEFAULT_FORWARD_POSITION_COMMAND_TOPIC)
        self.declare_parameter("arm_state_topic", DEFAULT_ARM_STATE_TOPIC)
        self.declare_parameter("gripper_command_topic", DEFAULT_GRIPPER_COMMAND_TOPIC)
        self.declare_parameter("joint_names", JOINT_NAMES)
        self.declare_parameter("max_joint_step_rad", 0.01)
        self.declare_parameter("publish_rate_hz", 0.0)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("control_profile", "safe_trajectory")
        self.declare_parameter("servo_rate_hz", 5.0)
        self.declare_parameter("servo_horizon_sec", 0.25)
        self.declare_parameter("reference_servo_rate_hz", 50.0)
        self.declare_parameter("reference_tracking_error_threshold_rad", 0.08)
        self.declare_parameter("forward_position_publish_rate_hz", 50.0)
        self.declare_parameter("trajectory_buffer_update_rate_hz", 10.0)
        self.declare_parameter("trajectory_buffer_horizon_sec", 0.30)
        self.declare_parameter("trajectory_buffer_point_dt_sec", 0.05)
        self.declare_parameter("max_cartesian_velocity_mps", 0.05)
        self.declare_parameter("max_joint_velocity_radps", 0.15)
        self.declare_parameter("max_cartesian_acceleration_mps2", 0.20)
        self.declare_parameter("max_joint_acceleration_radps2", 0.50)
        self.declare_parameter("velocity_servo_hardware_confirmed", False)
        self.declare_parameter("velocity_servo_time_from_start_sec", 0.05)
        self.declare_parameter("deadman_enabled", False)
        self.declare_parameter("deadman_timeout_sec", 0.5)
        self.declare_parameter("deadman_topic", "/deadman_enabled")
        self.declare_parameter("joint_state_timeout_sec", 0.5)
        self.declare_parameter("servo_ik_tolerance_m", 0.02)
        self.declare_parameter("axis_map", [2, 0, 1])
        self.declare_parameter("axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("palm_to_robot_axis_map", [2, 0, 1])
        self.declare_parameter("palm_to_robot_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("palm_axis_gain", [1.0, 1.0, 1.0])
        self.declare_parameter("mr_x_min", -0.30)
        self.declare_parameter("mr_x_max", 0.30)
        self.declare_parameter("mr_y_min", -0.50)
        self.declare_parameter("mr_y_max", 0.20)
        self.declare_parameter("mr_z_min", 0.00)
        self.declare_parameter("mr_z_max", 0.70)
        self.declare_parameter("abs_robot_x_min", 0.20)
        self.declare_parameter("abs_robot_x_max", 0.55)
        self.declare_parameter("abs_robot_y_min", -0.15)
        self.declare_parameter("abs_robot_y_max", 0.15)
        self.declare_parameter("abs_robot_z_min", 0.35)
        self.declare_parameter("abs_robot_z_max", 0.90)
        self.declare_parameter("debug_target_step_x", 0.0)
        self.declare_parameter("palm_target_mode", "relative")
        self.declare_parameter("cylindrical_r_gain", 1.0)
        self.declare_parameter("cylindrical_z_gain", 1.0)
        self.declare_parameter("cylindrical_y_gain", 0.0)
        self.declare_parameter("cylindrical_r_offset", 0.0)
        self.declare_parameter("cylindrical_z_offset", 0.0)
        self.declare_parameter("cylindrical_y_offset", 0.0)
        self.declare_parameter("cylindrical_min_r", 0.05)
        self.declare_parameter("cylindrical_max_r", 0.75)
        self.declare_parameter("cylindrical_min_z", 0.00)
        self.declare_parameter("cylindrical_max_z", 0.80)
        self.declare_parameter("enable_range_scaled_palm", False)
        self.declare_parameter("human_motion_range", [0.35, 0.30, 0.30])
        self.declare_parameter("robot_motion_range", [0.60, 0.40, 0.40])
        self.declare_parameter("range_scale_gain", 1.0)
        self.declare_parameter("range_scale_max", [3.0, 3.0, 3.0])
        self.declare_parameter("enable_extension_bias", False)
        self.declare_parameter("extension_bias_gain", 0.05)
        self.declare_parameter("extension_trigger_x", 0.02)
        self.declare_parameter("extension_trigger_norm", 0.02)
        self.declare_parameter("extension_use_motion_norm", True)
        self.declare_parameter("extension_posture", [0.0, 1.20, -2.20, -0.70, 0.0])
        self.declare_parameter("joint_limit_avoidance_gain", 0.02)
        self.declare_parameter("enable_reach_approach", False)
        self.declare_parameter("reach_trigger_error", 0.03)
        self.declare_parameter("reach_full_error", 0.15)
        self.declare_parameter("reach_bias_gain", 0.25)
        self.declare_parameter("reach_posture", [0.0, 2.10, -0.40, -0.15, 0.0])
        self.declare_parameter("reach_directional_yaw", True)
        self.declare_parameter("reach_yaw_trigger_norm", 0.01)

        self.mode = str(self.get_parameter("mode").value).strip().lower()
        if self.mode not in ("sim", "gazebo", "real"):
            self.get_logger().warn(f'Invalid mode "{self.mode}". Falling back to sim.')
            self.mode = "sim"

        requested_control_profile = str(
            self.get_parameter("control_profile").value).strip().lower()
        try:
            self.control_profile = normalize_control_profile(
                requested_control_profile)
        except ValueError:
            self.get_logger().warn(
                f'Invalid control_profile "{requested_control_profile}". '
                "Using safe_trajectory.")
            self.control_profile = "safe_trajectory"
        if requested_control_profile == "teleop_servo":
            self.get_logger().warn(
                "control_profile:=teleop_servo is a legacy alias. "
                "Using teleop_trajectory_stream.")
        if is_teleop_control_profile(self.control_profile) and self.mode != "real":
            self.get_logger().warn(
                f"control_profile:={self.control_profile} is real-mode only. "
                "Using safe_trajectory.")
            self.control_profile = "safe_trajectory"

        try:
            self.command_unit = normalize_command_unit(
                self.get_parameter("command_unit").value)
        except ValueError:
            self.get_logger().warn(
                f'Invalid command_unit "{self.get_parameter("command_unit").value}". '
                "Falling back to rad.")
            self.command_unit = "rad"

        self.servo_rate_hz = max(
            float(self.get_parameter("servo_rate_hz").value), 0.1)
        self.servo_horizon_sec = max(
            float(self.get_parameter("servo_horizon_sec").value), 0.02)
        self.reference_servo_rate_hz = max(
            float(self.get_parameter("reference_servo_rate_hz").value), 1.0)
        self.reference_tracking_error_threshold_rad = max(
            float(self.get_parameter(
                "reference_tracking_error_threshold_rad").value), 0.0)
        self.forward_position_publish_rate_hz = max(
            float(self.get_parameter(
                "forward_position_publish_rate_hz").value), 0.1)
        self.trajectory_buffer_update_rate_hz = max(
            float(self.get_parameter(
                "trajectory_buffer_update_rate_hz").value), 0.1)
        self.trajectory_buffer_horizon_sec = max(
            float(self.get_parameter(
                "trajectory_buffer_horizon_sec").value), 0.1)
        self.trajectory_buffer_point_dt_sec = max(
            float(self.get_parameter(
                "trajectory_buffer_point_dt_sec").value), 0.01)
        if is_teleop_buffer_profile(self.control_profile):
            if (self.trajectory_buffer_update_rate_hz >
                    self.reference_servo_rate_hz):
                self.get_logger().warn(
                    "trajectory_buffer_update_rate_hz is higher than "
                    "reference_servo_rate_hz. Clamping update rate to "
                    f"{self.reference_servo_rate_hz:.2f}Hz.")
                self.trajectory_buffer_update_rate_hz = (
                    self.reference_servo_rate_hz)
            if self.trajectory_buffer_update_rate_hz > 10.0:
                self.get_logger().warn(
                    "teleop_trajectory_buffer update rate is capped at "
                    "10Hz for real JTC safety "
                    f"(requested="
                    f"{self.trajectory_buffer_update_rate_hz:.2f}Hz).")
                self.trajectory_buffer_update_rate_hz = 10.0
            if self.trajectory_buffer_point_dt_sec > self.trajectory_buffer_horizon_sec:
                self.get_logger().warn(
                    "trajectory_buffer_point_dt_sec is larger than "
                    "trajectory_buffer_horizon_sec. Using horizon as dt.")
                self.trajectory_buffer_point_dt_sec = (
                    self.trajectory_buffer_horizon_sec)
        elif is_teleop_trajectory_profile(self.control_profile):
            if self.servo_rate_hz < 5.0 or self.servo_rate_hz > 10.0:
                self.get_logger().warn(
                    "teleop_trajectory_stream requires "
                    "servo_rate_hz between 5 and 10Hz. "
                    f"Clamping requested={self.servo_rate_hz:.2f}Hz.")
                self.servo_rate_hz = min(max(self.servo_rate_hz, 5.0), 10.0)
            if self.servo_horizon_sec < 0.15 or self.servo_horizon_sec > 0.30:
                self.get_logger().warn(
                    "teleop_trajectory_stream requires "
                    "servo_horizon_sec between 0.15 and 0.30s. "
                    f"Clamping requested={self.servo_horizon_sec:.3f}s.")
                self.servo_horizon_sec = min(
                    max(self.servo_horizon_sec, 0.15), 0.30)
        if is_teleop_velocity_profile(self.control_profile):
            if self.servo_rate_hz < 20.0:
                self.get_logger().warn(
                    "teleop_velocity_servo requires servo_rate_hz >= 20Hz. "
                    f"Using 20Hz instead of {self.servo_rate_hz:.2f}Hz.")
                self.servo_rate_hz = 20.0
        self.max_cartesian_velocity_mps = max(
            float(self.get_parameter("max_cartesian_velocity_mps").value), 0.0)
        self.max_joint_velocity_radps = max(
            float(self.get_parameter("max_joint_velocity_radps").value), 0.0)
        self.max_cartesian_acceleration_mps2 = max(
            float(self.get_parameter("max_cartesian_acceleration_mps2").value), 0.0)
        self.max_joint_acceleration_radps2 = max(
            float(self.get_parameter("max_joint_acceleration_radps2").value), 0.0)
        self.velocity_servo_hardware_confirmed = as_bool(
            self.get_parameter("velocity_servo_hardware_confirmed").value)
        self.velocity_servo_time_from_start_sec = max(
            float(self.get_parameter("velocity_servo_time_from_start_sec").value),
            1.0 / max(self.servo_rate_hz, 1e-6))
        self.deadman_enabled = as_bool(self.get_parameter("deadman_enabled").value)
        self.deadman_timeout_sec = max(
            float(self.get_parameter("deadman_timeout_sec").value), 0.0)
        self.deadman_topic = str(self.get_parameter("deadman_topic").value).strip()
        self.joint_state_timeout_sec = max(
            float(self.get_parameter("joint_state_timeout_sec").value), 0.0)
        self.servo_ik_tolerance_m = max(
            float(self.get_parameter("servo_ik_tolerance_m").value), 0.0)

        base_control_rate_hz = max(float(self.get_parameter("control_rate_hz").value), 0.1)
        requested_publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if is_teleop_control_profile(self.control_profile):
            if is_teleop_buffer_profile(self.control_profile):
                self.control_rate_hz = self.trajectory_buffer_update_rate_hz
            elif is_teleop_forward_position_profile(self.control_profile):
                self.control_rate_hz = self.forward_position_publish_rate_hz
            else:
                self.control_rate_hz = self.servo_rate_hz
        else:
            self.control_rate_hz = (
                requested_publish_rate_hz
                if requested_publish_rate_hz > 0.0 else
                base_control_rate_hz
            )
        self.control_rate_hz = max(self.control_rate_hz, 0.1)
        if (self.mode == "real" and
                is_teleop_trajectory_profile(self.control_profile) and
                self.control_rate_hz > 10.0):
            self.get_logger().warn(
                "real teleop trajectory profile is capped at 10Hz for safety "
                f"(requested={self.control_rate_hz:.2f}Hz).")
            self.control_rate_hz = 10.0
        elif (self.mode == "real" and
                not is_teleop_control_profile(self.control_profile) and
                self.control_rate_hz > 10.0):
            self.get_logger().warn(
                "real safe_trajectory publish_rate_hz is capped at 10Hz "
                f"for safety (requested={self.control_rate_hz:.2f}Hz).")
            self.control_rate_hz = 10.0
        dry_run_param = as_bool(self.get_parameter("dry_run").value)
        self.dry_run = dry_run_param if self.mode == "real" else False
        if self.mode == "real" and self.dry_run:
            self.get_logger().warn(
                "real mode dry_run is enabled; JointTrajectory commands are logged "
                "but not published until dry_run:=false is set explicitly.")
        self.kinematics_model = str(
            self.get_parameter("kinematics_model").value).strip().lower()
        self.enable_kinematics_target_adapter = as_bool(
            self.get_parameter("enable_kinematics_target_adapter").value)
        self.kinematics_target_adapter_mode = str(
            self.get_parameter("kinematics_target_adapter_mode").value).strip().lower()
        self.palm_pose_topic = str(self.get_parameter("palm_pose_topic").value).strip()
        self.control_center_topic = str(
            self.get_parameter("control_center_topic").value).strip()
        self.arm_command_topic = str(self.get_parameter("arm_command_topic").value).strip()
        self.forward_position_command_topic = str(
            self.get_parameter("forward_position_command_topic").value).strip()
        self.arm_state_topic = str(self.get_parameter("arm_state_topic").value).strip()
        self.gripper_command_topic = str(
            self.get_parameter("gripper_command_topic").value).strip()
        self.joint_names = [
            str(name) for name in self.get_parameter("joint_names").value]
        try:
            validate_joint_names(self.joint_names, JOINT_NAMES)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            raise
        if not self.palm_pose_topic.startswith("/"):
            self.get_logger().warn(
                f'palm_pose_topic "{self.palm_pose_topic}" should be absolute. '
                "Using /palm_pose_world.")
            self.palm_pose_topic = "/palm_pose_world"
        if not self.control_center_topic.startswith("/"):
            self.get_logger().warn(
                f'control_center_topic "{self.control_center_topic}" should be absolute. '
                "Using /palm_pose_control_center_world.")
            self.control_center_topic = "/palm_pose_control_center_world"
        if not self.arm_command_topic.startswith("/"):
            self.get_logger().warn(
                f'arm_command_topic "{self.arm_command_topic}" should be absolute. '
                f"Using {DEFAULT_ARM_COMMAND_TOPIC}.")
            self.arm_command_topic = DEFAULT_ARM_COMMAND_TOPIC
        if not self.forward_position_command_topic.startswith("/"):
            self.get_logger().warn(
                "forward_position_command_topic "
                f'"{self.forward_position_command_topic}" should be absolute. '
                f"Using {DEFAULT_FORWARD_POSITION_COMMAND_TOPIC}.")
            self.forward_position_command_topic = (
                DEFAULT_FORWARD_POSITION_COMMAND_TOPIC)
        if not self.arm_state_topic.startswith("/"):
            self.get_logger().warn(
                f'arm_state_topic "{self.arm_state_topic}" should be absolute. '
                f"Using {DEFAULT_ARM_STATE_TOPIC}.")
            self.arm_state_topic = DEFAULT_ARM_STATE_TOPIC
        if not self.gripper_command_topic.startswith("/"):
            self.get_logger().warn(
                f'gripper_command_topic "{self.gripper_command_topic}" should be absolute. '
                f"Using {DEFAULT_GRIPPER_COMMAND_TOPIC}.")
            self.gripper_command_topic = DEFAULT_GRIPPER_COMMAND_TOPIC
        if not self.deadman_topic.startswith("/"):
            self.get_logger().warn(
                f'deadman_topic "{self.deadman_topic}" should be absolute. '
                "Using /deadman_enabled.")
            self.deadman_topic = "/deadman_enabled"
        self.use_control_center = as_bool(self.get_parameter("use_control_center").value)
        self.require_control_center = as_bool(
            self.get_parameter("require_control_center").value)
        self.target_mapping_mode = str(
            self.get_parameter("target_mapping_mode").value).strip().lower()
        self.absolute_axis_mapping = "unity_z_to_x_unity_x_to_y_unity_y_to_z"
        self.palm_motion_gain = float(self.get_parameter("palm_motion_gain").value)
        self.scale_xyz = np.array([
            float(self.get_parameter("scale_x").value),
            float(self.get_parameter("scale_y").value),
            float(self.get_parameter("scale_z").value),
        ], dtype=np.float64)
        self.deadzone_m = max(float(self.get_parameter("deadzone_m").value), 0.0)
        self.max_delta_norm_m = max(float(self.get_parameter("max_delta_norm_m").value), 0.0)
        self.max_target_step_m = max(float(self.get_parameter("max_target_step_m").value), 0.0)
        self.lowpass_alpha = float(np.clip(
            float(self.get_parameter("lowpass_alpha").value), 0.0, 1.0))
        self.input_timeout_sec = max(float(self.get_parameter("input_timeout_sec").value), 0.0)
        self.target_timeout_sec = max(
            float(self.get_parameter("target_timeout_sec").value), 0.0)
        workspace_min_param = np.array(
            self.get_parameter("workspace_min").value, dtype=np.float64)
        workspace_max_param = np.array(
            self.get_parameter("workspace_max").value, dtype=np.float64)
        self.workspace_min = np.array([
            float(self.get_parameter("workspace_x_min").value),
            float(self.get_parameter("workspace_y_min").value),
            float(self.get_parameter("workspace_z_min").value),
        ], dtype=np.float64)
        self.workspace_max = np.array([
            float(self.get_parameter("workspace_x_max").value),
            float(self.get_parameter("workspace_y_max").value),
            float(self.get_parameter("workspace_z_max").value),
        ], dtype=np.float64)
        if (np.allclose(self.workspace_min, [0.12, -0.25, 0.05]) and
                np.allclose(self.workspace_max, [0.45, 0.25, 0.45]) and
                workspace_min_param.shape == (3,) and workspace_max_param.shape == (3,) and
                (not np.allclose(workspace_min_param, [0.12, -0.25, 0.05]) or
                 not np.allclose(workspace_max_param, [0.45, 0.25, 0.45]))):
            self.workspace_min = workspace_min_param
            self.workspace_max = workspace_max_param
        self.enable_reach_radius_limit = as_bool(
            self.get_parameter("enable_reach_radius_limit").value)
        self.reach_radius_max_m = max(
            float(self.get_parameter("reach_radius_max_m").value), 0.0)
        self.reach_radius_margin_m = max(
            float(self.get_parameter("reach_radius_margin_m").value), 0.0)
        self.reach_radius_origin = np.array([
            float(self.get_parameter("reach_radius_origin_x").value),
            float(self.get_parameter("reach_radius_origin_y").value),
            float(self.get_parameter("reach_radius_origin_z").value),
        ], dtype=np.float64)
        self.effective_reach_radius = max(
            self.reach_radius_max_m - self.reach_radius_margin_m, 0.0)
        legacy_trajectory_time_sec = float(
            self.get_parameter("trajectory_time_from_start_sec").value)
        requested_trajectory_time_sec = float(
            self.get_parameter("trajectory_time_sec").value)
        self.trajectory_time_from_start_sec = (
            requested_trajectory_time_sec
            if requested_trajectory_time_sec > 0.0 else
            legacy_trajectory_time_sec
        )
        if self.mode == "real" and self.trajectory_time_from_start_sec < 2.0:
            self.get_logger().warn(
                "real mode trajectory_time_sec is raised to 2.0s for safety "
                f"(requested={self.trajectory_time_from_start_sec:.3f}s).")
            self.trajectory_time_from_start_sec = 2.0
        self.max_joint_step_rad = max(
            float(self.get_parameter("max_joint_step_rad").value), 0.0)
        if self.mode == "real" and self.max_joint_step_rad > 0.01:
            self.get_logger().warn(
                "real mode max_joint_step_rad is capped at 0.01rad for safety "
                f"(requested={self.max_joint_step_rad:.5f}rad).")
            self.max_joint_step_rad = 0.01
        if self.mode == "real" and self.command_unit == "mrad":
            self.get_logger().warn(
                "command_unit:=mrad is kept for debug arrays only on the "
                "JointTrajectory path. /arm_controller/joint_trajectory "
                "positions are published in rad because ros2_control expects rad.")
        axis_map = np.array(self.get_parameter("axis_map").value, dtype=np.int64)
        axis_sign = np.array(self.get_parameter("axis_sign").value, dtype=np.float64)
        legacy_axis_map = np.array(
            self.get_parameter("palm_to_robot_axis_map").value, dtype=np.int64)
        legacy_axis_sign = np.array(
            self.get_parameter("palm_to_robot_axis_sign").value, dtype=np.float64)
        if (np.array_equal(axis_map, np.array([2, 0, 1], dtype=np.int64)) and
                legacy_axis_map.shape == (3,) and
                not np.array_equal(legacy_axis_map, np.array([2, 0, 1], dtype=np.int64))):
            axis_map = legacy_axis_map
        if (axis_sign.shape == (3,) and legacy_axis_sign.shape == (3,) and
                np.allclose(axis_sign, [1.0, 1.0, 1.0]) and
                not np.allclose(legacy_axis_sign, [1.0, 1.0, 1.0])):
            axis_sign = legacy_axis_sign
        self.palm_to_robot_axis_map = axis_map
        self.palm_to_robot_axis_sign = axis_sign
        self.palm_axis_gain = np.array(
            self.get_parameter("palm_axis_gain").value, dtype=np.float64)
        self.mr_min = np.array([
            float(self.get_parameter("mr_x_min").value),
            float(self.get_parameter("mr_y_min").value),
            float(self.get_parameter("mr_z_min").value),
        ], dtype=np.float64)
        self.mr_max = np.array([
            float(self.get_parameter("mr_x_max").value),
            float(self.get_parameter("mr_y_max").value),
            float(self.get_parameter("mr_z_max").value),
        ], dtype=np.float64)
        self.abs_robot_min = np.array([
            float(self.get_parameter("abs_robot_x_min").value),
            float(self.get_parameter("abs_robot_y_min").value),
            float(self.get_parameter("abs_robot_z_min").value),
        ], dtype=np.float64)
        self.abs_robot_max = np.array([
            float(self.get_parameter("abs_robot_x_max").value),
            float(self.get_parameter("abs_robot_y_max").value),
            float(self.get_parameter("abs_robot_z_max").value),
        ], dtype=np.float64)
        self.debug_target_step_x = float(self.get_parameter("debug_target_step_x").value)
        self.palm_target_mode = str(self.get_parameter("palm_target_mode").value).strip().lower()
        self.cylindrical_r_gain = float(self.get_parameter("cylindrical_r_gain").value)
        self.cylindrical_z_gain = float(self.get_parameter("cylindrical_z_gain").value)
        self.cylindrical_y_gain = float(self.get_parameter("cylindrical_y_gain").value)
        self.cylindrical_r_offset = float(self.get_parameter("cylindrical_r_offset").value)
        self.cylindrical_z_offset = float(self.get_parameter("cylindrical_z_offset").value)
        self.cylindrical_y_offset = float(self.get_parameter("cylindrical_y_offset").value)
        self.cylindrical_min_r = float(self.get_parameter("cylindrical_min_r").value)
        self.cylindrical_max_r = float(self.get_parameter("cylindrical_max_r").value)
        self.cylindrical_min_z = float(self.get_parameter("cylindrical_min_z").value)
        self.cylindrical_max_z = float(self.get_parameter("cylindrical_max_z").value)
        self.enable_range_scaled_palm = as_bool(
            self.get_parameter("enable_range_scaled_palm").value)
        self.human_motion_range = np.array(
            self.get_parameter("human_motion_range").value, dtype=np.float64)
        self.robot_motion_range = np.array(
            self.get_parameter("robot_motion_range").value, dtype=np.float64)
        self.range_scale_gain = float(self.get_parameter("range_scale_gain").value)
        self.range_scale_max = np.array(
            self.get_parameter("range_scale_max").value, dtype=np.float64)
        self.enable_extension_bias = as_bool(self.get_parameter("enable_extension_bias").value)
        self.extension_bias_gain = max(
            float(self.get_parameter("extension_bias_gain").value), 0.0)
        self.extension_trigger_x = max(
            float(self.get_parameter("extension_trigger_x").value), 0.0)
        self.extension_trigger_norm = max(
            float(self.get_parameter("extension_trigger_norm").value), 0.0)
        self.extension_use_motion_norm = as_bool(
            self.get_parameter("extension_use_motion_norm").value)
        self.extension_posture = np.array(
            self.get_parameter("extension_posture").value, dtype=np.float64)
        self.joint_limit_avoidance_gain = max(
            float(self.get_parameter("joint_limit_avoidance_gain").value), 0.0)
        self.enable_reach_approach = as_bool(
            self.get_parameter("enable_reach_approach").value)
        self.reach_trigger_error = max(
            float(self.get_parameter("reach_trigger_error").value), 0.0)
        self.reach_full_error = max(
            float(self.get_parameter("reach_full_error").value), self.reach_trigger_error)
        self.reach_bias_gain = max(float(self.get_parameter("reach_bias_gain").value), 0.0)
        self.reach_posture = np.array(
            self.get_parameter("reach_posture").value, dtype=np.float64)
        self.reach_directional_yaw = as_bool(
            self.get_parameter("reach_directional_yaw").value)
        self.reach_yaw_trigger_norm = max(
            float(self.get_parameter("reach_yaw_trigger_norm").value), 0.0)

        if self.workspace_min.shape != (3,) or self.workspace_max.shape != (3,):
            self.get_logger().warn("workspace_min/max must be length 3. Using defaults.")
            self.workspace_min = np.array([0.12, -0.25, 0.05], dtype=np.float64)
            self.workspace_max = np.array([0.45, 0.25, 0.45], dtype=np.float64)
        if np.any(self.workspace_min > self.workspace_max):
            self.get_logger().warn("workspace_min is greater than workspace_max. Swapping bounds.")
            low = np.minimum(self.workspace_min, self.workspace_max)
            high = np.maximum(self.workspace_min, self.workspace_max)
            self.workspace_min = low
            self.workspace_max = high
        if self.kinematics_model not in ("simple_3link", "urdf_like"):
            self.get_logger().warn(
                f'Invalid kinematics_model "{self.kinematics_model}". Using simple_3link.')
            self.kinematics_model = "simple_3link"
        if self.kinematics_target_adapter_mode not in (
                "none", "control_to_urdf_y_forward"):
            self.get_logger().warn(
                f'Invalid kinematics_target_adapter_mode '
                f'"{self.kinematics_target_adapter_mode}". Using control_to_urdf_y_forward.')
            self.kinematics_target_adapter_mode = "control_to_urdf_y_forward"
        if self.target_mapping_mode not in ("relative", "absolute"):
            self.get_logger().warn(
                f'Invalid target_mapping_mode "{self.target_mapping_mode}". Using relative.')
            self.target_mapping_mode = "relative"
        if (is_teleop_control_profile(self.control_profile) and
                self.target_mapping_mode != "relative"):
            self.get_logger().error(
                f"{self.control_profile} requires target_mapping_mode:=relative. "
                "Using safe_trajectory.")
            self.control_profile = "safe_trajectory"
        if np.any(self.mr_min > self.mr_max):
            self.get_logger().warn("MR min range is greater than max. Swapping bounds.")
            low = np.minimum(self.mr_min, self.mr_max)
            high = np.maximum(self.mr_min, self.mr_max)
            self.mr_min = low
            self.mr_max = high
        if np.any(self.abs_robot_min > self.abs_robot_max):
            self.get_logger().warn(
                "Absolute robot min range is greater than max. Swapping bounds.")
            low = np.minimum(self.abs_robot_min, self.abs_robot_max)
            high = np.maximum(self.abs_robot_min, self.abs_robot_max)
            self.abs_robot_min = low
            self.abs_robot_max = high
        if (self.palm_to_robot_axis_map.shape != (3,) or
                sorted([int(v) for v in self.palm_to_robot_axis_map.tolist()]) != [0, 1, 2]):
            self.get_logger().warn(
                "axis_map must be a permutation of [0, 1, 2]. Using [2, 0, 1].")
            self.palm_to_robot_axis_map = np.array([2, 0, 1], dtype=np.int64)
        if self.palm_to_robot_axis_sign.shape != (3,):
            self.get_logger().warn("axis_sign must be length 3. Using [1.0, 1.0, 1.0].")
            self.palm_to_robot_axis_sign = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        if not np.allclose(np.abs(self.palm_to_robot_axis_sign), [1.0, 1.0, 1.0]):
            self.get_logger().warn(
                "axis_sign is treated as signs only; normalizing values to +/-1.")
        self.palm_to_robot_axis_sign = np.where(
            self.palm_to_robot_axis_sign < 0.0, -1.0, 1.0).astype(np.float64)
        if self.palm_axis_gain.shape != (3,):
            self.get_logger().warn("palm_axis_gain must be length 3. Using [1.0, 1.0, 1.0].")
            self.palm_axis_gain = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        if self.palm_target_mode not in ("relative", "cylindrical_absolute"):
            self.get_logger().warn(
                f'Invalid palm_target_mode "{self.palm_target_mode}". Using relative.')
            self.palm_target_mode = "relative"
        if self.cylindrical_min_r > self.cylindrical_max_r:
            self.cylindrical_min_r, self.cylindrical_max_r = (
                self.cylindrical_max_r, self.cylindrical_min_r)
        if self.cylindrical_min_z > self.cylindrical_max_z:
            self.cylindrical_min_z, self.cylindrical_max_z = (
                self.cylindrical_max_z, self.cylindrical_min_z)
        if self.human_motion_range.shape != (3,):
            self.get_logger().warn("human_motion_range must be length 3. "
                                   "Using [0.35, 0.30, 0.30].")
            self.human_motion_range = np.array([0.35, 0.30, 0.30], dtype=np.float64)
        if self.robot_motion_range.shape != (3,):
            self.get_logger().warn("robot_motion_range must be length 3. "
                                   "Using [0.60, 0.40, 0.40].")
            self.robot_motion_range = np.array([0.60, 0.40, 0.40], dtype=np.float64)
        if self.range_scale_max.shape != (3,):
            self.get_logger().warn("range_scale_max must be length 3. Using [3.0, 3.0, 3.0].")
            self.range_scale_max = np.array([3.0, 3.0, 3.0], dtype=np.float64)
        self.human_motion_range = np.maximum(self.human_motion_range, 1e-6)
        self.robot_motion_range = np.maximum(self.robot_motion_range, 0.0)
        self.range_scale_max = np.maximum(self.range_scale_max, 0.0)
        if self.extension_posture.shape != (5,):
            self.get_logger().warn("extension_posture must be length 5. "
                                   "Using [0.0, 1.20, -2.20, -0.70, 0.0].")
            self.extension_posture = np.array([0.0, 1.20, -2.20, -0.70, 0.0],
                                             dtype=np.float64)
        self.extension_posture = clamp_array(
            self.extension_posture, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        if self.reach_posture.shape != (5,):
            self.get_logger().warn("reach_posture must be length 5. "
                                   "Using [0.0, 2.10, -0.40, -0.15, 0.0].")
            self.reach_posture = np.array([0.0, 2.10, -0.40, -0.15, 0.0],
                                         dtype=np.float64)
        self.reach_posture = clamp_array(
            self.reach_posture, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        self.current_joints = DEFAULT_JOINTS.copy()
        self.last_command_joints = DEFAULT_JOINTS.copy()
        self.palm_xyz = None
        self.last_palm_time = None
        self.palm_anchor_xyz = None
        self.ee_anchor_xyz = None
        self.has_control_center = False
        self.has_anchor = False
        self.control_center_mr = np.zeros(3, dtype=np.float64)
        self.ee_center_robot = np.zeros(3, dtype=np.float64)
        self.filtered_target_xyz = None
        self.r_robot_from_mr = self.make_axis_transform_matrix()
        self.last_timeout_flag = 0.0
        self.last_target_xyz = None
        self.last_error_norm = float("nan")
        self.last_clamped_flag = 0.0
        self.last_palm_delta_raw = np.zeros(3, dtype=np.float64)
        self.last_palm_delta_robot = np.zeros(3, dtype=np.float64)
        self.last_palm_now_mr = np.zeros(3, dtype=np.float64)
        self.last_scaled_delta = np.zeros(3, dtype=np.float64)
        self.last_target_raw_xyz = np.zeros(3, dtype=np.float64)
        self.last_target_before_safety_xyz = np.zeros(3, dtype=np.float64)
        self.last_target_control_xyz = np.zeros(3, dtype=np.float64)
        self.last_target_kinematics_xyz = np.zeros(3, dtype=np.float64)
        self.last_target_delta = np.zeros(3, dtype=np.float64)
        self.last_reach_radius = 0.0
        self.last_effective_reach_radius = self.effective_reach_radius
        self.last_reach_radius_clamped_flag = 0.0
        self.last_range_scale = np.ones(3, dtype=np.float64)
        self.last_palm_abs_robot = np.zeros(3, dtype=np.float64)
        self.last_r_palm = 0.0
        self.last_y_palm = 0.0
        self.last_z_palm = 0.0
        self.last_r_target = 0.0
        self.last_y_target = 0.0
        self.last_z_target = 0.0
        self.range_scaled_ignored_warned = False
        self.last_debug_log_time = 0.0
        self.last_q_current = DEFAULT_JOINTS.copy()
        self.last_q_solution = DEFAULT_JOINTS.copy()
        self.last_dq = np.zeros(5, dtype=np.float64)
        self.last_fk_before = self.fk_xyz(DEFAULT_JOINTS)
        self.last_fk_after = self.fk_xyz(DEFAULT_JOINTS)
        self.last_error_before = float("nan")
        self.last_extension_weight = 0.0
        self.last_motion_norm = 0.0
        self.last_limit_hit = 0.0
        self.last_reach_vec = np.zeros(3, dtype=np.float64)
        self.last_reach_error = 0.0
        self.last_reach_weight = 0.0
        self.last_q_reach = self.reach_posture.copy()
        self.last_desired_reach_yaw = 0.0
        self.joint_state_received = False
        self.skip_next_command_after_anchor = False
        self.anchor_q = None
        self.anchor_error_norm = float("inf")
        self.last_target_minus_anchor_ee = np.zeros(3, dtype=np.float64)
        self.last_q_candidate = DEFAULT_JOINTS.copy()
        self.last_candidate_accepted = False
        self.last_reject_reason = "not_evaluated"
        self.last_coordinate_transform_mode = "default"
        self.real_anchor_valid = False
        self.real_safety_stop = False
        self.last_joint_state_time = None
        self.last_deadman_time = seconds_now(self) if self.deadman_enabled else None
        self.last_servo_time = None
        self.last_servo_p_command = None
        self.last_servo_cartesian_velocity = np.zeros(3, dtype=np.float64)
        self.last_servo_joint_velocity = np.zeros(5, dtype=np.float64)
        self.last_velocity_servo_palm_xyz = None
        self.velocity_servo_zero_sent_after_stop = False
        self.velocity_servo_diagnostic_logged = False
        self.last_servo_latency_ms = 0.0
        self.last_servo_reason = "not_started"
        self.hold_sent_after_stop = False
        self.workspace_check_frame = "fk_frame"
        self.last_workspace_margins = np.zeros(3, dtype=np.float64)
        self.last_adapter_applied = False
        self.last_buffer_publish_time = None
        self.last_buffer_q_target = None
        self.last_buffer_waypoints = []
        self.last_buffer_publish_reason = "not_started"
        self.forward_position_hold_sent_after_stop = False
        self.last_forward_position_publish_time = None
        self.last_forward_position_publish_reason = "not_started"
        self.forward_position_pending_publish_reason = "not_started"
        self.last_forward_position_data = []
        self.q_reference = None
        self.q_reference_velocity = np.zeros(5, dtype=np.float64)
        self.q_reference_valid = False
        self.reference_reset_reason = "not_initialized"
        self.reference_reset_applied = False
        self.tracking_error_rad = 0.0
        self.tracking_error_joint_diffs = np.zeros(5, dtype=np.float64)
        self.reference_dt_sec = 0.0
        self.actual_reference_tick_hz = 0.0
        self.last_reference_tick_time = None
        self.latest_q_target = None
        self.latest_p_desired = None
        self.latest_p_command = None
        self.buffer_target_available = False

        self.create_subscription(PoseStamped, self.palm_pose_topic, self.palm_cb, 50)
        self.create_subscription(
            PoseStamped, self.control_center_topic, self.control_center_cb, 10)
        self.create_subscription(Bool, "/amir/reset_palm_anchor", self.reset_anchor_cb, 10)
        self.create_subscription(Bool, self.deadman_topic, self.deadman_cb, 10)
        self.create_subscription(JointState, self.arm_state_topic, self.joint_state_cb, 10)

        self.target_pub = self.create_publisher(PointStamped, "/amir/target_xyz", 10)
        self.joint_cmd_pub = self.create_publisher(Float32MultiArray, "/amir/joint_cmd", 10)
        self.joint_cmd_deg_pub = self.create_publisher(Float32MultiArray, "/amir/joint_cmd_deg", 10)
        self.palm_delta_raw_pub = self.create_publisher(
            Float32MultiArray, "/amir/palm_delta_raw", 10)
        self.palm_delta_robot_pub = self.create_publisher(
            Vector3Stamped, "/amir/palm_delta_robot", 10)
        self.palm_delta_mr_pub = self.create_publisher(
            Vector3Stamped, "/amir/palm_delta_mr", 10)
        self.control_center_pub = self.create_publisher(
            PointStamped, "/amir/control_center_mr", 10)
        self.ee_center_pub = self.create_publisher(
            PointStamped, "/amir/ee_center_robot", 10)
        self.target_delta_pub = self.create_publisher(
            Float32MultiArray, "/amir/target_delta", 10)
        self.ik_debug_pub = self.create_publisher(Float32MultiArray, "/amir/ik_debug", 10)
        self.metrics_pub = self.create_publisher(Float32MultiArray, "/amir_metrics", 10)
        self.trajectory_pub = None
        if (self.mode in ("gazebo", "real") and
                not is_teleop_forward_position_profile(self.control_profile)):
            self.trajectory_pub = self.create_publisher(
                JointTrajectory, self.arm_command_topic, 10)
        self.forward_position_pub = None
        if is_teleop_forward_position_profile(self.control_profile):
            self.forward_position_pub = self.create_publisher(
                Float64MultiArray,
                self.forward_position_command_topic,
                10)

        self.reference_timer = None
        if (is_teleop_buffer_profile(self.control_profile) or
                is_teleop_forward_position_profile(self.control_profile)):
            self.reference_timer = self.create_timer(
                1.0 / self.reference_servo_rate_hz,
                self.reference_servo_tick)
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self.control_loop)
        self.get_logger().info(
            "AmirPalmIkNode started "
            f"mode={self.mode} command_unit={self.command_unit} "
            f"control_profile={self.control_profile} "
            f"dry_run={self.dry_run} "
            f"rate={self.control_rate_hz:.1f}Hz "
            f"servo_rate_hz={self.servo_rate_hz:.2f} "
            f"servo_horizon_sec={self.servo_horizon_sec:.3f} "
            f"reference_servo_rate_hz={self.reference_servo_rate_hz:.2f} "
            f"reference_tracking_error_threshold_rad="
            f"{self.reference_tracking_error_threshold_rad:.4f} "
            f"forward_position_command_topic="
            f"{self.forward_position_command_topic} "
            f"forward_position_publish_rate_hz="
            f"{self.forward_position_publish_rate_hz:.2f} "
            f"trajectory_buffer_update_rate_hz="
            f"{self.trajectory_buffer_update_rate_hz:.2f} "
            f"trajectory_buffer_horizon_sec="
            f"{self.trajectory_buffer_horizon_sec:.3f} "
            f"trajectory_buffer_point_dt_sec="
            f"{self.trajectory_buffer_point_dt_sec:.3f} "
            f"max_cartesian_velocity_mps={self.max_cartesian_velocity_mps:.4f} "
            f"max_joint_velocity_radps={self.max_joint_velocity_radps:.4f} "
            f"max_cartesian_acceleration_mps2="
            f"{self.max_cartesian_acceleration_mps2:.4f} "
            f"max_joint_acceleration_radps2="
            f"{self.max_joint_acceleration_radps2:.4f} "
            f"velocity_servo_hardware_confirmed="
            f"{self.velocity_servo_hardware_confirmed} "
            f"velocity_servo_time_from_start_sec="
            f"{self.velocity_servo_time_from_start_sec:.4f} "
            f"deadman_enabled={self.deadman_enabled} "
            f"deadman_topic={self.deadman_topic} "
            f"deadman_timeout_sec={self.deadman_timeout_sec:.3f} "
            f"arm_command_topic={self.arm_command_topic} "
            f"arm_state_topic={self.arm_state_topic} "
            f"gripper_command_topic={self.gripper_command_topic} "
            f"joint_names={self.joint_names} "
            f"trajectory_time_sec={self.trajectory_time_from_start_sec:.4f} "
            f"max_joint_step_rad={self.max_joint_step_rad:.5f} "
            f"kinematics_model={self.kinematics_model} "
            f"enable_kinematics_target_adapter={self.enable_kinematics_target_adapter} "
            f"kinematics_target_adapter_mode={self.kinematics_target_adapter_mode} "
            f"urdf_source_file={URDF_SOURCE_FILE} "
            f"fk_base_link={URDF_BASE_LINK} "
            f"fk_tip_link={URDF_TIP_LINK} "
            f"joint_axes={URDF_JOINT_AXES.tolist()} "
            f"joint_origins_xyz={URDF_JOINT_ORIGINS_XYZ.tolist()} "
            f"joint_origins_rpy={URDF_JOINT_ORIGINS_RPY.tolist()} "
            f"joint_limits_source=urdf "
            f"palm_pose_topic={self.palm_pose_topic} "
            f"control_center_topic={self.control_center_topic} "
            f"target_mapping_mode={self.target_mapping_mode} "
            f"absolute_axis_mapping={self.absolute_axis_mapping} "
            f"use_control_center={self.use_control_center} "
            f"require_control_center={self.require_control_center} "
            f"scale_x={self.scale_xyz[0]:.4f} "
            f"scale_y={self.scale_xyz[1]:.4f} "
            f"scale_z={self.scale_xyz[2]:.4f} "
            f"deadzone_m={self.deadzone_m:.4f} "
            f"max_delta_norm_m={self.max_delta_norm_m:.4f} "
            f"enable_reach_radius_limit={self.enable_reach_radius_limit} "
            f"reach_radius_max_m={self.reach_radius_max_m:.4f} "
            f"reach_radius_margin_m={self.reach_radius_margin_m:.4f} "
            f"effective_reach_radius={self.effective_reach_radius:.4f} "
            f"reach_radius_origin={np.round(self.reach_radius_origin, 4).tolist()} "
            f"lowpass_alpha={self.lowpass_alpha:.4f} "
            f"input_timeout_sec={self.input_timeout_sec:.4f} "
            f"axis_map={self.palm_to_robot_axis_map.tolist()} "
            f"axis_sign={self.palm_to_robot_axis_sign.tolist()} "
            f"axis_gain={self.palm_axis_gain.tolist()} "
            f"mr_range_x=[{self.mr_min[0]:.3f}, {self.mr_max[0]:.3f}] "
            f"mr_range_y=[{self.mr_min[1]:.3f}, {self.mr_max[1]:.3f}] "
            f"mr_range_z=[{self.mr_min[2]:.3f}, {self.mr_max[2]:.3f}] "
            f"abs_robot_range_x=[{self.abs_robot_min[0]:.3f}, "
            f"{self.abs_robot_max[0]:.3f}] "
            f"abs_robot_range_y=[{self.abs_robot_min[1]:.3f}, "
            f"{self.abs_robot_max[1]:.3f}] "
            f"abs_robot_range_z=[{self.abs_robot_min[2]:.3f}, "
            f"{self.abs_robot_max[2]:.3f}] "
            f"debug_target_step_x={self.debug_target_step_x:.4f} "
            f"palm_target_mode={self.palm_target_mode} "
            f"cylindrical_r_gain={self.cylindrical_r_gain:.4f} "
            f"cylindrical_z_gain={self.cylindrical_z_gain:.4f} "
            f"cylindrical_y_gain={self.cylindrical_y_gain:.4f} "
            f"cylindrical_r_offset={self.cylindrical_r_offset:.4f} "
            f"cylindrical_z_offset={self.cylindrical_z_offset:.4f} "
            f"cylindrical_y_offset={self.cylindrical_y_offset:.4f} "
            f"cylindrical_r_range=[{self.cylindrical_min_r:.4f}, "
            f"{self.cylindrical_max_r:.4f}] "
            f"cylindrical_z_range=[{self.cylindrical_min_z:.4f}, "
            f"{self.cylindrical_max_z:.4f}] "
            f"enable_range_scaled_palm={self.enable_range_scaled_palm} "
            f"human_motion_range={self.human_motion_range.tolist()} "
            f"robot_motion_range={self.robot_motion_range.tolist()} "
            f"range_scale_gain={self.range_scale_gain:.4f} "
            f"range_scale_max={self.range_scale_max.tolist()} "
            f"enable_extension_bias={self.enable_extension_bias} "
            f"extension_bias_gain={self.extension_bias_gain:.4f} "
            f"extension_trigger_x={self.extension_trigger_x:.4f} "
            f"extension_trigger_norm={self.extension_trigger_norm:.4f} "
            f"extension_use_motion_norm={self.extension_use_motion_norm} "
            f"extension_posture={np.round(self.extension_posture, 4).tolist()} "
            f"joint_limit_avoidance_gain={self.joint_limit_avoidance_gain:.4f} "
            f"enable_reach_approach={self.enable_reach_approach} "
            f"reach_trigger_error={self.reach_trigger_error:.4f} "
            f"reach_full_error={self.reach_full_error:.4f} "
            f"reach_bias_gain={self.reach_bias_gain:.4f} "
            f"reach_posture={np.round(self.reach_posture, 4).tolist()} "
            f"reach_directional_yaw={self.reach_directional_yaw} "
            f"reach_yaw_trigger_norm={self.reach_yaw_trigger_norm:.4f}")

    def palm_cb(self, msg):
        self.palm_xyz = np.array([
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ], dtype=np.float64)
        self.last_palm_time = seconds_now(self)

        if self.mode == "real" and not self.joint_state_received:
            self.get_logger().warn(
                "palm_pose received before complete real joint state; "
                "anchor setup is deferred.",
                throttle_duration_sec=1.0)
            return

        if self.is_real_relative_mode():
            if not self.has_anchor:
                self.set_real_relative_anchor()
            return

        if self.target_mapping_mode == "absolute":
            self.last_palm_now_mr = self.palm_xyz.copy()
        elif self.use_control_center:
            if not self.has_anchor and (self.has_control_center or not self.require_control_center):
                self.set_ee_anchor_from_current_pose()
        elif self.palm_anchor_xyz is None:
            self.set_legacy_palm_anchor()

    def fk_xyz(self, joints):
        if self.kinematics_model == "urdf_like":
            return fk_xyz_urdf_like(joints)
        return fk_xyz(joints)

    def adapt_target_for_kinematics(self, target_control):
        if self.is_real_relative_mode():
            self.last_coordinate_transform_mode = "real_relative_fk_frame_no_adapter"
            return target_control.copy()
        if (self.kinematics_model != "urdf_like" or
                not self.enable_kinematics_target_adapter or
                self.kinematics_target_adapter_mode == "none"):
            self.last_coordinate_transform_mode = "none"
            return target_control.copy()
        if self.kinematics_target_adapter_mode == "control_to_urdf_y_forward":
            self.last_coordinate_transform_mode = "control_to_urdf_y_forward"
            return np.array([
                -target_control[1],
                target_control[0],
                target_control[2],
            ], dtype=np.float64)
        self.last_coordinate_transform_mode = self.kinematics_target_adapter_mode
        return target_control.copy()

    def control_center_cb(self, msg):
        self.control_center_mr = np.array([
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ], dtype=np.float64)
        first_center = not self.has_control_center
        self.has_control_center = True
        if first_center:
            self.get_logger().info(
                "control center received "
                f"control_center_mr={np.round(self.control_center_mr, 4).tolist()}")

    def deadman_cb(self, msg):
        self.deadman_enabled = bool(msg.data)
        if self.deadman_enabled:
            self.last_deadman_time = seconds_now(self)
        else:
            self.last_deadman_time = None

    def is_deadman_active(self, now):
        if not self.deadman_enabled:
            return False
        if self.deadman_timeout_sec <= 0.0:
            return True
        if self.last_deadman_time is None:
            return False
        return (now - self.last_deadman_time) <= self.deadman_timeout_sec

    def is_real_relative_mode(self):
        return self.mode == "real" and self.target_mapping_mode == "relative"

    def reset_reference_from_measured(self, q_measured, reason):
        self.q_reference = np.array(q_measured, dtype=np.float64).copy()
        self.q_reference_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.q_reference_valid = True
        self.reference_reset_reason = str(reason)
        self.reference_reset_applied = True

    def invalidate_reference(self, reason):
        self.q_reference = None
        self.q_reference_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.q_reference_valid = False
        self.reference_reset_reason = str(reason)
        self.reference_reset_applied = True
        self.tracking_error_rad = 0.0
        self.tracking_error_joint_diffs = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.last_buffer_publish_time = None
        self.last_buffer_q_target = None
        self.last_buffer_waypoints = []
        self.last_buffer_publish_reason = str(reason)
        self.last_forward_position_publish_time = None
        self.forward_position_pending_publish_reason = str(reason)
        self.latest_q_target = None
        self.latest_p_desired = None
        self.latest_p_command = None
        self.buffer_target_available = False

    def update_reference_tracking_state(self, q_measured):
        if not self.q_reference_valid or self.q_reference is None:
            self.tracking_error_rad = 0.0
            self.tracking_error_joint_diffs = np.zeros(
                len(JOINT_NAMES), dtype=np.float64)
            self.reset_reference_from_measured(
                q_measured, "reference_reinitialized")
            return
        self.tracking_error_joint_diffs = (
            np.array(q_measured, dtype=np.float64) -
            np.array(self.q_reference, dtype=np.float64)
        )
        self.tracking_error_rad = reference_tracking_error(
            self.q_reference, q_measured)
        if should_reset_reference_for_tracking_error(
                self.q_reference,
                q_measured,
                self.reference_tracking_error_threshold_rad):
            self.reset_reference_from_measured(
                q_measured, "tracking_error_exceeded")
        else:
            self.reference_reset_reason = "none"
            self.reference_reset_applied = False

    def step_reference_toward_target(
            self, q_target, q_measured, dt, hold_on_tracking_reset=False):
        self.update_reference_tracking_state(q_measured)
        if (hold_on_tracking_reset and self.reference_reset_applied and
                self.reference_reset_reason == "tracking_error_exceeded"):
            self.q_reference_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
            self.last_servo_joint_velocity = self.q_reference_velocity.copy()
            return True
        step = integrate_joint_reference(
            self.q_reference,
            q_target,
            self.q_reference_velocity,
            dt,
            self.max_joint_velocity_radps,
            self.max_joint_acceleration_radps2)
        self.q_reference = clamp_array(
            np.array(step["position"], dtype=np.float64),
            JOINT_LIMITS[:, 0],
            JOINT_LIMITS[:, 1])
        self.q_reference_velocity = np.array(
            step["velocity"], dtype=np.float64)
        self.last_servo_joint_velocity = self.q_reference_velocity.copy()
        if not self.reference_reset_applied:
            self.tracking_error_joint_diffs = (
                np.array(q_measured, dtype=np.float64) -
                np.array(self.q_reference, dtype=np.float64)
            )
            self.tracking_error_rad = reference_tracking_error(
                self.q_reference, q_measured)
        return False

    def reset_anchor_cb(self, msg):
        if not bool(msg.data):
            return
        current_ee = self.fk_xyz(self.current_joints)
        self.has_anchor = False
        self.palm_anchor_xyz = None
        self.ee_anchor_xyz = None
        self.anchor_q = None
        self.real_anchor_valid = False
        self.real_safety_stop = False
        self.anchor_error_norm = float("inf")
        self.last_servo_time = None
        self.last_servo_p_command = None
        self.last_servo_cartesian_velocity = np.zeros(3, dtype=np.float64)
        self.last_servo_joint_velocity = np.zeros(5, dtype=np.float64)
        self.last_velocity_servo_palm_xyz = None
        self.velocity_servo_zero_sent_after_stop = False
        self.hold_sent_after_stop = False
        self.forward_position_hold_sent_after_stop = False
        self.invalidate_reference("anchor_reset")
        self.ee_center_robot = current_ee.copy()
        self.last_target_xyz = current_ee.copy()
        self.filtered_target_xyz = current_ee.copy()
        self.get_logger().warn(
            "palm anchor reset "
            f"current_ee_xyz={np.round(current_ee, 4).tolist()}")

    def set_real_relative_anchor(self):
        self.anchor_q = self.current_joints.copy()
        self.palm_anchor_xyz = self.palm_xyz.copy()
        self.ee_anchor_xyz = self.fk_xyz(self.anchor_q)
        self.ee_center_robot = self.ee_anchor_xyz.copy()
        self.last_target_xyz = self.ee_anchor_xyz.copy()
        self.filtered_target_xyz = self.ee_anchor_xyz.copy()
        self.last_target_raw_xyz = self.ee_anchor_xyz.copy()
        self.last_target_before_safety_xyz = self.ee_anchor_xyz.copy()
        self.last_target_control_xyz = self.ee_anchor_xyz.copy()
        self.last_target_kinematics_xyz = self.ee_anchor_xyz.copy()
        self.last_target_delta = np.zeros(3, dtype=np.float64)
        self.last_target_minus_anchor_ee = np.zeros(3, dtype=np.float64)
        self.last_palm_delta_raw = np.zeros(3, dtype=np.float64)
        self.last_palm_delta_robot = np.zeros(3, dtype=np.float64)
        self.last_scaled_delta = np.zeros(3, dtype=np.float64)
        self.anchor_error_norm = 0.0
        self.real_anchor_valid = True
        self.real_safety_stop = False
        self.last_candidate_accepted = True
        self.last_reject_reason = "anchor_initialized"
        self.has_anchor = True
        self.skip_next_command_after_anchor = True
        self.last_servo_time = None
        self.last_servo_p_command = self.ee_anchor_xyz.copy()
        self.last_servo_cartesian_velocity = np.zeros(3, dtype=np.float64)
        self.last_servo_joint_velocity = np.zeros(5, dtype=np.float64)
        self.last_velocity_servo_palm_xyz = self.palm_anchor_xyz.copy()
        self.velocity_servo_zero_sent_after_stop = False
        self.hold_sent_after_stop = False
        self.forward_position_hold_sent_after_stop = False
        self.reset_reference_from_measured(
            self.anchor_q, "anchor_initialized")
        self.last_buffer_publish_time = None
        self.last_buffer_q_target = None
        self.last_buffer_waypoints = []
        self.last_buffer_publish_reason = "anchor_initialized"
        self.last_forward_position_publish_time = None
        self.forward_position_pending_publish_reason = "anchor_initialized"
        self.get_logger().info(
            "Real relative palm anchor set "
            f"anchor_q={np.round(self.anchor_q, 5).tolist()} "
            f"anchor_ee={np.round(self.ee_anchor_xyz, 5).tolist()} "
            f"palm_anchor={np.round(self.palm_anchor_xyz, 5).tolist()} "
            "target_control=anchor_ee")

    def set_ee_anchor_from_current_pose(self):
        self.ee_center_robot = self.fk_xyz(self.current_joints)
        self.ee_anchor_xyz = self.ee_center_robot.copy()
        self.last_target_xyz = self.ee_center_robot.copy()
        self.filtered_target_xyz = self.ee_center_robot.copy()
        self.has_anchor = True
        if self.mode == "real":
            self.skip_next_command_after_anchor = True
        palm_delta_mr = (
            self.palm_xyz - self.control_center_mr
            if self.has_control_center else
            np.zeros(3, dtype=np.float64)
        )
        self.get_logger().info(
            "EE anchor set from first valid palm_pose "
            f"ee_center_robot={np.round(self.ee_center_robot, 4).tolist()} "
            f"palm_now_mr={np.round(self.palm_xyz, 4).tolist()} "
            f"control_center_mr={np.round(self.control_center_mr, 4).tolist()} "
            f"palm_delta_mr={np.round(palm_delta_mr, 4).tolist()}")

    def set_legacy_palm_anchor(self):
        self.palm_anchor_xyz = self.palm_xyz.copy()
        self.ee_anchor_xyz = self.fk_xyz(self.current_joints)
        self.ee_center_robot = self.ee_anchor_xyz.copy()
        self.has_anchor = True
        if self.mode == "real":
            self.skip_next_command_after_anchor = True
        if self.palm_target_mode == "relative":
            self.last_target_xyz = clamp_array(
                self.ee_anchor_xyz.copy(), self.workspace_min, self.workspace_max)
            self.filtered_target_xyz = self.last_target_xyz.copy()
        else:
            self.last_target_xyz = None
            self.filtered_target_xyz = None
        self.get_logger().info(
            "Legacy palm anchor set "
            f"palm={self.palm_anchor_xyz.tolist()} ee={self.ee_anchor_xyz.tolist()} "
            f"palm_target_mode={self.palm_target_mode} "
            f"axis_map={self.palm_to_robot_axis_map.tolist()} "
            f"axis_sign={self.palm_to_robot_axis_sign.tolist()} "
            f"palm_motion_gain={self.palm_motion_gain:.4f} "
            f"max_target_step_m={self.max_target_step_m:.4f} "
            f"trajectory_time_from_start_sec={self.trajectory_time_from_start_sec:.4f}")

    def joint_state_cb(self, msg):
        try:
            updated = np.array(
                reorder_positions_from_joint_state(
                    msg.name, msg.position, self.joint_names),
                dtype=np.float64)
        except ValueError as exc:
            self.get_logger().error(
                f"Invalid {self.arm_state_topic} JointState: {exc}",
                throttle_duration_sec=1.0)
            if self.mode == "real":
                self.joint_state_received = False
            return
        self.current_joints = clamp_array(updated, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        self.joint_state_received = True
        self.last_joint_state_time = seconds_now(self)
        if not self.has_anchor:
            self.last_command_joints = self.current_joints.copy()

    def make_axis_transform_matrix(self):
        matrix = np.zeros((3, 3), dtype=np.float64)
        for robot_axis, mr_axis in enumerate(self.palm_to_robot_axis_map):
            matrix[robot_axis, int(mr_axis)] = self.palm_to_robot_axis_sign[robot_axis]
        return matrix

    def convert_palm_delta_to_robot_delta(self, palm_delta_raw):
        mapped = palm_delta_raw[self.palm_to_robot_axis_map]
        return self.palm_to_robot_axis_sign * self.palm_axis_gain * mapped

    def convert_palm_xyz_to_robot_abs(self, palm_xyz):
        mapped = palm_xyz[self.palm_to_robot_axis_map]
        return self.palm_to_robot_axis_sign * mapped

    def palm_delta_to_target_delta(self, palm_delta_robot):
        if not self.enable_range_scaled_palm:
            self.last_range_scale = np.ones(3, dtype=np.float64)
            return self.palm_motion_gain * palm_delta_robot

        scale = self.robot_motion_range / self.human_motion_range
        scale = np.clip(scale, 0.0, self.range_scale_max)
        self.last_range_scale = scale
        return self.range_scale_gain * scale * palm_delta_robot

    def limit_target_step(self, clamped_target, clamped_flag):
        if self.last_target_xyz is None:
            limited_target = clamped_target
        else:
            delta = clamped_target - self.last_target_xyz
            delta_norm = float(np.linalg.norm(delta))
            if self.max_target_step_m > 0.0 and delta_norm > self.max_target_step_m:
                delta = delta * (self.max_target_step_m / delta_norm)
                clamped_flag = 1.0
            limited_target = self.last_target_xyz + delta

        limited_target = clamp_array(limited_target, self.workspace_min, self.workspace_max)
        self.last_target_xyz = limited_target
        self.last_clamped_flag = clamped_flag
        return limited_target

    def make_target(self):
        if self.is_real_relative_mode():
            return self.make_real_relative_target()
        if self.target_mapping_mode == "absolute":
            return self.make_absolute_target()
        if self.use_control_center:
            return self.make_control_center_target()
        if self.palm_target_mode == "cylindrical_absolute":
            return self.make_cylindrical_absolute_target()
        return self.make_relative_target()

    def make_real_relative_target(self):
        if self.palm_anchor_xyz is None or self.ee_anchor_xyz is None:
            self.get_logger().error(
                "real relative target requested before anchor is available")
            self.real_safety_stop = True
            self.last_reject_reason = "missing_real_relative_anchor"
            return self.fk_xyz(self.current_joints)

        target_data = compute_relative_target(
            self.ee_anchor_xyz,
            self.palm_anchor_xyz,
            self.palm_xyz,
            self.palm_to_robot_axis_map,
            self.palm_to_robot_axis_sign,
            self.palm_axis_gain,
            self.scale_xyz,
            self.deadzone_m,
        )
        delta_hand = np.array(target_data["delta_hand"], dtype=np.float64)
        delta_robot = np.array(target_data["delta_robot"], dtype=np.float64)
        scaled_delta = np.array(target_data["scaled_delta"], dtype=np.float64)
        target_control = np.array(target_data["target_control"], dtype=np.float64)
        target_minus_anchor = np.array(
            target_data["target_minus_anchor_ee"], dtype=np.float64)

        self.last_palm_delta_raw = delta_hand.copy()
        self.last_palm_delta_robot = delta_robot.copy()
        self.last_palm_now_mr = self.palm_xyz.copy()
        self.last_scaled_delta = scaled_delta.copy()
        self.last_target_raw_xyz = target_control.copy()
        self.last_target_before_safety_xyz = target_control.copy()
        self.last_target_delta = target_minus_anchor.copy()
        self.last_target_minus_anchor_ee = target_minus_anchor.copy()
        self.last_motion_norm = float(np.linalg.norm(delta_hand))
        self.last_target_xyz = target_control.copy()
        self.filtered_target_xyz = target_control.copy()
        self.last_clamped_flag = 0.0

        if float(np.linalg.norm(delta_hand)) <= 1e-9:
            self.anchor_error_norm = float(np.linalg.norm(target_minus_anchor))
            if self.anchor_error_norm > 0.005:
                self.real_safety_stop = True
                self.real_anchor_valid = False
                self.last_reject_reason = "anchor_target_mismatch"
                self.get_logger().error(
                    "FATAL real relative anchor mismatch: "
                    f"anchor_error_norm={self.anchor_error_norm:.6f}m "
                    f"anchor_ee={np.round(self.ee_anchor_xyz, 5).tolist()} "
                    f"target_control={np.round(target_control, 5).tolist()}")
            else:
                self.real_anchor_valid = True

        return target_control

    def apply_target_safety(self, target_raw, clamped_flag=0.0):
        target_raw_before_safety = target_raw.copy()
        target_safety, reach_radius_clamped = self.apply_reach_radius_limit(target_raw)
        clamped_flag = max(clamped_flag, reach_radius_clamped)

        workspace_limited = clamp_array(target_safety, self.workspace_min, self.workspace_max)
        if not np.allclose(target_safety, workspace_limited):
            clamped_flag = 1.0
            self.get_logger().warn(
                "target clamp applied "
                f"target_raw={np.round(target_safety, 4).tolist()} "
                f"workspace_limited={np.round(workspace_limited, 4).tolist()}",
                throttle_duration_sec=1.0)

        target_limited = self.limit_target_step(workspace_limited, clamped_flag)
        if self.filtered_target_xyz is None:
            self.filtered_target_xyz = target_limited.copy()
        else:
            self.filtered_target_xyz = (
                self.lowpass_alpha * target_limited +
                (1.0 - self.lowpass_alpha) * self.filtered_target_xyz
            )

        self.last_target_raw_xyz = target_raw_before_safety.copy()
        self.last_target_before_safety_xyz = target_raw_before_safety.copy()
        self.last_clamped_flag = max(self.last_clamped_flag, clamped_flag)
        return self.filtered_target_xyz.copy()

    def make_absolute_target(self):
        palm_now_mr = self.palm_xyz.copy()
        target_raw = np.array([
            map_range_clamped(
                palm_now_mr[2],
                self.mr_min[2],
                self.mr_max[2],
                self.abs_robot_min[0],
                self.abs_robot_max[0]),
            map_range_clamped(
                palm_now_mr[0],
                self.mr_min[0],
                self.mr_max[0],
                self.abs_robot_max[1],
                self.abs_robot_min[1]),
            map_range_clamped(
                palm_now_mr[1],
                self.mr_min[1],
                self.mr_max[1],
                self.abs_robot_min[2],
                self.abs_robot_max[2]),
        ], dtype=np.float64)

        filtered_target = self.apply_target_safety(target_raw)
        self.last_palm_now_mr = palm_now_mr.copy()
        self.last_palm_delta_raw = np.zeros(3, dtype=np.float64)
        self.last_palm_delta_robot = np.zeros(3, dtype=np.float64)
        self.last_scaled_delta = np.zeros(3, dtype=np.float64)
        self.last_target_delta = filtered_target - self.fk_xyz(self.current_joints)
        self.last_motion_norm = 0.0
        return filtered_target

    def apply_reach_radius_limit(self, target_xyz):
        self.last_reach_radius_clamped_flag = 0.0
        self.last_effective_reach_radius = self.effective_reach_radius
        p = target_xyz - self.reach_radius_origin
        radius = float(np.linalg.norm(p))
        self.last_reach_radius = radius

        if (not self.enable_reach_radius_limit or
                self.effective_reach_radius <= 0.0 or
                radius <= self.effective_reach_radius):
            return target_xyz.copy(), 0.0

        limited = self.reach_radius_origin + p * (self.effective_reach_radius / radius)
        self.last_reach_radius_clamped_flag = 1.0
        self.get_logger().warn(
            "reach radius clamp applied "
            f"r={radius:.4f} effective_radius={self.effective_reach_radius:.4f}",
            throttle_duration_sec=1.0)
        return limited, 1.0

    def make_control_center_target(self):
        palm_delta_mr = self.palm_xyz - self.control_center_mr
        palm_delta_norm = float(np.linalg.norm(palm_delta_mr))
        if palm_delta_norm < self.deadzone_m:
            palm_delta_mr = np.zeros(3, dtype=np.float64)
            palm_delta_norm = 0.0

        palm_delta_robot = self.r_robot_from_mr @ palm_delta_mr
        robot_delta_norm = float(np.linalg.norm(palm_delta_robot))
        if abs(robot_delta_norm - palm_delta_norm) > 1e-6:
            self.get_logger().warn(
                "palm_delta norm mismatch after R_robot_from_mr "
                f"palm_delta_norm={palm_delta_norm:.6f} "
                f"robot_delta_norm={robot_delta_norm:.6f}",
                throttle_duration_sec=1.0)
        scaled_delta = self.scale_xyz * palm_delta_robot
        target_raw = self.ee_center_robot + self.palm_motion_gain * scaled_delta
        target_raw_unclamped = target_raw.copy()

        clamped_flag = 0.0
        center_delta = target_raw - self.ee_center_robot
        center_delta_norm = float(np.linalg.norm(center_delta))
        if self.max_delta_norm_m > 0.0 and center_delta_norm > self.max_delta_norm_m:
            target_raw = self.ee_center_robot + center_delta * (
                self.max_delta_norm_m / center_delta_norm)
            clamped_flag = 1.0

        filtered_target = self.apply_target_safety(target_raw, clamped_flag)

        self.last_palm_delta_raw = palm_delta_mr.copy()
        self.last_palm_delta_robot = palm_delta_robot.copy()
        self.last_palm_now_mr = self.palm_xyz.copy()
        self.last_scaled_delta = scaled_delta.copy()
        self.last_target_raw_xyz = target_raw_unclamped.copy()
        self.last_target_before_safety_xyz = target_raw_unclamped.copy()
        self.last_target_delta = filtered_target - self.ee_center_robot
        self.last_motion_norm = palm_delta_norm
        return filtered_target

    def make_relative_target(self):
        palm_delta_raw = self.palm_xyz - self.palm_anchor_xyz
        palm_delta_robot = self.convert_palm_delta_to_robot_delta(palm_delta_raw)
        target_delta = self.palm_delta_to_target_delta(palm_delta_robot)
        raw_target = self.ee_anchor_xyz + target_delta
        raw_target[0] += self.debug_target_step_x

        self.last_palm_delta_raw = palm_delta_raw
        self.last_palm_delta_robot = palm_delta_robot

        clamped_target = clamp_array(raw_target, self.workspace_min, self.workspace_max)
        clamped_flag = 1.0 if not np.allclose(raw_target, clamped_target) else 0.0

        limited_target = self.limit_target_step(clamped_target, clamped_flag)
        self.last_target_delta = limited_target - self.ee_anchor_xyz
        return limited_target

    def make_cylindrical_absolute_target(self):
        if self.enable_range_scaled_palm and not self.range_scaled_ignored_warned:
            self.get_logger().warn(
                "enable_range_scaled_palm is ignored when "
                "palm_target_mode=cylindrical_absolute.")
            self.range_scaled_ignored_warned = True
        self.last_range_scale = np.ones(3, dtype=np.float64)

        if self.palm_anchor_xyz is not None:
            palm_delta_raw = self.palm_xyz - self.palm_anchor_xyz
            palm_delta_robot = self.convert_palm_delta_to_robot_delta(palm_delta_raw)
        else:
            palm_delta_raw = np.zeros(3, dtype=np.float64)
            palm_delta_robot = np.zeros(3, dtype=np.float64)
        self.last_palm_delta_raw = palm_delta_raw
        self.last_palm_delta_robot = palm_delta_robot

        palm_abs_robot = self.convert_palm_xyz_to_robot_abs(self.palm_xyz)
        r_palm = float(math.hypot(float(palm_abs_robot[0]), float(palm_abs_robot[1])))
        y_palm = float(palm_abs_robot[1])
        z_palm = float(palm_abs_robot[2])

        r_target = float(np.clip(
            self.cylindrical_r_gain * r_palm + self.cylindrical_r_offset,
            self.cylindrical_min_r,
            self.cylindrical_max_r))
        z_target = float(np.clip(
            self.cylindrical_z_gain * z_palm + self.cylindrical_z_offset,
            self.cylindrical_min_z,
            self.cylindrical_max_z))
        y_target = self.cylindrical_y_gain * y_palm + self.cylindrical_y_offset

        raw_target = np.array([r_target, y_target, z_target], dtype=np.float64)
        raw_target[0] += self.debug_target_step_x

        self.last_palm_abs_robot = palm_abs_robot.copy()
        self.last_r_palm = r_palm
        self.last_y_palm = y_palm
        self.last_z_palm = z_palm
        self.last_r_target = r_target
        self.last_y_target = float(y_target)
        self.last_z_target = z_target

        clamped_target = clamp_array(raw_target, self.workspace_min, self.workspace_max)
        clamped_flag = 1.0 if not np.allclose(raw_target, clamped_target) else 0.0
        limited_target = self.limit_target_step(clamped_target, clamped_flag)
        if self.ee_anchor_xyz is not None:
            self.last_target_delta = limited_target - self.ee_anchor_xyz
        else:
            self.last_target_delta = np.zeros(3, dtype=np.float64)
        return limited_target

    def make_debug_target_from_joint_state(self):
        self.ee_anchor_xyz = self.fk_xyz(self.current_joints)
        self.last_palm_delta_raw = np.zeros(3, dtype=np.float64)
        self.last_palm_delta_robot = np.zeros(3, dtype=np.float64)
        self.last_palm_abs_robot = np.zeros(3, dtype=np.float64)
        self.last_r_palm = 0.0
        self.last_y_palm = 0.0
        self.last_z_palm = 0.0
        self.last_r_target = 0.0
        self.last_y_target = 0.0
        self.last_z_target = 0.0
        self.last_range_scale = (
            np.clip(
                self.robot_motion_range / self.human_motion_range,
                0.0,
                self.range_scale_max)
            if self.enable_range_scaled_palm else
            np.ones(3, dtype=np.float64)
        )

        raw_target = self.ee_anchor_xyz + np.array(
            [self.debug_target_step_x, 0.0, 0.0], dtype=np.float64)
        target_xyz = clamp_array(raw_target, self.workspace_min, self.workspace_max)

        self.last_target_xyz = target_xyz.copy()
        self.last_target_delta = target_xyz - self.ee_anchor_xyz
        self.last_clamped_flag = 1.0 if not np.allclose(raw_target, target_xyz) else 0.0
        return target_xyz

    def publish_float_array(self, publisher, values):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        publisher.publish(msg)

    def make_point_stamped(self, xyz, frame_id="amir_base"):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        return msg

    def make_vector3_stamped(self, xyz, frame_id="amir_base"):
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.vector.x = float(xyz[0])
        msg.vector.y = float(xyz[1])
        msg.vector.z = float(xyz[2])
        return msg

    def publish_joint_outputs(self, joints_rad):
        if self.command_unit == "mrad":
            joint_cmd = joints_rad * 1000.0
        else:
            joint_cmd = joints_rad
        self.publish_float_array(self.joint_cmd_pub, joint_cmd)
        self.publish_float_array(self.joint_cmd_deg_pub, np.degrees(joints_rad))

    def publish_debug_deltas(self):
        if self.use_control_center:
            self.last_palm_delta_robot = self.r_robot_from_mr @ self.last_palm_delta_raw
        self.publish_float_array(self.palm_delta_raw_pub, self.last_palm_delta_raw)
        self.palm_delta_mr_pub.publish(self.make_vector3_stamped(self.last_palm_delta_raw))
        self.palm_delta_robot_pub.publish(
            self.make_vector3_stamped(self.last_palm_delta_robot))
        self.control_center_pub.publish(self.make_point_stamped(self.control_center_mr))
        self.ee_center_pub.publish(self.make_point_stamped(self.ee_center_robot))
        self.publish_float_array(self.target_delta_pub, self.last_target_delta)

    def extension_weight(self):
        if not self.enable_extension_bias:
            self.last_motion_norm = float(np.linalg.norm(self.last_target_delta))
            return 0.0

        if self.extension_use_motion_norm:
            motion = float(np.linalg.norm(self.last_target_delta))
            trigger = self.extension_trigger_norm
        else:
            motion = float(self.last_target_delta[0])
            trigger = self.extension_trigger_x

        self.last_motion_norm = float(np.linalg.norm(self.last_target_delta))
        if motion <= trigger:
            return 0.0
        if trigger <= 1e-6:
            return 1.0
        return float(np.clip((motion - trigger) / trigger, 0.0, 1.0))

    def extension_bias_delta(self, joints_rad):
        weight = self.extension_weight()
        self.last_extension_weight = weight
        if weight <= 0.0 or self.extension_bias_gain <= 0.0:
            return np.zeros(5, dtype=np.float64)

        bias_mask = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        return (
            self.extension_bias_gain * weight * bias_mask *
            (self.extension_posture - joints_rad)
        )

    def reach_weight(self, reach_error):
        if not self.enable_reach_approach:
            return 0.0
        if reach_error <= self.reach_trigger_error:
            return 0.0
        denom = max(self.reach_full_error - self.reach_trigger_error, 1e-6)
        return float(np.clip((reach_error - self.reach_trigger_error) / denom, 0.0, 1.0))

    def reach_approach_delta(self, joints_rad, q_current, reach_vec, reach_error):
        weight = self.reach_weight(reach_error)
        self.last_reach_weight = weight
        self.last_reach_error = float(reach_error)
        self.last_reach_vec = reach_vec.copy()
        self.last_q_reach = self.reach_posture.copy()
        self.last_desired_reach_yaw = float(q_current[0])

        if weight <= 0.0 or self.reach_bias_gain <= 0.0:
            return np.zeros(5, dtype=np.float64)

        q_reach = self.reach_posture.copy()
        bias_mask = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
        reach_xy_norm = float(np.linalg.norm(reach_vec[:2]))
        if self.reach_directional_yaw and reach_xy_norm > self.reach_yaw_trigger_norm:
            desired_yaw = math.atan2(float(reach_vec[1]), float(reach_vec[0]))
            q_reach[0] = float(np.clip(
                desired_yaw, JOINT_LIMITS[0, 0], JOINT_LIMITS[0, 1]))
            bias_mask[0] = 1.0
            self.last_desired_reach_yaw = q_reach[0]
        else:
            q_reach[0] = float(q_current[0])
            self.last_desired_reach_yaw = float(q_current[0])

        self.last_q_reach = q_reach.copy()
        return self.reach_bias_gain * weight * bias_mask * (q_reach - joints_rad)

    def joint_limit_avoidance_delta(self, joints_rad):
        if self.joint_limit_avoidance_gain <= 0.0:
            self.last_limit_hit = 0.0
            return np.zeros(5, dtype=np.float64)

        lower = JOINT_LIMITS[:, 0]
        upper = JOINT_LIMITS[:, 1]
        center = 0.5 * (lower + upper)
        span = np.maximum(upper - lower, 1e-6)
        margin_ratio = np.minimum(joints_rad - lower, upper - joints_rad) / span
        threshold = 0.20
        proximity = np.clip((threshold - margin_ratio) / threshold, 0.0, 1.0)
        self.last_limit_hit = 1.0 if np.any(proximity > 0.0) else 0.0
        return self.joint_limit_avoidance_gain * proximity * (center - joints_rad)

    def apply_posture_biases(self, joints_rad, target_xyz, base_error, q_current, fk_before):
        reach_vec = target_xyz - fk_before
        reach_error = float(np.linalg.norm(reach_vec))
        reach_delta = self.reach_approach_delta(
            joints_rad, q_current, reach_vec, reach_error)
        if float(np.linalg.norm(reach_delta)) > 1e-12:
            # Reach approach owns the posture preference for this cycle.
            self.last_extension_weight = 0.0
            posture_delta = reach_delta
        else:
            posture_delta = self.extension_bias_delta(joints_rad)
            self.last_reach_weight = 0.0
            self.last_reach_error = reach_error
            self.last_reach_vec = reach_vec.copy()
            self.last_q_reach = self.reach_posture.copy()
            self.last_desired_reach_yaw = float(q_current[0])

        bias_delta = posture_delta + self.joint_limit_avoidance_delta(joints_rad)
        if float(np.linalg.norm(bias_delta)) <= 1e-12:
            return joints_rad

        # Keep the posture preference subordinate to the hand-position task.
        max_allowed_error = base_error + max(0.005, 0.5 * base_error)
        for scale in (1.0, 0.5, 0.25, 0.1):
            candidate = clamp_array(
                joints_rad + scale * bias_delta,
                JOINT_LIMITS[:, 0],
                JOINT_LIMITS[:, 1])
            candidate_error = float(np.linalg.norm(target_xyz - self.fk_xyz(candidate)))
            if candidate_error <= max_allowed_error:
                self.last_extension_weight *= scale
                self.last_reach_weight *= scale
                return candidate

        self.last_extension_weight = 0.0
        self.last_reach_weight = 0.0
        return joints_rad

    def publish_ik_debug(self):
        self.publish_float_array(self.ik_debug_pub, [
            self.last_error_before,
            self.last_error_norm,
            self.last_dq[0],
            self.last_dq[1],
            self.last_dq[2],
            self.last_dq[3],
            self.last_dq[4],
            self.last_target_delta[0],
            self.last_target_delta[1],
            self.last_target_delta[2],
            self.last_fk_before[0],
            self.last_fk_before[1],
            self.last_fk_before[2],
            self.last_fk_after[0],
            self.last_fk_after[1],
            self.last_fk_after[2],
            self.last_q_current[0],
            self.last_q_current[1],
            self.last_q_current[2],
            self.last_q_current[3],
            self.last_q_current[4],
            self.last_q_solution[0],
            self.last_q_solution[1],
            self.last_q_solution[2],
            self.last_q_solution[3],
            self.last_q_solution[4],
            self.last_dq[0],
            self.last_dq[1],
            self.last_dq[2],
            self.last_dq[3],
            self.last_dq[4],
            self.last_limit_hit,
            self.last_extension_weight,
            self.last_error_before,
            self.last_error_norm,
            1.0 if self.extension_use_motion_norm else 0.0,
            self.extension_trigger_norm,
            self.last_motion_norm,
            self.last_extension_weight,
            1.0 if self.enable_reach_approach else 0.0,
            self.last_reach_error,
            self.last_reach_weight,
            self.last_reach_vec[0],
            self.last_reach_vec[1],
            self.last_reach_vec[2],
            self.last_q_reach[0],
            self.last_q_reach[1],
            self.last_q_reach[2],
            self.last_q_reach[3],
            self.last_q_reach[4],
            self.reach_bias_gain,
            1.0 if self.reach_directional_yaw else 0.0,
            self.last_desired_reach_yaw,
            1.0 if self.enable_range_scaled_palm else 0.0,
            self.last_range_scale[0],
            self.last_range_scale[1],
            self.last_range_scale[2],
            1.0 if self.palm_target_mode == "cylindrical_absolute" else 0.0,
            self.last_palm_abs_robot[0],
            self.last_palm_abs_robot[1],
            self.last_palm_abs_robot[2],
            self.last_r_palm,
            self.last_y_palm,
            self.last_z_palm,
            self.last_r_target,
            self.last_y_target,
            self.last_z_target,
            self.last_target_xyz[0] if self.last_target_xyz is not None else 0.0,
            self.last_target_xyz[1] if self.last_target_xyz is not None else 0.0,
            self.last_target_xyz[2] if self.last_target_xyz is not None else 0.0,
            self.last_palm_now_mr[0],
            self.last_palm_now_mr[1],
            self.last_palm_now_mr[2],
            self.control_center_mr[0],
            self.control_center_mr[1],
            self.control_center_mr[2],
            self.last_palm_delta_raw[0],
            self.last_palm_delta_raw[1],
            self.last_palm_delta_raw[2],
            self.last_palm_delta_robot[0],
            self.last_palm_delta_robot[1],
            self.last_palm_delta_robot[2],
            self.last_scaled_delta[0],
            self.last_scaled_delta[1],
            self.last_scaled_delta[2],
            self.last_target_raw_xyz[0],
            self.last_target_raw_xyz[1],
            self.last_target_raw_xyz[2],
            self.palm_to_robot_axis_map[0],
            self.palm_to_robot_axis_map[1],
            self.palm_to_robot_axis_map[2],
            self.palm_to_robot_axis_sign[0],
            self.palm_to_robot_axis_sign[1],
            self.palm_to_robot_axis_sign[2],
            self.last_reach_radius,
            self.last_effective_reach_radius,
            self.last_reach_radius_clamped_flag,
            1.0 if self.target_mapping_mode == "absolute" else 0.0,
            1.0 if self.target_mapping_mode == "absolute" else 0.0,
            self.mr_min[0],
            self.mr_max[0],
            self.mr_min[1],
            self.mr_max[1],
            self.mr_min[2],
            self.mr_max[2],
            self.abs_robot_min[0],
            self.abs_robot_max[0],
            self.abs_robot_min[1],
            self.abs_robot_max[1],
            self.abs_robot_min[2],
            self.abs_robot_max[2],
            self.last_target_before_safety_xyz[0],
            self.last_target_before_safety_xyz[1],
            self.last_target_before_safety_xyz[2],
            self.last_target_control_xyz[0],
            self.last_target_control_xyz[1],
            self.last_target_control_xyz[2],
            self.last_target_kinematics_xyz[0],
            self.last_target_kinematics_xyz[1],
            self.last_target_kinematics_xyz[2],
            1.0 if self.enable_kinematics_target_adapter else 0.0,
            1.0 if self.kinematics_target_adapter_mode == "control_to_urdf_y_forward" else 0.0,
        ])

    def log_debug_state(self, now, target_xyz):
        if now - self.last_debug_log_time < 1.0:
            return
        self.last_debug_log_time = now
        self.get_logger().info(
            "palm_ik_debug "
            f"palm_now_mr={np.round(self.last_palm_now_mr, 4).tolist()} "
            f"control_center_mr={np.round(self.control_center_mr, 4).tolist()} "
            f"palm_delta_raw={np.round(self.last_palm_delta_raw, 4).tolist()} "
            f"palm_delta_robot={np.round(self.last_palm_delta_robot, 4).tolist()} "
            f"scaled_delta={np.round(self.last_scaled_delta, 4).tolist()} "
            f"target_raw_xyz={np.round(self.last_target_raw_xyz, 4).tolist()} "
            f"target_raw_before_safety={np.round(self.last_target_before_safety_xyz, 4).tolist()} "
            f"target_xyz_control_frame={np.round(self.last_target_control_xyz, 4).tolist()} "
            f"target_xyz_kinematics_frame={np.round(self.last_target_kinematics_xyz, 4).tolist()} "
            f"enable_kinematics_target_adapter={self.enable_kinematics_target_adapter} "
            f"kinematics_target_adapter_mode={self.kinematics_target_adapter_mode} "
            f"reach_radius={self.last_reach_radius:.4f} "
            f"effective_reach_radius={self.last_effective_reach_radius:.4f} "
            f"reach_radius_clamped={self.last_reach_radius_clamped_flag:.0f} "
            f"target_mapping_mode={self.target_mapping_mode} "
            f"absolute_mapping_enabled={int(self.target_mapping_mode == 'absolute')} "
            f"absolute_axis_mapping={self.absolute_axis_mapping} "
            f"palm_target_mode={self.palm_target_mode} "
            f"palm_abs_robot={np.round(self.last_palm_abs_robot, 4).tolist()} "
            f"r_palm={self.last_r_palm:.4f} "
            f"y_palm={self.last_y_palm:.4f} "
            f"z_palm={self.last_z_palm:.4f} "
            f"r_target={self.last_r_target:.4f} "
            f"y_target={self.last_y_target:.4f} "
            f"z_target={self.last_z_target:.4f} "
            f"target_xyz={np.round(target_xyz, 4).tolist()} "
            f"axis_map={self.palm_to_robot_axis_map.tolist()} "
            f"axis_sign={self.palm_to_robot_axis_sign.tolist()} "
            f"palm_motion_gain={self.palm_motion_gain:.4f} "
            f"max_target_step_m={self.max_target_step_m:.4f} "
            f"trajectory_time_from_start_sec={self.trajectory_time_from_start_sec:.4f} "
            f"debug_target_step_x={self.debug_target_step_x:.4f} "
            f"q_current={np.round(self.last_q_current, 4).tolist()} "
            f"q_solution={np.round(self.last_q_solution, 4).tolist()} "
            f"dq={np.round(self.last_dq, 4).tolist()} "
            f"fk_before={np.round(self.last_fk_before, 4).tolist()} "
            f"fk_after={np.round(self.last_fk_after, 4).tolist()} "
            f"target_delta={np.round(self.last_target_delta, 4).tolist()} "
            f"error_before={self.last_error_before:.5f} "
            f"error_after={self.last_error_norm:.5f} "
            f"extension_weight={self.last_extension_weight:.3f} "
            f"extension_use_motion_norm={self.extension_use_motion_norm} "
            f"extension_trigger_norm={self.extension_trigger_norm:.4f} "
            f"motion_norm={self.last_motion_norm:.4f} "
            f"enable_range_scaled_palm={self.enable_range_scaled_palm} "
            f"human_motion_range={self.human_motion_range.tolist()} "
            f"robot_motion_range={self.robot_motion_range.tolist()} "
            f"range_scale_gain={self.range_scale_gain:.4f} "
            f"range_scale={np.round(self.last_range_scale, 4).tolist()} "
            f"enable_reach_approach={self.enable_reach_approach} "
            f"reach_error={self.last_reach_error:.4f} "
            f"reach_weight={self.last_reach_weight:.3f} "
            f"reach_vec={np.round(self.last_reach_vec, 4).tolist()} "
            f"q_reach={np.round(self.last_q_reach, 4).tolist()} "
            f"reach_bias_gain={self.reach_bias_gain:.4f} "
            f"reach_directional_yaw={self.reach_directional_yaw} "
            f"desired_reach_yaw={self.last_desired_reach_yaw:.4f} "
            f"limit_hit={self.last_limit_hit:.0f}")

    def log_real_relative_safety_state(self, target_xyz_control):
        if not self.is_real_relative_mode():
            return
        anchor_ee = (
            self.ee_anchor_xyz
            if self.ee_anchor_xyz is not None else
            np.zeros(3, dtype=np.float64)
        )
        palm_anchor = (
            self.palm_anchor_xyz
            if self.palm_anchor_xyz is not None else
            np.zeros(3, dtype=np.float64)
        )
        reason = (
            "accepted"
            if self.last_candidate_accepted else
            self.last_reject_reason
        )
        self.get_logger().info(
            "real_relative_safety "
            f"anchor_ee={np.round(anchor_ee, 5).tolist()} "
            f"palm_anchor={np.round(palm_anchor, 5).tolist()} "
            f"delta_hand={np.round(self.last_palm_delta_raw, 5).tolist()} "
            f"delta_robot={np.round(self.last_palm_delta_robot, 5).tolist()} "
            f"scaled_delta={np.round(self.last_scaled_delta, 5).tolist()} "
            f"target_control={np.round(target_xyz_control, 5).tolist()} "
            f"target_minus_anchor_ee="
            f"{np.round(self.last_target_minus_anchor_ee, 5).tolist()} "
            f"anchor_error_norm={self.anchor_error_norm:.6f} "
            f"coordinate_transform_mode={self.last_coordinate_transform_mode} "
            f"q_current={np.round(self.last_q_current, 5).tolist()} "
            f"q_solution={np.round(self.last_q_candidate, 5).tolist()} "
            f"error_before={self.last_error_before:.6f} "
            f"error_after={self.last_error_norm:.6f} "
            f"accepted={self.last_candidate_accepted} "
            f"reason={reason}")

    def is_joint_state_fresh(self, now):
        if not self.joint_state_received or self.last_joint_state_time is None:
            return False
        if self.joint_state_timeout_sec <= 0.0:
            return True
        return (now - self.last_joint_state_time) <= self.joint_state_timeout_sec

    def log_teleop_servo_state(self, p_desired, p_command, reason):
        anchor_ee_log = None
        if self.ee_anchor_xyz is not None:
            anchor_ee_log = np.round(self.ee_anchor_xyz, 5).tolist()
        self.get_logger().info(
            f"{self.control_profile} "
            f"servo_rate_hz={self.servo_rate_hz:.2f} "
            f"reference_servo_rate_hz={self.reference_servo_rate_hz:.2f} "
            f"reference_dt_sec={self.reference_dt_sec:.5f} "
            f"actual_reference_tick_hz={self.actual_reference_tick_hz:.2f} "
            f"palm_delta={np.round(self.last_palm_delta_raw, 5).tolist()} "
            f"p_desired={np.round(p_desired, 5).tolist()} "
            f"p_command={np.round(p_command, 5).tolist()} "
            f"workspace_check_frame={self.workspace_check_frame} "
            f"workspace_bounds={{'min': "
            f"{np.round(self.workspace_min, 5).tolist()}, 'max': "
            f"{np.round(self.workspace_max, 5).tolist()}}} "
            f"per_axis_margin_to_workspace="
            f"{np.round(self.last_workspace_margins, 5).tolist()} "
            f"anchor_ee={anchor_ee_log} "
            f"adapter_applied={self.last_adapter_applied} "
            f"coordinate_transform_mode={self.last_coordinate_transform_mode} "
            f"q_measured={np.round(self.last_q_current, 5).tolist()} "
            f"q_command={np.round(self.last_q_candidate, 5).tolist()} "
            f"q_reference="
            f"{np.round(self.q_reference, 5).tolist() if self.q_reference is not None else None} "
            f"reference_reset_reason={self.reference_reset_reason} "
            f"reference_reset_applied={self.reference_reset_applied} "
            f"tracking_error_rad={self.tracking_error_rad:.5f} "
            f"reference_tracking_error_threshold_rad="
            f"{self.reference_tracking_error_threshold_rad:.5f} "
            f"tracking_error_joint_diffs="
            f"{np.round(self.tracking_error_joint_diffs, 5).tolist()} "
            f"joint_velocity_command="
            f"{np.round(self.last_servo_joint_velocity, 5).tolist()} "
            f"accepted={self.last_candidate_accepted} "
            f"reason={reason} "
            f"latency_ms={self.last_servo_latency_ms:.2f}",
            throttle_duration_sec=0.5)

    def make_servo_trajectory(self, q_command, q_measured, velocity_command):
        if (self.mode == "real" and
                is_teleop_trajectory_profile(self.control_profile)):
            return make_real_teleop_joint_trajectory(
                q_command,
                self.servo_horizon_sec,
                JOINT_NAMES)
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in q_command]
        point.velocities = [float(value) for value in velocity_command]
        point.time_from_start = duration_from_seconds(self.servo_horizon_sec)
        msg.points = [point]
        return msg

    def make_velocity_servo_trajectory(self, joint_velocity_command):
        return make_velocity_joint_trajectory(
            joint_velocity_command,
            self.velocity_servo_time_from_start_sec,
            JOINT_NAMES)

    def log_servo_trajectory_format(self, msg, known_good_format, reason):
        fields = trajectory_log_fields(
            msg, self.mode, self.control_profile)
        self.get_logger().info(
            "teleop trajectory publish_prepare "
            f"known_good_format={str(bool(known_good_format)).lower()} "
            f"header_stamp={fields['header_stamp']} "
            f"positions={np.round(fields['positions'], 5).tolist()} "
            f"velocities_length={fields['velocities_length']} "
            f"time_from_start={fields['time_from_start']} "
            f"mode={fields['mode']} "
            f"control_profile={fields['control_profile']} "
            f"reason={reason}")

    def publish_servo_trajectory(self, q_command, q_measured, velocity_command):
        if self.trajectory_pub is None:
            self.last_candidate_accepted = False
            self.last_reject_reason = "trajectory_publisher_unavailable"
            return False
        msg = self.make_servo_trajectory(
            q_command, q_measured, velocity_command)
        known_good_format = (
            self.mode == "real" and
            is_teleop_trajectory_profile(self.control_profile))
        self.log_servo_trajectory_format(
            msg, known_good_format, self.last_reject_reason)
        if self.dry_run:
            self.get_logger().info(
                "dry_run active: teleop_trajectory_stream "
                "JointTrajectory not published "
                f"known_good_format={str(bool(known_good_format)).lower()} "
                f"reason={self.last_reject_reason}",
                throttle_duration_sec=0.5)
            return True
        self.trajectory_pub.publish(msg)
        if known_good_format:
            self.get_logger().info(
                "teleop trajectory published known_good_format=true")
        return True

    def buffer_publish_decision(self, q_target, now):
        q_target = np.array(q_target, dtype=np.float64)
        q_reference = (
            self.q_reference
            if self.q_reference is not None else
            q_target.copy()
        )
        at_rest = (
            float(np.linalg.norm(q_target - q_reference)) <= 1e-6 and
            float(np.linalg.norm(self.q_reference_velocity)) <= 1e-6
        )
        zero_deadzone = (
            float(np.linalg.norm(self.last_scaled_delta)) <= 1e-12 and
            at_rest
        )
        if zero_deadzone and self.last_buffer_q_target is None:
            return False, "zero_delta_no_publish"
        if (self.last_buffer_q_target is not None and
                np.allclose(
                    q_target,
                    self.last_buffer_q_target,
                    atol=1e-5,
                    rtol=0.0) and
                at_rest):
            return False, "unchanged_target_no_publish"
        if self.last_buffer_publish_time is not None:
            min_period = 1.0 / max(
                self.trajectory_buffer_update_rate_hz, 1e-6)
            if (now - self.last_buffer_publish_time) < min_period:
                return False, "update_rate_limited_no_publish"
        if self.last_buffer_q_target is None:
            return True, "first_buffer"
        return True, "target_updated"

    def verify_buffer_limits(self, velocities, times, previous_velocity):
        velocity_values = np.array(velocities, dtype=np.float64)
        time_values = [float(value) for value in times]
        if velocity_values.size == 0:
            return True
        if np.any(
                np.abs(velocity_values) >
                self.max_joint_velocity_radps + 1e-9):
            self.last_reject_reason = "buffer_joint_velocity_limit_exceeded"
            return False
        previous_velocity = np.array(previous_velocity, dtype=np.float64)
        previous_time = 0.0
        for idx, time_value in enumerate(time_values):
            dt = max(time_value - previous_time, 1e-6)
            acceleration = (
                velocity_values[idx] - previous_velocity) / dt
            if np.any(
                    np.abs(acceleration) >
                    self.max_joint_acceleration_radps2 + 1e-9):
                self.last_reject_reason = (
                    "buffer_joint_acceleration_limit_exceeded")
                return False
            previous_velocity = velocity_values[idx]
            previous_time = time_value
        return True

    def log_buffer_trajectory_format(
            self, msg, q_measured, q_reference, q_target,
            publish_reason, reason):
        fields = trajectory_log_fields(
            msg, self.mode, self.control_profile)
        self.get_logger().info(
            "teleop_trajectory_buffer publish_prepare "
            f"buffer_update_rate_hz="
            f"{self.trajectory_buffer_update_rate_hz:.2f} "
            f"reference_servo_rate_hz={self.reference_servo_rate_hz:.2f} "
            f"reference_dt_sec={self.reference_dt_sec:.5f} "
            f"actual_reference_tick_hz={self.actual_reference_tick_hz:.2f} "
            f"horizon_sec={self.trajectory_buffer_horizon_sec:.3f} "
            f"point_dt_sec={self.trajectory_buffer_point_dt_sec:.3f} "
            f"waypoint_count={fields['point_count']} "
            f"first_waypoint="
            f"{np.round(fields['first_waypoint'], 5).tolist()} "
            f"last_waypoint="
            f"{np.round(fields['last_waypoint'], 5).tolist()} "
            f"q_measured={np.round(q_measured, 5).tolist()} "
            f"q_reference={np.round(q_reference, 5).tolist()} "
            f"q_target={np.round(q_target, 5).tolist()} "
            f"reference_reset_reason={self.reference_reset_reason} "
            f"reference_reset_applied={self.reference_reset_applied} "
            f"tracking_error_rad={self.tracking_error_rad:.5f} "
            f"reference_tracking_error_threshold_rad="
            f"{self.reference_tracking_error_threshold_rad:.5f} "
            f"tracking_error_joint_diffs="
            f"{np.round(self.tracking_error_joint_diffs, 5).tolist()} "
            f"accepted={self.last_candidate_accepted} "
            f"reason={reason} "
            f"publish_reason={publish_reason} "
            "known_good_format=true",
            throttle_duration_sec=0.5)

    def publish_buffer_trajectory(self, q_target, q_measured, now, reason):
        if self.trajectory_pub is None:
            self.last_candidate_accepted = False
            self.last_reject_reason = "trajectory_publisher_unavailable"
            return False
        if not self.q_reference_valid or self.q_reference is None:
            self.reset_reference_from_measured(
                q_measured, "reference_reinitialized_before_publish")
        q_reference = self.q_reference.copy()
        q_reference_velocity = self.q_reference_velocity.copy()
        buffer_data = predict_joint_reference_buffer(
            q_reference,
            q_target,
            q_reference_velocity,
            self.trajectory_buffer_horizon_sec,
            self.trajectory_buffer_point_dt_sec,
            1.0 / max(self.reference_servo_rate_hz, 1e-6),
            self.max_joint_velocity_radps,
            self.max_joint_acceleration_radps2)
        if not self.verify_buffer_limits(
                buffer_data["velocities"],
                buffer_data["times"],
                q_reference_velocity):
            diagnostic_reason = self.last_reject_reason
            self.get_logger().warn(
                "teleop_trajectory_buffer limit diagnostic failed after "
                "constrained prediction; publish is not rejected by the "
                f"diagnostic reason={diagnostic_reason}",
                throttle_duration_sec=1.0)
            self.last_reject_reason = reason

        msg = make_real_teleop_buffer_joint_trajectory(
            buffer_data["waypoints"],
            buffer_data["times"],
            JOINT_NAMES)
        should_publish, publish_reason = self.buffer_publish_decision(
            q_target, now)
        self.last_buffer_publish_reason = publish_reason
        self.last_buffer_waypoints = [
            list(waypoint) for waypoint in buffer_data["waypoints"]
        ]
        self.log_buffer_trajectory_format(
            msg,
            q_measured,
            q_reference,
            q_target,
            publish_reason,
            reason)

        if not should_publish:
            if publish_reason in (
                    "zero_delta_no_publish",
                    "unchanged_target_no_publish"):
                self.last_buffer_q_target = np.array(
                    q_target, dtype=np.float64)
            return True
        if self.dry_run:
            self.get_logger().info(
                "dry_run active: teleop_trajectory_buffer "
                "JointTrajectory not published known_good_format=true "
                f"publish_reason={publish_reason} "
                f"reason={reason}",
                throttle_duration_sec=0.5)
        else:
            self.trajectory_pub.publish(msg)
            self.get_logger().info(
                "teleop trajectory buffer published "
                "known_good_format=true")
        self.last_buffer_publish_time = now
        self.last_buffer_q_target = np.array(q_target, dtype=np.float64)
        return True

    def log_velocity_servo_trajectory_format(self, msg, reason):
        fields = trajectory_log_fields(
            msg, self.mode, self.control_profile)
        self.get_logger().info(
            "velocity_servo trajectory publish_prepare "
            f"header_stamp={fields['header_stamp']} "
            f"positions_length={fields['positions_length']} "
            f"velocities={np.round(fields['velocities'], 5).tolist()} "
            f"velocities_length={fields['velocities_length']} "
            f"time_from_start={fields['time_from_start']} "
            f"mode={fields['mode']} "
            f"control_profile={fields['control_profile']} "
            f"hardware_confirmed={self.velocity_servo_hardware_confirmed} "
            f"reason={reason}")

    def log_velocity_servo_hardware_diagnostic(self):
        if self.velocity_servo_diagnostic_logged:
            return
        self.velocity_servo_diagnostic_logged = True
        self.get_logger().warn(
            "teleop_velocity_servo hardware diagnostic: repository "
            "configuration exports velocity command interfaces in "
            "amir_description/urdf/amir_mecanum3.xacro and "
            "amir_driver/config/amir_ros2_controllers.yaml. Runtime "
            "controller state is not verified by this node. Confirm with "
            "`ros2 control list_hardware_interfaces`, "
            "`ros2 control list_controllers`, and "
            "`ros2 topic info -v /arm_controller/joint_trajectory`. "
            "Set velocity_servo_hardware_confirmed:=true only after "
            "the active controller accepts velocity trajectory points.")

    def publish_velocity_servo_trajectory(self, velocity_command, reason):
        if self.trajectory_pub is None:
            self.last_candidate_accepted = False
            self.last_reject_reason = "trajectory_publisher_unavailable"
            return False
        msg = self.make_velocity_servo_trajectory(velocity_command)
        self.log_velocity_servo_trajectory_format(msg, reason)
        self.log_velocity_servo_hardware_diagnostic()
        if self.dry_run:
            self.get_logger().info(
                "dry_run active: teleop_velocity_servo JointTrajectory "
                "not published "
                f"velocity={np.round(velocity_command, 5).tolist()} "
                f"reason={reason}",
                throttle_duration_sec=0.5)
            return True
        if not self.velocity_servo_hardware_confirmed:
            self.get_logger().error(
                "teleop_velocity_servo publish suppressed: "
                "velocity_servo_hardware_confirmed is false. "
                "Run hardware/controller diagnostics before enabling "
                "velocity servo output.",
                throttle_duration_sec=1.0)
            return False
        self.trajectory_pub.publish(msg)
        self.get_logger().info(
            "teleop_velocity_servo trajectory published "
            "velocity_only=true")
        return True

    def publish_zero_velocity_once(self, reason):
        if self.velocity_servo_zero_sent_after_stop:
            return
        zero_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.last_servo_joint_velocity = zero_velocity.copy()
        self.last_candidate_accepted = False
        self.last_reject_reason = reason
        self.publish_velocity_servo_trajectory(zero_velocity, reason)
        self.velocity_servo_zero_sent_after_stop = True

    def publish_hold_trajectory_once(self, reason):
        if self.hold_sent_after_stop:
            return
        if not self.joint_state_received:
            self.last_reject_reason = "hold_without_joint_state"
            if should_invalidate_reference(reason):
                self.invalidate_reference(reason)
            return
        q_hold = self.current_joints.copy()
        zero_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.last_q_current = q_hold.copy()
        self.last_q_candidate = q_hold.copy()
        self.last_servo_joint_velocity = zero_velocity.copy()
        self.last_candidate_accepted = False
        self.last_reject_reason = reason
        if self.trajectory_pub is None:
            self.get_logger().error(
                "hold trajectory suppressed: trajectory publisher unavailable",
                throttle_duration_sec=1.0)
            if should_invalidate_reference(reason):
                self.invalidate_reference(reason)
            return
        msg = self.make_servo_trajectory(q_hold, q_hold, zero_velocity)
        known_good_format = (
            self.mode == "real" and
            is_teleop_trajectory_profile(self.control_profile))
        self.log_servo_trajectory_format(msg, known_good_format, reason)
        if self.dry_run:
            self.get_logger().info(
                "dry_run active: hold JointTrajectory not published "
                f"reason={reason} q_hold={np.round(q_hold, 5).tolist()}",
                throttle_duration_sec=1.0)
        else:
            self.trajectory_pub.publish(msg)
            if known_good_format:
                self.get_logger().info(
                    "teleop trajectory published known_good_format=true")
            self.get_logger().warn(
                f"hold trajectory published once reason={reason}",
                throttle_duration_sec=1.0)
        self.hold_sent_after_stop = True
        if should_invalidate_reference(reason):
            self.invalidate_reference(reason)

    def log_forward_position_command(
            self, q_measured, q_reference, q_target, published_data,
            publish_reason, accepted, reason):
        q_reference_log = (
            np.round(q_reference, 5).tolist()
            if q_reference is not None else
            None
        )
        q_target_log = (
            np.round(q_target, 5).tolist()
            if q_target is not None else
            None
        )
        published_data_log = (
            np.round(published_data, 5).tolist()
            if published_data is not None else
            None
        )
        self.get_logger().info(
            "teleop_forward_position publish_prepare "
            f"control_profile={self.control_profile} "
            f"forward_position_command_topic="
            f"{self.forward_position_command_topic} "
            f"forward_position_publish_rate_hz="
            f"{self.forward_position_publish_rate_hz:.2f} "
            f"reference_servo_rate_hz={self.reference_servo_rate_hz:.2f} "
            f"reference_dt_sec={self.reference_dt_sec:.5f} "
            f"actual_reference_tick_hz={self.actual_reference_tick_hz:.2f} "
            f"q_measured={np.round(q_measured, 5).tolist()} "
            f"q_reference={q_reference_log} "
            f"q_target={q_target_log} "
            f"published_data={published_data_log} "
            f"published_data_length="
            f"{len(published_data) if published_data is not None else 0} "
            f"publish_reason={publish_reason} "
            f"accepted={accepted} "
            f"reason={reason} "
            f"reference_reset_reason={self.reference_reset_reason} "
            f"reference_reset_applied={self.reference_reset_applied} "
            f"tracking_error_rad={self.tracking_error_rad:.5f} "
            f"reference_tracking_error_threshold_rad="
            f"{self.reference_tracking_error_threshold_rad:.5f} "
            f"tracking_error_joint_diffs="
            f"{np.round(self.tracking_error_joint_diffs, 5).tolist()}",
            throttle_duration_sec=0.5)

    def publish_forward_position_command(
            self, q_reference, q_measured, q_target, publish_reason, reason):
        if self.forward_position_pub is None:
            self.last_candidate_accepted = False
            self.last_reject_reason = "forward_position_publisher_unavailable"
            self.log_forward_position_command(
                q_measured,
                q_reference,
                q_target,
                None,
                publish_reason,
                False,
                self.last_reject_reason)
            return False
        try:
            msg = make_forward_position_command(
                q_reference,
                JOINT_NAMES,
                JOINT_LIMITS.tolist())
        except ValueError as exc:
            self.last_candidate_accepted = False
            self.last_reject_reason = "forward_position_command_invalid"
            self.log_forward_position_command(
                q_measured,
                q_reference,
                q_target,
                None,
                publish_reason,
                False,
                self.last_reject_reason)
            self.get_logger().warn(
                "forward position command rejected; publish skipped "
                f"reason={exc}",
                throttle_duration_sec=0.5)
            return False

        fields = forward_position_log_fields(
            msg, self.mode, self.control_profile)
        published_data = np.array(fields["data"], dtype=np.float64)
        self.log_forward_position_command(
            q_measured,
            q_reference,
            q_target,
            published_data,
            publish_reason,
            self.last_candidate_accepted,
            reason)
        if self.dry_run:
            self.get_logger().info(
                "dry_run active: teleop_forward_position "
                "Float64MultiArray not published "
                f"topic={self.forward_position_command_topic} "
                f"data={np.round(published_data, 5).tolist()} "
                f"publish_reason={publish_reason} "
                f"reason={reason}",
                throttle_duration_sec=0.5)
        else:
            self.forward_position_pub.publish(msg)
            self.get_logger().info(
                "teleop_forward_position command published "
                "type=std_msgs/msg/Float64MultiArray data_only=true",
                throttle_duration_sec=0.5)
        self.last_forward_position_publish_time = seconds_now(self)
        self.last_forward_position_publish_reason = str(publish_reason)
        self.last_forward_position_data = fields["data"]
        return True

    def publish_forward_position_hold_once(self, reason):
        if self.forward_position_hold_sent_after_stop:
            return
        if not self.joint_state_received:
            self.last_reject_reason = "hold_without_joint_state"
            if should_invalidate_reference(reason):
                self.invalidate_reference(reason)
            return
        q_hold = self.current_joints.copy()
        self.last_q_current = q_hold.copy()
        self.last_q_candidate = q_hold.copy()
        self.last_q_solution = q_hold.copy()
        self.last_dq = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.last_servo_joint_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self.last_candidate_accepted = False
        self.last_reject_reason = reason
        published = self.publish_forward_position_command(
            q_hold,
            q_hold,
            q_hold,
            "hold_after_stop",
            reason)
        if published:
            self.forward_position_hold_sent_after_stop = True
            self.get_logger().warn(
                "forward position hold published once "
                f"reason={reason} q_hold={np.round(q_hold, 5).tolist()}",
                throttle_duration_sec=1.0)
        if should_invalidate_reference(reason):
            self.invalidate_reference(reason)

    def reject_teleop_servo(self, reason, p_desired=None, p_command=None):
        self.last_candidate_accepted = False
        self.last_reject_reason = reason
        desired = (
            np.array(p_desired, dtype=np.float64)
            if p_desired is not None else
            self.last_target_raw_xyz.copy()
        )
        command = (
            np.array(p_command, dtype=np.float64)
            if p_command is not None else
            (self.last_servo_p_command.copy()
             if self.last_servo_p_command is not None else
             desired.copy())
        )
        self.log_teleop_servo_state(desired, command, reason)

    def log_velocity_servo_state(
            self, palm_step, cartesian_velocity, q_measured,
            q_velocity, reason):
        self.get_logger().info(
            "teleop_velocity_servo "
            f"servo_rate_hz={self.servo_rate_hz:.2f} "
            f"palm_step={np.round(palm_step, 5).tolist()} "
            f"cartesian_velocity_command="
            f"{np.round(cartesian_velocity, 5).tolist()} "
            f"q_measured={np.round(q_measured, 5).tolist()} "
            f"joint_velocity_command={np.round(q_velocity, 5).tolist()} "
            f"max_joint_velocity_radps={self.max_joint_velocity_radps:.5f} "
            f"hardware_confirmed={self.velocity_servo_hardware_confirmed} "
            f"accepted={self.last_candidate_accepted} "
            f"reason={reason} "
            f"latency_ms={self.last_servo_latency_ms:.2f}",
            throttle_duration_sec=0.5)

    def compute_palm_step_velocity(self, dt):
        if self.last_velocity_servo_palm_xyz is None:
            self.last_velocity_servo_palm_xyz = self.palm_xyz.copy()
            return {
                "palm_step": np.zeros(3, dtype=np.float64),
                "delta_robot": np.zeros(3, dtype=np.float64),
                "scaled_step": np.zeros(3, dtype=np.float64),
                "desired_cartesian_velocity": np.zeros(3, dtype=np.float64),
            }
        palm_step = self.palm_xyz - self.last_velocity_servo_palm_xyz
        if float(np.linalg.norm(palm_step)) < self.deadzone_m:
            palm_step = np.zeros(3, dtype=np.float64)
        delta_robot = np.array([
            self.palm_to_robot_axis_sign[robot_axis] *
            self.palm_axis_gain[robot_axis] *
            palm_step[int(self.palm_to_robot_axis_map[robot_axis])]
            for robot_axis in range(3)
        ], dtype=np.float64)
        scaled_step = self.scale_xyz * delta_robot
        desired_cartesian_velocity = scaled_step / max(dt, 1e-6)
        self.last_velocity_servo_palm_xyz = self.palm_xyz.copy()
        return {
            "palm_step": palm_step,
            "delta_robot": delta_robot,
            "scaled_step": scaled_step,
            "desired_cartesian_velocity": desired_cartesian_velocity,
        }

    def clamp_velocity_at_joint_limits(self, q_measured, q_velocity, dt):
        q_next = q_measured + q_velocity * dt
        q_next = clamp_array(q_next, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        return (q_next - q_measured) / max(dt, 1e-6)

    def update_reference_timer_stats(self, now):
        if self.last_reference_tick_time is None:
            self.reference_dt_sec = 1.0 / max(
                self.reference_servo_rate_hz, 1e-6)
        else:
            self.reference_dt_sec = max(
                now - self.last_reference_tick_time, 1e-6)
        self.actual_reference_tick_hz = 1.0 / self.reference_dt_sec
        self.last_reference_tick_time = now
        return self.reference_dt_sec

    def reference_servo_tick(self):
        forward_position_profile = is_teleop_forward_position_profile(
            self.control_profile)
        if (not is_teleop_buffer_profile(self.control_profile) and
                not forward_position_profile):
            return
        now = seconds_now(self)
        workspace_policy = teleop_servo_workspace_policy(
            self.mode,
            self.target_mapping_mode,
            self.control_profile,
            self.enable_kinematics_target_adapter and
            self.kinematics_target_adapter_mode != "none")
        self.workspace_check_frame = workspace_policy[
            "workspace_check_frame"]
        self.last_adapter_applied = workspace_policy["adapter_applied"]
        if not self.last_adapter_applied:
            self.last_coordinate_transform_mode = (
                "real_relative_fk_frame_no_adapter")
        self.last_servo_latency_ms = (
            max(now - self.last_palm_time, 0.0) * 1000.0
            if self.last_palm_time is not None else
            float("inf")
        )
        dt = self.update_reference_timer_stats(now)
        palm_timed_out = (
            self.last_palm_time is None or
            (now - self.last_palm_time) > self.input_timeout_sec
        )
        joint_state_timed_out = not self.is_joint_state_fresh(now)
        deadman_active = self.is_deadman_active(now)
        stop_reason = servo_stop_reason(
            palm_timed_out, joint_state_timed_out, deadman_active)
        if stop_reason:
            self.buffer_target_available = False
            self.reject_teleop_servo(stop_reason)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)),
                1.0 if stop_reason != "deadman_inactive" else 0.0)
            return

        if not self.is_real_relative_mode():
            self.buffer_target_available = False
            self.reject_teleop_servo(
                f"{self.control_profile}_requires_real_relative")
            return
        if not self.has_anchor:
            self.set_real_relative_anchor()
            self.last_q_current = self.current_joints.copy()
            self.last_q_candidate = self.current_joints.copy()
            self.latest_q_target = self.current_joints.copy()
            self.buffer_target_available = True
            self.log_teleop_servo_state(
                self.ee_anchor_xyz, self.ee_anchor_xyz, "anchor_initialized")
            self.publish_metrics(0.0, 0.0)
            return
        if self.skip_next_command_after_anchor:
            self.skip_next_command_after_anchor = False
            self.last_q_current = self.current_joints.copy()
            self.last_q_candidate = self.current_joints.copy()
            self.latest_q_target = self.current_joints.copy()
            self.buffer_target_available = True
            self.log_teleop_servo_state(
                self.ee_anchor_xyz, self.ee_anchor_xyz, "anchor_cycle_no_publish")
            self.publish_metrics(0.0, 0.0)
            return

        p_desired = self.make_real_relative_target()
        previous_p_command = (
            self.last_servo_p_command
            if self.last_servo_p_command is not None else
            self.ee_anchor_xyz
        )
        cartesian_command = limit_cartesian_command(
            p_desired,
            previous_p_command,
            self.last_servo_cartesian_velocity,
            dt,
            self.max_cartesian_velocity_mps,
            self.max_cartesian_acceleration_mps2)
        p_command = np.array(cartesian_command["position"], dtype=np.float64)
        cartesian_velocity = np.array(
            cartesian_command["velocity"], dtype=np.float64)
        self.last_workspace_margins = np.array(
            workspace_axis_margins(
                p_command, self.workspace_min, self.workspace_max),
            dtype=np.float64)
        if not is_target_within_workspace(
                p_command, self.workspace_min, self.workspace_max):
            self.buffer_target_available = False
            self.reject_teleop_servo(
                "target_outside_workspace", p_desired, p_command)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return

        q_measured = self.current_joints.copy()
        fk_before = self.fk_xyz(q_measured)
        error_before = float(np.linalg.norm(p_command - fk_before))
        zero_delta_hold = (
            float(np.linalg.norm(self.last_scaled_delta)) <= 1e-9 and
            self.ee_anchor_xyz is not None and
            float(np.linalg.norm(p_command - self.ee_anchor_xyz)) <= 1e-9
        )
        if zero_delta_hold:
            q_target = q_measured.copy()
        else:
            q_target, base_error_after = lm_ik(
                p_command, q_measured.copy(), fk_func=self.fk_xyz)
            fk_after = self.fk_xyz(q_target)
            error_after = float(np.linalg.norm(p_command - fk_after))
            if base_error_after > self.servo_ik_tolerance_m:
                self.last_q_current = q_measured.copy()
                self.last_q_candidate = q_target.copy()
                self.last_error_before = error_before
                self.last_error_norm = error_after
                self.buffer_target_available = False
                self.reject_teleop_servo(
                    "ik_not_converged", p_desired, p_command)
                return
            if error_after > error_before + 1e-5:
                self.last_q_current = q_measured.copy()
                self.last_q_candidate = q_target.copy()
                self.last_error_before = error_before
                self.last_error_norm = error_after
                self.buffer_target_available = False
                self.reject_teleop_servo(
                    "ik_error_increased", p_desired, p_command)
                return
            if (np.any(q_target < JOINT_LIMITS[:, 0] - 1e-9) or
                    np.any(q_target > JOINT_LIMITS[:, 1] + 1e-9)):
                self.last_q_current = q_measured.copy()
                self.last_q_candidate = q_target.copy()
                self.buffer_target_available = False
                self.reject_teleop_servo(
                    "joint_limit_violation", p_desired, p_command)
                return

        tracking_reset_hold = self.step_reference_toward_target(
            q_target,
            q_measured,
            dt,
            hold_on_tracking_reset=forward_position_profile)
        q_command = self.q_reference.copy()
        self.latest_q_target = q_target.copy()
        self.latest_p_desired = p_desired.copy()
        self.latest_p_command = p_command.copy()
        self.buffer_target_available = True
        self.last_candidate_accepted = not tracking_reset_hold
        if tracking_reset_hold:
            self.last_reject_reason = "tracking_error_exceeded"
            self.forward_position_pending_publish_reason = "tracking_error_hold"
        else:
            self.last_reject_reason = (
                "zero_delta_hold" if zero_delta_hold else "accepted")
            self.forward_position_pending_publish_reason = (
                "zero_delta_hold" if zero_delta_hold else "reference_update")
        self.last_q_current = q_measured.copy()
        self.last_q_candidate = q_command.copy()
        self.last_q_solution = q_target.copy()
        self.last_dq = q_command - q_measured
        self.last_fk_before = fk_before.copy()
        self.last_fk_after = self.fk_xyz(q_command)
        self.last_error_before = error_before
        self.last_error_norm = float(
            np.linalg.norm(p_command - self.last_fk_after))
        self.last_servo_p_command = p_command.copy()
        self.last_servo_cartesian_velocity = cartesian_velocity.copy()
        self.last_servo_time = now
        self.hold_sent_after_stop = False
        self.forward_position_hold_sent_after_stop = False
        self.log_teleop_servo_state(
            p_desired, p_command, self.last_reject_reason)
        self.publish_joint_outputs(q_command)
        self.publish_metrics(
            float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)

    def control_loop_teleop_buffer_publish(self, now):
        palm_timed_out = (
            self.last_palm_time is None or
            (now - self.last_palm_time) > self.input_timeout_sec
        )
        joint_state_timed_out = not self.is_joint_state_fresh(now)
        deadman_active = self.is_deadman_active(now)
        stop_reason = servo_stop_reason(
            palm_timed_out, joint_state_timed_out, deadman_active)
        if stop_reason == "joint_state_timeout":
            self.reject_teleop_servo(stop_reason)
            self.publish_metrics(0.0, 1.0)
            return
        if stop_reason in ("palm_pose_timeout", "deadman_inactive"):
            self.publish_hold_trajectory_once(stop_reason)
            self.reject_teleop_servo(stop_reason)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 1.0)
            return
        if not self.buffer_target_available or self.latest_q_target is None:
            self.reject_teleop_servo("reference_target_not_ready")
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return
        if not self.q_reference_valid or self.q_reference is None:
            self.reject_teleop_servo("reference_not_ready")
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return

        q_measured = self.current_joints.copy()
        q_target = self.latest_q_target.copy()
        reason = self.last_reject_reason
        if self.publish_buffer_trajectory(q_target, q_measured, now, reason):
            if self.last_buffer_waypoints:
                self.last_command_joints = np.array(
                    self.last_buffer_waypoints[-1], dtype=np.float64)
            else:
                self.last_command_joints = self.q_reference.copy()
        self.publish_metrics(
            float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)

    def control_loop_teleop_forward_position(self, now):
        palm_timed_out = (
            self.last_palm_time is None or
            (now - self.last_palm_time) > self.input_timeout_sec
        )
        joint_state_timed_out = not self.is_joint_state_fresh(now)
        deadman_active = self.is_deadman_active(now)
        stop_reason = servo_stop_reason(
            palm_timed_out, joint_state_timed_out, deadman_active)
        if stop_reason == "joint_state_timeout":
            self.reject_teleop_servo(stop_reason)
            self.publish_metrics(0.0, 1.0)
            return
        if stop_reason in ("palm_pose_timeout", "deadman_inactive"):
            self.publish_forward_position_hold_once(stop_reason)
            self.reject_teleop_servo(stop_reason)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 1.0)
            return
        if not self.buffer_target_available or self.latest_q_target is None:
            self.reject_teleop_servo("reference_target_not_ready")
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return
        if not self.q_reference_valid or self.q_reference is None:
            self.reject_teleop_servo("reference_not_ready")
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return

        q_measured = self.current_joints.copy()
        q_reference = self.q_reference.copy()
        q_target = self.latest_q_target.copy()
        publish_reason = self.forward_position_pending_publish_reason
        if publish_reason in ("not_started", "reference_update"):
            publish_reason = "periodic_reference"
        reason = self.last_reject_reason
        if self.publish_forward_position_command(
                q_reference, q_measured, q_target, publish_reason, reason):
            self.last_command_joints = q_reference.copy()
        self.publish_metrics(
            float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)

    def control_loop_teleop_servo(self, now):
        workspace_policy = teleop_servo_workspace_policy(
            self.mode,
            self.target_mapping_mode,
            self.control_profile,
            self.enable_kinematics_target_adapter and
            self.kinematics_target_adapter_mode != "none")
        self.workspace_check_frame = workspace_policy[
            "workspace_check_frame"]
        self.last_adapter_applied = workspace_policy["adapter_applied"]
        if not self.last_adapter_applied:
            self.last_coordinate_transform_mode = (
                "real_relative_fk_frame_no_adapter")
        self.last_servo_latency_ms = (
            max(now - self.last_palm_time, 0.0) * 1000.0
            if self.last_palm_time is not None else
            float("inf")
        )
        palm_timed_out = (
            self.last_palm_time is None or
            (now - self.last_palm_time) > self.input_timeout_sec
        )
        joint_state_timed_out = not self.is_joint_state_fresh(now)
        deadman_active = self.is_deadman_active(now)
        stop_reason = servo_stop_reason(
            palm_timed_out, joint_state_timed_out, deadman_active)
        if stop_reason == "joint_state_timeout":
            self.reject_teleop_servo(stop_reason)
            self.publish_metrics(0.0, 1.0)
            return
        if stop_reason in ("palm_pose_timeout", "deadman_inactive"):
            self.publish_hold_trajectory_once(stop_reason)
            self.reject_teleop_servo(stop_reason)
            self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 1.0)
            return

        if not self.is_real_relative_mode():
            self.reject_teleop_servo("teleop_servo_requires_real_relative")
            return
        if not self.has_anchor:
            self.set_real_relative_anchor()
            self.last_q_current = self.current_joints.copy()
            self.last_q_candidate = self.current_joints.copy()
            self.log_teleop_servo_state(
                self.ee_anchor_xyz, self.ee_anchor_xyz, "anchor_initialized")
            self.publish_metrics(0.0, 0.0)
            return
        if self.skip_next_command_after_anchor:
            self.skip_next_command_after_anchor = False
            self.last_q_current = self.current_joints.copy()
            self.last_q_candidate = self.current_joints.copy()
            self.log_teleop_servo_state(
                self.ee_anchor_xyz, self.ee_anchor_xyz, "anchor_cycle_no_publish")
            self.publish_metrics(0.0, 0.0)
            return

        p_desired = self.make_real_relative_target()
        dt = (
            now - self.last_servo_time
            if self.last_servo_time is not None else
            1.0 / (
                self.reference_servo_rate_hz
                if is_teleop_buffer_profile(self.control_profile) else
                self.servo_rate_hz
            )
        )
        dt = max(dt, 1e-3)
        previous_p_command = (
            self.last_servo_p_command
            if self.last_servo_p_command is not None else
            self.ee_anchor_xyz
        )
        cartesian_command = limit_cartesian_command(
            p_desired,
            previous_p_command,
            self.last_servo_cartesian_velocity,
            dt,
            self.max_cartesian_velocity_mps,
            self.max_cartesian_acceleration_mps2)
        p_command = np.array(cartesian_command["position"], dtype=np.float64)
        cartesian_velocity = np.array(
            cartesian_command["velocity"], dtype=np.float64)
        self.last_workspace_margins = np.array(
            workspace_axis_margins(
                p_command, self.workspace_min, self.workspace_max),
            dtype=np.float64)
        if not is_target_within_workspace(
                p_command, self.workspace_min, self.workspace_max):
            self.reject_teleop_servo(
                "target_outside_workspace", p_desired, p_command)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return

        q_measured = self.current_joints.copy()
        fk_before = self.fk_xyz(q_measured)
        error_before = float(np.linalg.norm(p_command - fk_before))
        zero_delta_hold = (
            float(np.linalg.norm(self.last_scaled_delta)) <= 1e-9 and
            self.ee_anchor_xyz is not None and
            float(np.linalg.norm(p_command - self.ee_anchor_xyz)) <= 1e-9
        )
        if zero_delta_hold:
            if is_teleop_buffer_profile(self.control_profile):
                q_target = q_measured.copy()
                self.step_reference_toward_target(q_target, q_measured, dt)
                q_command = self.q_reference.copy()
                velocity_command = self.q_reference_velocity.copy()
                self.last_candidate_accepted = True
                self.last_reject_reason = "zero_delta_hold"
                self.last_q_current = q_measured.copy()
                self.last_q_candidate = q_command.copy()
                self.last_q_solution = q_target.copy()
                self.last_dq = q_command - q_measured
                self.last_fk_before = fk_before.copy()
                self.last_fk_after = self.fk_xyz(q_command)
                self.last_error_before = error_before
                self.last_error_norm = float(
                    np.linalg.norm(p_command - self.last_fk_after))
                self.last_servo_p_command = p_command.copy()
                self.last_servo_cartesian_velocity = cartesian_velocity.copy()
                self.last_servo_joint_velocity = velocity_command.copy()
                self.last_servo_time = now
                self.hold_sent_after_stop = False

                if self.publish_buffer_trajectory(
                        q_target, q_measured, now, "zero_delta_hold"):
                    if self.last_buffer_waypoints:
                        self.last_command_joints = np.array(
                            self.last_buffer_waypoints[-1],
                            dtype=np.float64)
                    else:
                        self.last_command_joints = q_command.copy()
                else:
                    self.log_teleop_servo_state(
                        p_desired, p_command, self.last_reject_reason)
                    self.publish_metrics(
                        float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
                    return
                self.log_teleop_servo_state(
                    p_desired, p_command, "zero_delta_hold")
                self.publish_joint_outputs(q_command)
                self.publish_metrics(
                    float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
                return

            q_command = q_measured.copy()
            velocity_command = np.zeros(len(JOINT_NAMES), dtype=np.float64)
            self.last_candidate_accepted = True
            self.last_reject_reason = "zero_delta_hold"
            self.last_q_current = q_measured.copy()
            self.last_q_candidate = q_command.copy()
            self.last_q_solution = q_command.copy()
            self.last_dq = np.zeros(len(JOINT_NAMES), dtype=np.float64)
            self.last_fk_before = fk_before.copy()
            self.last_fk_after = fk_before.copy()
            self.last_error_before = error_before
            self.last_error_norm = error_before
            self.last_servo_p_command = p_command.copy()
            self.last_servo_cartesian_velocity = cartesian_velocity.copy()
            self.last_servo_joint_velocity = velocity_command.copy()
            self.last_servo_time = now
            self.hold_sent_after_stop = False

            published = self.publish_servo_trajectory(
                q_command, q_measured, velocity_command)
            if published:
                self.last_command_joints = q_command.copy()
            self.log_teleop_servo_state(
                p_desired, p_command, "zero_delta_hold")
            self.publish_joint_outputs(q_command)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return

        q_ik, base_error_after = lm_ik(
            p_command, q_measured.copy(), fk_func=self.fk_xyz)
        fk_after = self.fk_xyz(q_ik)
        error_after = float(np.linalg.norm(p_command - fk_after))

        if base_error_after > self.servo_ik_tolerance_m:
            self.last_q_current = q_measured.copy()
            self.last_q_candidate = q_ik.copy()
            self.last_error_before = error_before
            self.last_error_norm = error_after
            self.reject_teleop_servo("ik_not_converged", p_desired, p_command)
            return
        if error_after > error_before + 1e-5:
            self.last_q_current = q_measured.copy()
            self.last_q_candidate = q_ik.copy()
            self.last_error_before = error_before
            self.last_error_norm = error_after
            self.reject_teleop_servo("ik_error_increased", p_desired, p_command)
            return
        if (np.any(q_ik < JOINT_LIMITS[:, 0] - 1e-9) or
                np.any(q_ik > JOINT_LIMITS[:, 1] + 1e-9)):
            self.last_q_current = q_measured.copy()
            self.last_q_candidate = q_ik.copy()
            self.reject_teleop_servo("joint_limit_violation", p_desired, p_command)
            return

        if is_teleop_buffer_profile(self.control_profile):
            q_target = q_ik.copy()
            self.step_reference_toward_target(q_target, q_measured, dt)
            q_command = self.q_reference.copy()
            self.last_candidate_accepted = True
            self.last_reject_reason = "accepted"
            self.last_q_current = q_measured.copy()
            self.last_q_candidate = q_command.copy()
            self.last_q_solution = q_target.copy()
            self.last_dq = q_command - q_measured
            self.last_fk_before = fk_before.copy()
            self.last_fk_after = self.fk_xyz(q_command)
            self.last_error_before = error_before
            self.last_error_norm = float(
                np.linalg.norm(p_command - self.last_fk_after))
            self.last_servo_p_command = p_command.copy()
            self.last_servo_cartesian_velocity = cartesian_velocity.copy()
            self.last_servo_time = now
            self.hold_sent_after_stop = False

            if self.publish_buffer_trajectory(
                    q_target, q_measured, now, "accepted"):
                if self.last_buffer_waypoints:
                    self.last_command_joints = np.array(
                        self.last_buffer_waypoints[-1], dtype=np.float64)
                else:
                    self.last_command_joints = q_target.copy()
            else:
                self.log_teleop_servo_state(
                    p_desired, p_command, self.last_reject_reason)
                self.publish_metrics(
                    float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
                return
            self.log_teleop_servo_state(p_desired, p_command, "accepted")
            self.publish_joint_outputs(q_command)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return

        joint_command = limit_joint_servo_command(
            q_ik,
            q_measured,
            self.last_servo_joint_velocity,
            self.servo_horizon_sec,
            dt,
            self.max_joint_velocity_radps,
            self.max_joint_acceleration_radps2)
        q_command = np.array(joint_command["position"], dtype=np.float64)
        velocity_command = np.array(joint_command["velocity"], dtype=np.float64)
        acceleration_command = (
            (velocity_command - self.last_servo_joint_velocity) / dt)
        if np.any(np.abs(velocity_command) > self.max_joint_velocity_radps + 1e-9):
            self.reject_teleop_servo("joint_velocity_limit_exceeded",
                                     p_desired, p_command)
            return
        if np.any(np.abs(acceleration_command) >
                  self.max_joint_acceleration_radps2 + 1e-9):
            self.reject_teleop_servo("joint_acceleration_limit_exceeded",
                                     p_desired, p_command)
            return

        self.last_candidate_accepted = True
        self.last_reject_reason = "accepted"
        self.last_q_current = q_measured.copy()
        self.last_q_candidate = q_command.copy()
        self.last_q_solution = q_command.copy()
        self.last_dq = q_command - q_measured
        self.last_fk_before = fk_before.copy()
        self.last_fk_after = self.fk_xyz(q_command)
        self.last_error_before = error_before
        self.last_error_norm = float(np.linalg.norm(p_command - self.last_fk_after))
        self.last_servo_p_command = p_command.copy()
        self.last_servo_cartesian_velocity = cartesian_velocity.copy()
        self.last_servo_joint_velocity = velocity_command.copy()
        self.last_servo_time = now
        self.hold_sent_after_stop = False

        if self.publish_servo_trajectory(
                q_command, q_measured, velocity_command):
            self.last_command_joints = q_command.copy()
        self.log_teleop_servo_state(p_desired, p_command, "accepted")
        self.publish_joint_outputs(q_command)
        self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)

    def control_loop_teleop_velocity_servo(self, now):
        workspace_policy = teleop_servo_workspace_policy(
            self.mode,
            self.target_mapping_mode,
            self.control_profile,
            self.enable_kinematics_target_adapter and
            self.kinematics_target_adapter_mode != "none")
        self.workspace_check_frame = workspace_policy[
            "workspace_check_frame"]
        self.last_adapter_applied = workspace_policy["adapter_applied"]
        if not self.last_adapter_applied:
            self.last_coordinate_transform_mode = (
                "real_relative_fk_frame_no_adapter")
        self.last_servo_latency_ms = (
            max(now - self.last_palm_time, 0.0) * 1000.0
            if self.last_palm_time is not None else
            float("inf")
        )
        palm_timed_out = (
            self.last_palm_time is None or
            (now - self.last_palm_time) > self.input_timeout_sec
        )
        joint_state_timed_out = not self.is_joint_state_fresh(now)
        deadman_active = self.is_deadman_active(now)
        stop_reason = servo_stop_reason(
            palm_timed_out, joint_state_timed_out, deadman_active)
        if stop_reason:
            if stop_reason in (
                    "palm_pose_timeout",
                    "deadman_inactive",
                    "joint_state_timeout"):
                self.publish_zero_velocity_once(stop_reason)
            self.last_candidate_accepted = False
            self.last_reject_reason = stop_reason
            self.log_velocity_servo_state(
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                self.current_joints.copy(),
                np.zeros(len(JOINT_NAMES), dtype=np.float64),
                stop_reason)
            self.publish_metrics(
                float(np.linalg.norm(self.last_palm_delta_raw)), 1.0)
            return

        if not self.is_real_relative_mode():
            self.last_candidate_accepted = False
            self.last_reject_reason = "teleop_velocity_requires_real_relative"
            self.log_velocity_servo_state(
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                self.current_joints.copy(),
                np.zeros(len(JOINT_NAMES), dtype=np.float64),
                self.last_reject_reason)
            return
        if not self.has_anchor:
            self.set_real_relative_anchor()
            self.last_velocity_servo_palm_xyz = self.palm_xyz.copy()
            self.last_q_current = self.current_joints.copy()
            self.last_q_candidate = self.current_joints.copy()
            self.log_velocity_servo_state(
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                self.current_joints.copy(),
                np.zeros(len(JOINT_NAMES), dtype=np.float64),
                "anchor_initialized")
            self.publish_metrics(0.0, 0.0)
            return
        if self.skip_next_command_after_anchor:
            self.skip_next_command_after_anchor = False
            self.last_velocity_servo_palm_xyz = self.palm_xyz.copy()
            self.last_q_current = self.current_joints.copy()
            self.last_q_candidate = self.current_joints.copy()
            self.log_velocity_servo_state(
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                self.current_joints.copy(),
                np.zeros(len(JOINT_NAMES), dtype=np.float64),
                "anchor_cycle_no_publish")
            self.publish_metrics(0.0, 0.0)
            return

        dt = (
            now - self.last_servo_time
            if self.last_servo_time is not None else
            1.0 / self.servo_rate_hz
        )
        dt = max(dt, 1e-3)
        step_data = self.compute_palm_step_velocity(dt)
        palm_step = np.array(step_data["palm_step"], dtype=np.float64)
        scaled_step = np.array(step_data["scaled_step"], dtype=np.float64)
        desired_cartesian_velocity = np.array(
            step_data["desired_cartesian_velocity"], dtype=np.float64)
        self.last_palm_delta_raw = palm_step.copy()
        self.last_palm_delta_robot = np.array(
            step_data["delta_robot"], dtype=np.float64)
        self.last_scaled_delta = scaled_step.copy()
        zero_deadzone_velocity = float(np.linalg.norm(palm_step)) <= 1e-12
        if zero_deadzone_velocity:
            cartesian_velocity = np.zeros(3, dtype=np.float64)
        else:
            cartesian_limit = limit_cartesian_velocity(
                desired_cartesian_velocity,
                self.last_servo_cartesian_velocity,
                dt,
                self.max_cartesian_velocity_mps,
                self.max_cartesian_acceleration_mps2)
            cartesian_velocity = np.array(
                cartesian_limit["velocity"], dtype=np.float64)

        q_measured = self.current_joints.copy()
        fk_before = self.fk_xyz(q_measured)
        p_next = fk_before + cartesian_velocity * dt
        self.last_workspace_margins = np.array(
            workspace_axis_margins(
                p_next, self.workspace_min, self.workspace_max),
            dtype=np.float64)
        if not is_target_within_workspace(
                p_next, self.workspace_min, self.workspace_max):
            self.last_candidate_accepted = False
            self.last_reject_reason = "velocity_target_outside_workspace"
            self.publish_zero_velocity_once(self.last_reject_reason)
            self.log_velocity_servo_state(
                palm_step,
                cartesian_velocity,
                q_measured,
                np.zeros(len(JOINT_NAMES), dtype=np.float64),
                self.last_reject_reason)
            self.publish_metrics(float(np.linalg.norm(palm_step)), 0.0)
            return

        if zero_deadzone_velocity:
            q_velocity = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        else:
            jac = numeric_jacobian(q_measured, fk_func=self.fk_xyz)
            q_velocity_raw = np.linalg.pinv(jac) @ cartesian_velocity
            joint_velocity = limit_joint_velocity_command(
                q_velocity_raw,
                self.last_servo_joint_velocity,
                dt,
                self.max_joint_velocity_radps,
                self.max_joint_acceleration_radps2)
            q_velocity = np.array(joint_velocity["velocity"], dtype=np.float64)
            q_velocity = self.clamp_velocity_at_joint_limits(
                q_measured, q_velocity, dt)

        self.last_candidate_accepted = True
        self.last_reject_reason = "accepted"
        self.last_q_current = q_measured.copy()
        self.last_q_candidate = q_measured.copy()
        self.last_q_solution = q_measured.copy()
        self.last_dq = q_velocity * dt
        self.last_fk_before = fk_before.copy()
        self.last_fk_after = self.fk_xyz(q_measured + self.last_dq)
        self.last_error_before = 0.0
        self.last_error_norm = 0.0
        self.last_servo_p_command = p_next.copy()
        self.last_servo_cartesian_velocity = cartesian_velocity.copy()
        self.last_servo_joint_velocity = q_velocity.copy()
        self.last_servo_time = now
        self.velocity_servo_zero_sent_after_stop = False
        self.hold_sent_after_stop = False

        if self.publish_velocity_servo_trajectory(q_velocity, "accepted"):
            self.last_command_joints = q_measured.copy()
        self.log_velocity_servo_state(
            palm_step, cartesian_velocity, q_measured, q_velocity, "accepted")
        self.publish_joint_outputs(q_measured)
        self.publish_metrics(float(np.linalg.norm(palm_step)), 0.0)

    def publish_trajectory(self, joints_rad):
        if self.mode not in ("gazebo", "real") or self.trajectory_pub is None:
            return
        if self.mode == "real" and not self.joint_state_received:
            self.get_logger().warn(
                "real command suppressed: arm_state_topic has not provided "
                "a complete JointState yet.",
                throttle_duration_sec=1.0)
            return
        if self.mode == "real" and self.real_safety_stop:
            self.get_logger().error(
                "real command suppressed: safety stop is active "
                f"reason={self.last_reject_reason}",
                throttle_duration_sec=1.0)
            return
        if self.mode == "real" and not self.last_candidate_accepted:
            self.get_logger().warn(
                "real command suppressed: last IK candidate was rejected "
                f"reason={self.last_reject_reason}",
                throttle_duration_sec=1.0)
            return
        if self.is_real_relative_mode() and self.anchor_error_norm > 0.005:
            self.get_logger().error(
                "real command suppressed: anchor error is too large "
                f"anchor_error_norm={self.anchor_error_norm:.6f}m",
                throttle_duration_sec=1.0)
            return
        if self.mode == "real" and self.dry_run:
            self.get_logger().info(
                "dry_run active: JointTrajectory not published "
                f"topic={self.arm_command_topic} "
                f"joints_rad={np.round(joints_rad, 5).tolist()} "
                f"accepted={self.last_candidate_accepted} "
                f"reason={self.last_reject_reason}",
                throttle_duration_sec=1.0)
            return

        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in joints_rad]
        point.time_from_start = duration_from_seconds(self.trajectory_time_from_start_sec)
        msg.points = [point]
        self.trajectory_pub.publish(msg)

    def publish_metrics(self, palm_delta_norm=0.0, timeout_flag=0.0):
        self.publish_float_array(self.metrics_pub, [
            self.last_error_norm,
            palm_delta_norm,
            self.last_clamped_flag,
            timeout_flag,
            1.0 if self.has_anchor else 0.0,
            1.0 if self.has_control_center else 0.0,
            self.last_reach_radius,
            self.last_effective_reach_radius,
            self.last_reach_radius_clamped_flag,
            1.0 if self.target_mapping_mode == "absolute" else 0.0,
        ])

    def control_loop(self):
        now = seconds_now(self)
        if is_teleop_buffer_profile(self.control_profile):
            self.control_loop_teleop_buffer_publish(now)
            return
        if is_teleop_forward_position_profile(self.control_profile):
            self.control_loop_teleop_forward_position(now)
            return
        if is_teleop_trajectory_profile(self.control_profile):
            self.control_loop_teleop_servo(now)
            return
        if is_teleop_velocity_profile(self.control_profile):
            self.control_loop_teleop_velocity_servo(now)
            return

        debug_without_palm = (
            abs(self.debug_target_step_x) > 1e-9 and
            self.joint_state_received and
            self.last_palm_time is None
        )

        if self.mode == "real" and not self.joint_state_received:
            self.get_logger().warn(
                "real command inactive: waiting for complete current joint state "
                f"from {self.arm_state_topic}.",
                throttle_duration_sec=1.0)
            self.publish_metrics(0.0, 0.0)
            return

        if self.last_palm_time is None and not debug_without_palm:
            self.publish_metrics(0.0, 0.0)
            return

        target_age = 0.0 if debug_without_palm else now - self.last_palm_time
        input_timeout = self.input_timeout_sec if self.use_control_center else self.target_timeout_sec
        if not debug_without_palm and target_age > input_timeout:
            self.last_timeout_flag = 1.0
            self.get_logger().warn(
                "palm_pose timeout; holding command output "
                f"age={target_age:.3f}s input_timeout_sec={input_timeout:.3f}s",
                throttle_duration_sec=1.0)
            self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 1.0)
            return
        self.last_timeout_flag = 0.0

        if (self.target_mapping_mode != "absolute" and
                not self.is_real_relative_mode() and
                self.use_control_center and self.require_control_center and
                not self.has_control_center):
            self.get_logger().warn(
                "control center not received; arm command inactive",
                throttle_duration_sec=1.0)
            self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return

        if debug_without_palm:
            target_xyz = self.make_debug_target_from_joint_state()
        elif self.is_real_relative_mode():
            if not self.has_anchor:
                if self.palm_xyz is None:
                    self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
                    return
                self.set_real_relative_anchor()
            target_xyz = self.make_target()
        elif (self.target_mapping_mode != "absolute" and
              self.use_control_center and not self.has_anchor):
            if self.palm_xyz is not None and (self.has_control_center or not self.require_control_center):
                self.set_ee_anchor_from_current_pose()
            else:
                self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
                return
            target_xyz = self.make_target()
        elif (self.target_mapping_mode != "absolute" and
              not self.use_control_center and
              (self.palm_anchor_xyz is None or self.ee_anchor_xyz is None)):
            self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
            return
        else:
            target_xyz = self.make_target()

        target_xyz_control = target_xyz.copy()
        target_xyz_kinematics = self.adapt_target_for_kinematics(target_xyz_control)
        self.last_target_control_xyz = target_xyz_control.copy()
        self.last_target_kinematics_xyz = target_xyz_kinematics.copy()

        q_current = (
            self.current_joints.copy()
            if self.joint_state_received else
            self.last_command_joints.copy()
        )
        fk_before = self.fk_xyz(q_current)
        if self.palm_target_mode == "cylindrical_absolute" and not debug_without_palm:
            self.last_target_delta = target_xyz_control - fk_before
        error_before = float(np.linalg.norm(target_xyz_kinematics - fk_before))
        zero_real_relative_delta = (
            self.is_real_relative_mode() and
            float(np.linalg.norm(self.last_scaled_delta)) <= 1e-9
        )
        if zero_real_relative_delta:
            joints_rad = q_current.copy()
            base_error_after = error_before
        else:
            seed = q_current.copy() if self.is_real_relative_mode() else self.last_command_joints.copy()
            joints_rad, base_error_after = lm_ik(
                target_xyz_kinematics, seed, fk_func=self.fk_xyz)
            joints_rad = self.apply_posture_biases(
                joints_rad, target_xyz_kinematics, base_error_after, q_current, fk_before)
        if self.mode == "real":
            joints_rad = np.array(
                clamp_joint_step(joints_rad, q_current, self.max_joint_step_rad),
                dtype=np.float64)
            joints_rad = clamp_array(joints_rad, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        fk_after = self.fk_xyz(joints_rad)
        error_norm = float(np.linalg.norm(target_xyz_kinematics - fk_after))
        if base_error_after > 0.03:
            self.get_logger().warn(
                "IK may not have converged "
                f"base_error_after={base_error_after:.5f} "
                f"target_control={np.round(target_xyz_control, 4).tolist()} "
                f"target_kinematics={np.round(target_xyz_kinematics, 4).tolist()}",
                throttle_duration_sec=1.0)
        dq = joints_rad - q_current

        self.last_candidate_accepted = True
        self.last_reject_reason = "accepted"
        if self.is_real_relative_mode():
            if self.real_safety_stop or not self.real_anchor_valid:
                self.last_candidate_accepted = False
                if self.last_reject_reason == "accepted":
                    self.last_reject_reason = "real_anchor_invalid"
            elif self.anchor_error_norm > 0.005:
                self.last_candidate_accepted = False
                self.last_reject_reason = "anchor_error_norm_too_large"
            elif error_norm > error_before + 1e-5:
                self.last_candidate_accepted = False
                self.last_reject_reason = "ik_error_increased"
            elif (zero_real_relative_delta and
                  float(np.linalg.norm(joints_rad - q_current)) > 1e-6):
                self.last_candidate_accepted = False
                self.last_reject_reason = "zero_delta_changed_q"

        self.last_error_norm = error_norm
        self.last_q_current = q_current.copy()
        self.last_q_solution = joints_rad.copy()
        self.last_q_candidate = joints_rad.copy()
        self.last_dq = dq.copy()
        self.last_fk_before = fk_before.copy()
        self.last_fk_after = fk_after.copy()
        self.last_error_before = error_before

        self.target_pub.publish(self.make_point_stamped(target_xyz_control))
        self.publish_debug_deltas()
        self.publish_ik_debug()
        self.publish_joint_outputs(joints_rad)
        self.publish_metrics(float(np.linalg.norm(self.last_palm_delta_raw)), 0.0)
        self.log_real_relative_safety_state(target_xyz_control)
        if self.mode == "real" and not self.last_candidate_accepted:
            self.get_logger().warn(
                "real command candidate rejected; publish skipped "
                f"reason={self.last_reject_reason}",
                throttle_duration_sec=1.0)
            self.log_debug_state(now, target_xyz_control)
            return

        self.last_command_joints = joints_rad.copy()
        if not self.joint_state_received:
            self.current_joints = joints_rad.copy()
        if self.mode == "real" and self.skip_next_command_after_anchor:
            self.skip_next_command_after_anchor = False
            self.get_logger().info(
                "first palm_pose used only as real-mode anchor; "
                "robot command suppressed for this cycle.")
            self.log_debug_state(now, target_xyz_control)
            return
        self.publish_trajectory(joints_rad)
        self.log_debug_state(now, target_xyz_control)


def main(args=None):
    rclpy.init(args=args)
    node = AmirPalmIkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
