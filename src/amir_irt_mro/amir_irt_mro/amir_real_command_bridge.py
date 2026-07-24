#!/usr/bin/env python3
"""Bridge JointTrajectory arm commands to the low-level AmirCmd topic."""

# Disabled by default; use only after confirming /motor_sub live type.

import math

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

from amir_interfaces.msg import AmirCmd, AmirSensor

from amir_irt_mro.real_safety import (
    DEFAULT_ARM_COMMAND_TOPIC,
    DEFAULT_ARM_JOINT_NAMES,
    clamp_joint_step,
    motor_mrad_to_ros2_control_rad,
    mrad_to_rad,
    normalize_command_unit,
    rad_to_mrad,
    ros2_control_rad_to_motor_mrad,
    validate_joint_names,
)


MOTOR_COMMAND_TYPE = "amir_interfaces/msg/AmirCmd"


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class AmirRealCommandBridge(Node):
    """Convert rad JointTrajectory commands to mrad AmirCmd commands."""

    def __init__(self):
        super().__init__("amir_real_command_bridge")

        self.declare_parameter(
            "input_trajectory_topic", DEFAULT_ARM_COMMAND_TOPIC)
        self.declare_parameter("motor_command_topic", "/motor_sub")
        self.declare_parameter("motor_state_topic", "/encoder_pub")
        self.declare_parameter("confirmed_motor_command_type", "")
        self.declare_parameter("command_unit", "rad")
        self.declare_parameter("joint_names", list(DEFAULT_ARM_JOINT_NAMES))
        self.declare_parameter("max_joint_step_rad", 0.01)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("require_state", True)

        self.input_trajectory_topic = str(
            self.get_parameter("input_trajectory_topic").value).strip()
        self.motor_command_topic = str(
            self.get_parameter("motor_command_topic").value).strip()
        self.motor_state_topic = str(
            self.get_parameter("motor_state_topic").value).strip()
        self.confirmed_motor_command_type = str(
            self.get_parameter("confirmed_motor_command_type").value).strip()
        try:
            self.command_unit = normalize_command_unit(
                self.get_parameter("command_unit").value)
        except ValueError as exc:
            self.get_logger().warn(f"{exc} Falling back to rad.")
            self.command_unit = "rad"
        self.joint_names = [
            str(name) for name in self.get_parameter("joint_names").value]
        validate_joint_names(self.joint_names, DEFAULT_ARM_JOINT_NAMES)
        self.max_joint_step_rad = max(
            float(self.get_parameter("max_joint_step_rad").value), 0.0)
        if self.max_joint_step_rad > 0.01:
            self.get_logger().warn(
                "max_joint_step_rad is capped at 0.01rad for "
                "direct motor safety "
                f"(requested={self.max_joint_step_rad:.5f}rad).")
            self.max_joint_step_rad = 0.01
        self.dry_run = as_bool(self.get_parameter("dry_run").value)
        self.require_state = as_bool(self.get_parameter("require_state").value)

        self.latest_state_rad = None
        self.latest_raw_motor_angle_mrad = None

        self.create_subscription(
            JointTrajectory,
            self.input_trajectory_topic,
            self.trajectory_cb,
            10,
        )
        self.create_subscription(
            AmirSensor,
            self.motor_state_topic,
            self.motor_state_cb,
            10,
        )
        self.motor_pub = self.create_publisher(
            AmirCmd, self.motor_command_topic, 10)

        self.get_logger().warn(
            "AmirRealCommandBridge started. It will not publish unless "
            "dry_run:=false and confirmed_motor_command_type:="
            f"{MOTOR_COMMAND_TYPE}. "
            f"input={self.input_trajectory_topic} "
            f"output={self.motor_command_topic} "
            f"state={self.motor_state_topic} command_unit={self.command_unit} "
            f"dry_run={self.dry_run} require_state={self.require_state}")

    def motor_state_cb(self, msg):
        self.latest_raw_motor_angle_mrad = [
            float(value) for value in msg.angle]
        try:
            self.latest_state_rad = motor_mrad_to_ros2_control_rad(
                self.latest_raw_motor_angle_mrad[:len(self.joint_names)])
        except ValueError as exc:
            self.latest_state_rad = None
            self.get_logger().error(
                f"Invalid {self.motor_state_topic} state: {exc}",
                throttle_duration_sec=1.0)

    def trajectory_cb(self, msg):
        try:
            validate_joint_names(msg.joint_names, self.joint_names)
        except ValueError as exc:
            self.get_logger().error(
                f"JointTrajectory rejected: {exc}",
                throttle_duration_sec=1.0)
            return

        if not msg.points:
            self.get_logger().warn("JointTrajectory rejected: no points.")
            return
        if self.require_state and self.latest_state_rad is None:
            self.get_logger().warn(
                "JointTrajectory rejected: waiting for encoder state from "
                f"{self.motor_state_topic}.",
                throttle_duration_sec=1.0)
            return

        point = msg.points[0]
        if len(point.positions) != len(self.joint_names):
            self.get_logger().error(
                "JointTrajectory rejected: position count does not match "
                "joint_names "
                f"({len(point.positions)} != {len(self.joint_names)}).",
                throttle_duration_sec=1.0)
            return

        if self.command_unit == "mrad":
            target_rad = mrad_to_rad(point.positions)
        else:
            target_rad = [float(value) for value in point.positions]

        if self.latest_state_rad is not None:
            target_rad = clamp_joint_step(
                target_rad, self.latest_state_rad, self.max_joint_step_rad)

        try:
            motor_angle_mrad = ros2_control_rad_to_motor_mrad(target_rad)
        except ValueError as exc:
            self.get_logger().error(f"Motor command conversion failed: {exc}")
            return

        angles = [0.0] * 6
        velocities = [0.0] * 6
        for idx, value in enumerate(motor_angle_mrad[:len(self.joint_names)]):
            angles[idx] = float(value)

        if (self.latest_raw_motor_angle_mrad and
                len(self.latest_raw_motor_angle_mrad) >= 6):
            angles[5] = float(self.latest_raw_motor_angle_mrad[5])

        if len(point.velocities) >= len(self.joint_names):
            if self.command_unit == "mrad":
                motor_velocity_mrad = [
                    float(value) for value in point.velocities]
            else:
                motor_velocity_mrad = rad_to_mrad(
                    point.velocities[:len(self.joint_names)])
            for idx, value in enumerate(
                    motor_velocity_mrad[:len(self.joint_names)]):
                velocities[idx] = float(value)
        else:
            for idx in range(len(self.joint_names)):
                if (self.latest_raw_motor_angle_mrad and
                        idx < len(self.latest_raw_motor_angle_mrad)):
                    delta_mrad = (
                        angles[idx] -
                        float(self.latest_raw_motor_angle_mrad[idx]))
                else:
                    delta_mrad = 0.0
                velocities[idx] = (
                    math.copysign(10.0, delta_mrad)
                    if abs(delta_mrad) > 1e-6 else
                    0.0
                )

        if self.dry_run:
            self.get_logger().info(
                "dry_run active: AmirCmd not published "
                f"angle_mrad={[round(value, 3) for value in angles]} "
                f"vel_mrad_s={[round(value, 3) for value in velocities]}",
                throttle_duration_sec=1.0)
            return

        if self.confirmed_motor_command_type != MOTOR_COMMAND_TYPE:
            self.get_logger().error(
                "AmirCmd publish blocked: confirm live topic type first with "
                f"`ros2 topic type {self.motor_command_topic}` and set "
                f"confirmed_motor_command_type:={MOTOR_COMMAND_TYPE}.",
                throttle_duration_sec=1.0)
            return

        out = AmirCmd()
        out.angle = angles
        out.vel = velocities
        self.motor_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AmirRealCommandBridge()
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
