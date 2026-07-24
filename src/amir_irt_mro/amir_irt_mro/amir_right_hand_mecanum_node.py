#!/usr/bin/env python3
"""Right-hand mecanum rover controller for AMIR."""

import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float32MultiArray


@dataclass(frozen=True)
class MecanumInput:
    active: bool
    active_value: float
    delta_x: float
    delta_z: float
    roll_delta: float


def parse_mecanum_input(data):
    """Parse [active, dx, dz, roll] with an optional fifth timestamp field."""
    if len(data) < 4:
        raise ValueError(
            f'Input data length is {len(data)}; expected at least 4 '
            'fields: [active, dx, dz, roll].'
        )

    try:
        active_value, delta_x, delta_z, roll_delta = [
            float(value) for value in data[:4]
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Input contains a non-numeric value: {exc}') from exc

    values = [active_value, delta_x, delta_z, roll_delta]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Input contains NaN or inf.')

    return MecanumInput(
        active=active_value >= 0.5,
        active_value=active_value,
        delta_x=delta_x,
        delta_z=delta_z,
        roll_delta=roll_delta,
    )


class AmirRightHandMecanumNode(Node):
    def __init__(self):
        super().__init__('amir_right_hand_mecanum_node')

        self.declare_parameter('input_topic', '/amir/right_hand_mecanum_input')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('timeout_sec', 0.3)
        self.declare_parameter('gain_x', 0.4)
        self.declare_parameter('gain_y', 0.4)
        self.declare_parameter('gain_yaw', 0.5)
        self.declare_parameter('deadzone_pos_m', 0.02)
        self.declare_parameter('deadzone_roll_rad', 0.0872665)
        self.declare_parameter('max_linear_x', 0.03)
        self.declare_parameter('max_linear_y', 0.03)
        self.declare_parameter('max_angular_z', 0.15)
        self.declare_parameter('invert_x', False)
        self.declare_parameter('invert_y', False)
        self.declare_parameter('invert_yaw', False)
        self.declare_parameter('diagnostic_log_period_sec', 1.0)

        self.input_topic = self.get_parameter('input_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.control_rate_hz = float(
            self.get_parameter('control_rate_hz').value)
        if self.control_rate_hz <= 0.0:
            self.get_logger().warn(
                'control_rate_hz must be > 0. Using 30.0 Hz.')
            self.control_rate_hz = 30.0

        self.timeout_sec = float(self.get_parameter('timeout_sec').value)
        self.gain_x = float(self.get_parameter('gain_x').value)
        self.gain_y = float(self.get_parameter('gain_y').value)
        self.gain_yaw = float(self.get_parameter('gain_yaw').value)
        self.deadzone_pos_m = abs(float(
            self.get_parameter('deadzone_pos_m').value))
        self.deadzone_roll_rad = abs(float(
            self.get_parameter('deadzone_roll_rad').value))
        self.max_linear_x = abs(float(
            self.get_parameter('max_linear_x').value))
        self.max_linear_y = abs(float(
            self.get_parameter('max_linear_y').value))
        self.max_angular_z = abs(float(
            self.get_parameter('max_angular_z').value))
        self.invert_x = bool(self.get_parameter('invert_x').value)
        self.invert_y = bool(self.get_parameter('invert_y').value)
        self.invert_yaw = bool(self.get_parameter('invert_yaw').value)
        self.diagnostic_log_period_sec = float(
            self.get_parameter('diagnostic_log_period_sec').value)
        if self.diagnostic_log_period_sec <= 0.0:
            self.get_logger().warn(
                'diagnostic_log_period_sec must be > 0. Using 1.0 sec.')
            self.diagnostic_log_period_sec = 1.0

        self.active = False
        self.input_valid = False
        self.active_value = 0.0
        self.delta_x = 0.0
        self.delta_z = 0.0
        self.roll_delta = 0.0
        self.last_input_time = None
        self.last_msg_time = None
        self.last_log_time = self.get_clock().now()
        self.last_cmd = Twist()
        self.last_deadband = False
        self.last_zero_reason = 'no_input'
        self.last_invalid_reason = 'no_input'
        self.input_count = 0
        self.valid_input_count = 0
        self.output_count = 0
        self.diag_input_count = 0
        self.diag_output_count = 0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.input_sub = self.create_subscription(
            Float32MultiArray,
            self.input_topic,
            self.input_callback,
            10,
        )
        self.timer = self.create_timer(
            1.0 / self.control_rate_hz, self.control_loop)

        self.get_logger().info(
            f'AmirRightHandMecanumNode started. '
            f'Listening on {self.input_topic}, '
            f'publishing {self.cmd_vel_topic}. '
            'Expected input: Float32MultiArray [active, dx, dz, roll] '
            'with optional timestamp as field 5.'
        )

    def input_callback(self, msg):
        now = self.get_clock().now()
        self.last_input_time = now
        self.input_count += 1
        self.diag_input_count += 1

        try:
            parsed = parse_mecanum_input(msg.data)
        except ValueError as exc:
            self.active = False
            self.input_valid = False
            self.active_value = 0.0
            self.delta_x = 0.0
            self.delta_z = 0.0
            self.roll_delta = 0.0
            self.last_deadband = False
            self.last_invalid_reason = str(exc)
            self.get_logger().warn(
                f'{exc} Publishing zero command.',
                throttle_duration_sec=1.0,
            )
            return

        self.last_msg_time = now
        self.active = parsed.active
        self.input_valid = True
        self.active_value = parsed.active_value
        self.delta_x = parsed.delta_x
        self.delta_z = parsed.delta_z
        self.roll_delta = parsed.roll_delta
        self.last_invalid_reason = None
        self.valid_input_count += 1

    def control_loop(self):
        timeout = self.is_timeout()
        if not self.input_valid:
            cmd = Twist()
            self.last_deadband = False
            self.last_zero_reason = self.last_invalid_reason or 'invalid_input'
        elif timeout:
            cmd = Twist()
            self.last_deadband = False
            self.last_zero_reason = 'timeout'
        elif not self.active:
            cmd = Twist()
            self.last_deadband = False
            self.last_zero_reason = 'inactive'
        else:
            cmd = self.make_cmd_vel()
            if self.is_zero_cmd(cmd):
                self.last_zero_reason = (
                    'deadband' if self.last_deadband else 'zero_input')
            else:
                self.last_zero_reason = None

        self.last_cmd = cmd
        self.cmd_pub.publish(cmd)
        self.output_count += 1
        self.diag_output_count += 1
        self.log_status(timeout)

    def make_cmd_vel(self):
        delta_z = self.apply_deadzone(self.delta_z, self.deadzone_pos_m)
        delta_x = self.apply_deadzone(self.delta_x, self.deadzone_pos_m)
        roll_delta = self.apply_deadzone(
            self.roll_delta, self.deadzone_roll_rad)
        self.last_deadband = (
            self.is_deadband_suppressed(self.delta_z, self.deadzone_pos_m) or
            self.is_deadband_suppressed(self.delta_x, self.deadzone_pos_m) or
            self.is_deadband_suppressed(
                self.roll_delta, self.deadzone_roll_rad)
        )

        linear_x = self.gain_x * delta_z
        linear_y = self.gain_y * delta_x
        angular_z = self.gain_yaw * roll_delta

        if self.invert_x:
            linear_x *= -1.0
        if self.invert_y:
            linear_y *= -1.0
        if self.invert_yaw:
            angular_z *= -1.0

        cmd = Twist()
        cmd.linear.x = self.clamp(linear_x, self.max_linear_x)
        cmd.linear.y = self.clamp(linear_y, self.max_linear_y)
        cmd.angular.z = self.clamp(angular_z, self.max_angular_z)
        return cmd

    def is_timeout(self):
        if self.last_msg_time is None:
            return True
        elapsed_sec = (
            self.get_clock().now() - self.last_msg_time
        ).nanoseconds * 1e-9
        return elapsed_sec >= self.timeout_sec

    def publish_zero(self):
        self.cmd_pub.publish(Twist())

    def log_status(self, timeout):
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds * 1e-9
        if elapsed_sec < self.diagnostic_log_period_sec:
            return
        self.last_log_time = now
        in_hz = self.diag_input_count / elapsed_sec
        out_hz = self.diag_output_count / elapsed_sec
        self.diag_input_count = 0
        self.diag_output_count = 0
        input_age = self.input_age_sec(now)
        if input_age is None:
            age_text = 'None'
        else:
            age_text = f'{input_age:.2f}'
        zero_reason = (
            self.last_zero_reason if self.last_zero_reason is not None
            else 'None'
        )
        self.get_logger().info(
            f'[MecanumDiag] in_hz={in_hz:.1f} out_hz={out_hz:.1f} '
            f'active={self.active} age={age_text} '
            f'raw=({self.delta_x:.4f},{self.delta_z:.4f},'
            f'{self.roll_delta:.4f}) '
            f'twist=({self.last_cmd.linear.x:.4f},'
            f'{self.last_cmd.linear.y:.4f},'
            f'{self.last_cmd.angular.z:.4f}) '
            f'timeout={timeout} deadband={self.last_deadband} '
            f'zero_reason={zero_reason} '
            f'publish_count={self.output_count}'
        )

    def input_age_sec(self, now):
        if self.last_input_time is None:
            return None
        return (now - self.last_input_time).nanoseconds * 1e-9

    @staticmethod
    def apply_deadzone(value, deadzone):
        if abs(value) < deadzone:
            return 0.0
        return value

    @staticmethod
    def is_deadband_suppressed(value, deadzone):
        return value != 0.0 and abs(value) < deadzone

    @staticmethod
    def is_zero_cmd(cmd):
        return (
            cmd.linear.x == 0.0 and
            cmd.linear.y == 0.0 and
            cmd.angular.z == 0.0
        )

    @staticmethod
    def clamp(value, max_abs):
        return max(-max_abs, min(max_abs, value))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = AmirRightHandMecanumNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
