#!/usr/bin/env python3
"""
初期姿勢設定ノード（ROS2 Humble / Amir対応）
元：initialposition_node.py（ROS1）をROS2に移植
機能：実験開始時にAmirを初期姿勢に移動
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Bool, String

from amir_irt_mro.real_safety import (
    DEFAULT_ARM_INITIAL_POSITIONS,
    DEFAULT_ARM_JOINT_NAMES,
    validate_joint_positions,
)

# Amirの初期関節角度（rad）
JOINT_NAMES = list(DEFAULT_ARM_JOINT_NAMES)


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


class AmirInitialPositionNode(Node):
    def __init__(self):
        super().__init__('amir_initial_position')
        self.declare_parameter('auto_send', False)
        self.declare_parameter('deadman_required', True)
        self.declare_parameter('deadman_topic', '/deadman_enabled')

        self.auto_send = as_bool(self.get_parameter('auto_send').value)
        self.deadman_required = as_bool(self.get_parameter('deadman_required').value)
        self.deadman_enabled = False
        self.deadman_topic = str(self.get_parameter('deadman_topic').value)

        # Publisher
        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10)

        # Subscriber（外部からトリガーを受け取る）
        self.create_subscription(
            String,
            '/init_command',
            self.init_cb, 10)
        self.create_subscription(
            Bool,
            self.deadman_topic,
            self.deadman_cb, 10)

        # 起動時に一度だけ初期姿勢へ移動
        self.create_timer(2.0, self.send_initial_once)
        self.initialized = False

        self.get_logger().info('AmirInitialPositionNode started')

    def deadman_cb(self, msg):
        self.deadman_enabled = bool(msg.data)

    def send_initial_once(self):
        if self.auto_send and not self.initialized:
            self.send_initial_position()
            self.initialized = True

    def init_cb(self, msg):
        if msg.data == 'init':
            self.get_logger().info('Received init command')
            self.send_initial_position()

    def send_initial_position(self):
        # 変更根拠: 初期姿勢指令も実機では動作指令なので Deadman で保護する。
        # 期待される効果: launch 直後に意図せずアームが動き始めるリスクを下げる。
        if self.deadman_required and not self.deadman_enabled:
            self.get_logger().warn('Deadman is OFF. Initial position command is not sent.')
            return

        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = validate_joint_positions(
            DEFAULT_ARM_INITIAL_POSITIONS, JOINT_NAMES)
        point.time_from_start = Duration(sec=2, nanosec=0)
        msg.points = [point]

        self.arm_pub.publish(msg)
        self.get_logger().info(
            "Initial position sent: "
            f"{[f'{p:.2f}' for p in DEFAULT_ARM_INITIAL_POSITIONS]}")


def main(args=None):
    rclpy.init(args=args)
    node = AmirInitialPositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
