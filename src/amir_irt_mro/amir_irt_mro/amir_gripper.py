#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand

TORQUE_THRESHOLD = 50.0
ACTION_GRIPPER = '/gripper_controller/gripper_cmd'


class AmirGripperNode(Node):
    def __init__(self):
        super().__init__('amir_gripper')
        self.declare_parameter('open_position', -1.0)
        self.declare_parameter('close_position', 0.06)
        self.declare_parameter('max_effort', 10.0)
        self.declare_parameter('grasp_command_topic', '/grasp_command')
        self.declare_parameter('gripper_state_topic', '/gripper_state')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('gripper_action_name', ACTION_GRIPPER)

        self.open_position = float(self.get_parameter('open_position').value)
        self.close_position = float(self.get_parameter('close_position').value)
        self.max_effort = float(self.get_parameter('max_effort').value)
        self.grasp_command_topic = str(
            self.get_parameter('grasp_command_topic').value).strip()
        self.gripper_state_topic = str(
            self.get_parameter('gripper_state_topic').value).strip()
        self.joint_state_topic = str(
            self.get_parameter('joint_state_topic').value).strip()
        self.gripper_action_name = str(
            self.get_parameter('gripper_action_name').value).strip()

        self.is_open = False
        self.last_command = None
        self.last_command_time = None
        self.last_state = None
        self._action_client = ActionClient(
            self, GripperCommand, self.gripper_action_name)
        self.state_pub = self.create_publisher(Bool, self.gripper_state_topic, 10)
        self.create_subscription(
            String, self.grasp_command_topic, self.grasp_cb, 10)
        self.create_subscription(
            Bool, self.gripper_state_topic, self.gripper_state_cb, 10)
        self.create_subscription(
            JointState, self.joint_state_topic, self.joint_state_cb, 10)
        self.get_logger().info(
            'AmirGripperNode started '
            f'grasp_command_topic={self.grasp_command_topic} '
            f'gripper_state_topic={self.gripper_state_topic} '
            f'joint_state_topic={self.joint_state_topic} '
            f'gripper_action_name={self.gripper_action_name}')

    def send_gripper_goal(self, position):
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = self.max_effort
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'Gripper action server unavailable: {self.gripper_action_name}')
            return False
        self._action_client.send_goal_async(goal)
        return True

    def grasp_cb(self, msg):
        cmd = msg.data.lower().strip()
        self.get_logger().info(f'Received: {cmd}')
        if cmd == 'open':
            if self.send_gripper_goal(self.open_position):
                self.is_open = True
                self.last_command = cmd
                self.last_command_time = self.get_clock().now().nanoseconds * 1e-9
        elif cmd == 'close':
            if self.send_gripper_goal(self.close_position):
                self.is_open = False
                self.last_command = cmd
                self.last_command_time = self.get_clock().now().nanoseconds * 1e-9
        else:
            self.get_logger().warn(f'Invalid grasp command: {msg.data}')

    def gripper_state_cb(self, msg):
        self.last_state = bool(msg.data)
        if self.last_command_time is None:
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - self.last_command_time
        if age <= 5.0:
            self.get_logger().info(
                f'Gripper state after command={self.last_command}: '
                f'grasped={self.last_state}',
                throttle_duration_sec=1.0)

    def joint_state_cb(self, msg):
        if 'Gripper' not in msg.name:
            return
        idx = msg.name.index('Gripper')
        if idx < len(msg.effort):
            grasped = abs(msg.effort[idx]) > TORQUE_THRESHOLD
            self.state_pub.publish(Bool(data=grasped))


def main(args=None):
    rclpy.init(args=args)
    node = AmirGripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
