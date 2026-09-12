"""Semantic-only publisher and explicitly synthetic executable; no raw subscribers."""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor

from amir_irt_mro.local_hold_state_estimator import LocalHoldStateEstimator
from amir_irt_mro.local_hold_mock import mock_inputs
from share_semantic_interfaces.msg import LocalHoldState as RosLocalHoldState
from .mapper import SemanticStream


class LocalHoldSemanticPublisher(Node):
    """Inject normalized input via publish_inputs; no hardware binding is implied."""

    def __init__(self, hold_config=None, **kwargs):
        super().__init__('local_hold_semantic_publisher', **kwargs)
        self.declare_parameter('participant_id', 'Unassigned',
                               ParameterDescriptor(read_only=True))
        self.stream = SemanticStream(self.get_parameter('participant_id').value)
        self.estimator = LocalHoldStateEstimator(hold_config)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.publisher = self.create_publisher(RosLocalHoldState, 'local_hold_state', qos)

    def publish_inputs(self, inputs):
        """Publish exactly the Core result; caller supplies its own source clock."""
        message = self.stream.encode(self.estimator.update(inputs))
        self.publisher.publish(message)
        return message


class MockHoldSemanticPublisher(LocalHoldSemanticPublisher):
    """Synthetic open / hold / commit / release cycle for transport verification."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started_ns = time.monotonic_ns()
        self.timer = self.create_timer(0.02, self.tick)

    def tick(self):
        """Use source-local monotonic time, not a synchronized-PC clock assumption."""
        now = time.monotonic_ns()
        phase = (now - self.started_ns) % 3_000_000_000
        self.publish_inputs(mock_inputs(now, close=500_000_000 <= phase < 2_000_000_000))


def main(args=None):
    """Run the explicitly named mock executable; never starts a controller."""
    rclpy.init(args=args)
    node = None
    try:
        node = MockHoldSemanticPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
