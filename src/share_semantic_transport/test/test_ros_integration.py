"""Local DDS integration: semantic output only, no raw hardware publishing."""

import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from amir_irt_mro.local_hold_mock import mock_inputs
from share_semantic_interfaces.msg import LocalHoldState
from share_semantic_transport.node import LocalHoldSemanticPublisher


def test_ros_two_pairs_and_remap():
    context = Context()
    rclpy.init(context=context)
    executor = SingleThreadedExecutor(context=context)
    nodes, received = [], {'User1': [], 'User2': []}
    try:
        observer = Node('semantic_test_observer', context=context)
        nodes.append(observer)
        for identity in received:
            node = LocalHoldSemanticPublisher(
                context=context, namespace='semantic_test/' + identity,
                parameter_overrides=[Parameter('participant_id', value=identity)],
                cli_args=['--ros-args', '-r', 'local_hold_state:=state_under_test'])
            nodes.append(node)
            observer.create_subscription(
                LocalHoldState, f'/semantic_test/{identity}/state_under_test',
                lambda msg, key=identity: received[key].append(msg), 10)
        for node in nodes:
            executor.add_node(node)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
            if all(n.publisher.get_subscription_count() == 1 for n in nodes[1:]):
                break
        assert all(n.publisher.get_subscription_count() == 1 for n in nodes[1:])
        for ns, close in [(0, False), (10_000_000, True), (210_000_000, True),
                          (610_000_000, True), (620_000_000, False), (720_000_000, False)]:
            for node in nodes[1:]:
                node.publish_inputs(mock_inputs(ns, close=close))
            deadline = time.monotonic() + 2
            expected = nodes[1].stream.sequence
            while time.monotonic() < deadline and any(
                    not values or values[-1].sequence != expected for values in received.values()):
                executor.spin_once(timeout_sec=0.02)
            assert all(values and values[-1].sequence == expected for values in received.values())
        for identity, values in received.items():
            assert all(m.participant_id == identity for m in values)
            assert [m.sequence for m in values] == list(range(1, 7))
            assert values[2].holds_object and values[3].held_object_valid
            assert not values[-1].holds_object and not values[-1].held_object_valid
        assert received['User1'][0].session_id != received['User2'][0].session_id
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        context.shutdown()
