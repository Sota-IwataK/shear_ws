import pytest
import numpy as np

from amir_irt_mro.amir_palm_ik_node import AmirPalmIkNode
from amir_irt_mro.real_safety import (
    DEFAULT_ARM_COMMAND_TOPIC,
    DEFAULT_ARM_INITIAL_POSITIONS,
    DEFAULT_ARM_JOINT_NAMES,
    DEFAULT_FORWARD_POSITION_COMMAND_TOPIC,
    clamp_joint_step,
    compute_relative_target,
    is_teleop_buffer_profile,
    is_teleop_forward_position_profile,
    is_teleop_trajectory_profile,
    is_teleop_velocity_profile,
    is_target_within_workspace,
    integrate_joint_reference,
    limit_cartesian_command,
    limit_cartesian_velocity,
    limit_joint_servo_command,
    limit_joint_velocity_command,
    make_joint_trajectory_buffer,
    mrad_to_rad,
    normalize_control_profile,
    predict_joint_reference_buffer,
    rad_to_mrad,
    reference_tracking_error,
    reorder_positions_from_joint_state,
    should_send_command,
    should_invalidate_reference,
    should_reset_reference_for_tracking_error,
    servo_stop_reason,
    teleop_servo_workspace_policy,
    unity_to_amir_base,
    validate_joint_positions,
    validate_joint_names,
    workspace_axis_margins,
)
from amir_irt_mro.trajectory_format import (
    make_forward_position_command,
    make_real_teleop_buffer_joint_trajectory,
    make_real_teleop_joint_trajectory,
    make_velocity_joint_trajectory,
)


class FakeLogger:

    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakePublisher:

    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeTime:
    nanoseconds = 1_000_000_000


class FakeClock:

    def now(self):
        return FakeTime()


def make_fake_forward_position_node(q_measured):
    node = object.__new__(AmirPalmIkNode)
    publisher = FakePublisher()
    node.forward_position_pub = publisher
    node.forward_position_command_topic = DEFAULT_FORWARD_POSITION_COMMAND_TOPIC
    node.forward_position_publish_rate_hz = 50.0
    node.reference_servo_rate_hz = 50.0
    node.reference_dt_sec = 0.02
    node.actual_reference_tick_hz = 50.0
    node.current_joints = np.array(q_measured, dtype=np.float64)
    node.joint_state_received = True
    node.forward_position_hold_sent_after_stop = False
    node.mode = "real"
    node.control_profile = "teleop_forward_position"
    node.dry_run = False
    node.last_candidate_accepted = True
    node.last_reject_reason = "accepted"
    node.reference_reset_reason = "none"
    node.reference_reset_applied = False
    node.reference_tracking_error_threshold_rad = 0.08
    node.tracking_error_rad = 0.0
    node.tracking_error_joint_diffs = np.zeros(5, dtype=np.float64)
    node.q_reference = np.array(q_measured, dtype=np.float64)
    node.q_reference_velocity = np.ones(5, dtype=np.float64)
    node.q_reference_valid = True
    node.last_buffer_publish_time = None
    node.last_buffer_q_target = None
    node.last_buffer_waypoints = []
    node.last_buffer_publish_reason = "not_started"
    node.last_forward_position_publish_time = None
    node.last_forward_position_publish_reason = "not_started"
    node.forward_position_pending_publish_reason = "not_started"
    node.last_forward_position_data = []
    node.latest_q_target = np.array(q_measured, dtype=np.float64)
    node.latest_p_desired = None
    node.latest_p_command = None
    node.buffer_target_available = True
    node.last_q_current = np.array(q_measured, dtype=np.float64)
    node.last_q_candidate = np.array(q_measured, dtype=np.float64)
    node.last_q_solution = np.array(q_measured, dtype=np.float64)
    node.last_dq = np.zeros(5, dtype=np.float64)
    node.last_servo_joint_velocity = np.zeros(5, dtype=np.float64)
    node.get_logger = lambda: FakeLogger()
    node.get_clock = lambda: FakeClock()
    return node, publisher


def test_rad_to_mrad_conversion():
    assert rad_to_mrad([0.0, 1.0, -0.5]) == [0.0, 1000.0, -500.0]


def test_mrad_to_rad_conversion():
    assert mrad_to_rad([0.0, 1000.0, -500.0]) == [0.0, 1.0, -0.5]


def test_joint_name_order_validation_accepts_expected_order():
    assert validate_joint_names(DEFAULT_ARM_JOINT_NAMES) == list(
        DEFAULT_ARM_JOINT_NAMES)


def test_joint_name_order_validation_rejects_swapped_order():
    swapped = ["Joint_2", "Joint_1", "Joint_3", "Joint_4", "Joint_5"]
    with pytest.raises(ValueError):
        validate_joint_names(swapped)


def test_joint_state_reorder_uses_expected_order():
    names = ["Joint_3", "Joint_1", "Joint_5", "Joint_2", "Joint_4"]
    positions = [-3.0, -1.0, -5.0, -2.0, -4.0]
    assert reorder_positions_from_joint_state(names, positions) == [
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
    ]


def test_unity_to_amir_base_uses_axis_map_sign_and_offset():
    assert unity_to_amir_base(
        unity_xyz=[0.1, 0.2, 0.3],
        axis_map=[2, 0, 1],
        axis_sign=[1.0, -1.0, 1.0],
        base_offset_xyz=[1.0, 2.0, 3.0],
    ) == pytest.approx([1.3, 1.9, 3.2])


def test_initial_position_is_canonical_and_within_limits():
    assert validate_joint_positions(
        DEFAULT_ARM_INITIAL_POSITIONS,
        DEFAULT_ARM_JOINT_NAMES,
    ) == pytest.approx([
        0.0,
        1.5707963267948966,
        -1.5707963267948966,
        0.7853981633974483,
        0.0,
    ])


def test_max_joint_step_rad_limit():
    current = [0.0, 1.0, -1.0]
    target = [0.1, 0.98, -2.0]
    assert clamp_joint_step(target, current, 0.01) == [0.01, 0.99, -1.01]


def test_dry_run_does_not_send_real_command():
    assert not should_send_command(
        mode="real",
        dry_run=True,
        joint_state_received=True,
        palm_timed_out=False,
    )


def test_palm_pose_timeout_stops_command():
    assert not should_send_command(
        mode="real",
        dry_run=False,
        joint_state_received=True,
        palm_timed_out=True,
    )


def test_gazebo_mode_keeps_existing_arm_controller_topic():
    assert DEFAULT_ARM_COMMAND_TOPIC == "/arm_controller/joint_trajectory"
    assert should_send_command(
        mode="gazebo",
        dry_run=True,
        joint_state_received=False,
        palm_timed_out=False,
    )


def test_real_mode_without_arm_state_does_not_send():
    assert not should_send_command(
        mode="real",
        dry_run=False,
        joint_state_received=False,
        palm_timed_out=False,
    )


def test_control_profile_names_and_legacy_alias():
    assert normalize_control_profile("safe_trajectory") == "safe_trajectory"
    assert normalize_control_profile("teleop_trajectory_stream") == (
        "teleop_trajectory_stream")
    assert normalize_control_profile("teleop_trajectory_buffer") == (
        "teleop_trajectory_buffer")
    assert normalize_control_profile("teleop_velocity_servo") == (
        "teleop_velocity_servo")
    assert normalize_control_profile("teleop_forward_position") == (
        "teleop_forward_position")
    assert normalize_control_profile("teleop_servo") == (
        "teleop_trajectory_stream")
    assert is_teleop_trajectory_profile("teleop_servo")
    assert is_teleop_trajectory_profile("teleop_trajectory_buffer")
    assert is_teleop_buffer_profile("teleop_trajectory_buffer")
    assert is_teleop_velocity_profile("teleop_velocity_servo")
    assert is_teleop_forward_position_profile("teleop_forward_position")
    assert not is_teleop_trajectory_profile("teleop_forward_position")
    assert not is_teleop_buffer_profile("teleop_forward_position")


def test_relative_target_zero_delta_stays_at_anchor_ee():
    result = compute_relative_target(
        anchor_ee=[-0.0035, 0.32, 0.025],
        palm_anchor=[0.0, 0.0, 0.0],
        palm_now=[0.0, 0.0, 0.0],
        axis_map=[2, 0, 1],
        axis_sign=[1.0, 1.0, 1.0],
        axis_gain=[1.0, 1.0, 1.0],
        scale_xyz=[0.6, 0.3, 0.6],
    )
    assert result["target_control"] == [-0.0035, 0.32, 0.025]
    assert result["target_minus_anchor_ee"] == [0.0, 0.0, 0.0]


def test_relative_target_uses_axis_transform_and_scale():
    result = compute_relative_target(
        anchor_ee=[1.0, 2.0, 3.0],
        palm_anchor=[0.0, 0.0, 0.0],
        palm_now=[0.1, 0.2, 0.3],
        axis_map=[2, 0, 1],
        axis_sign=[1.0, -1.0, 1.0],
        axis_gain=[1.0, 1.0, 2.0],
        scale_xyz=[0.5, 0.25, 0.1],
    )
    assert result["delta_hand"] == [0.1, 0.2, 0.3]
    assert result["delta_robot"] == pytest.approx([0.3, -0.1, 0.4])
    assert result["scaled_delta"] == pytest.approx([0.15, -0.025, 0.04])
    assert result["target_control"] == pytest.approx([1.15, 1.975, 3.04])


def test_relative_teleop_zero_delta_p_command_matches_anchor_ee():
    anchor_ee = [-0.0035, 0.32, 0.025]
    target = compute_relative_target(
        anchor_ee=anchor_ee,
        palm_anchor=[0.0, 0.0, 0.0],
        palm_now=[0.0, 0.0, 0.0],
        axis_map=[2, 0, 1],
        axis_sign=[1.0, 1.0, 1.0],
        axis_gain=[1.0, 1.0, 1.0],
        scale_xyz=[0.6, 0.3, 0.6],
    )
    command = limit_cartesian_command(
        desired_position=target["target_control"],
        previous_position=anchor_ee,
        previous_velocity=[0.0, 0.0, 0.0],
        dt_sec=0.1,
        max_velocity_mps=0.05,
        max_acceleration_mps2=0.20,
    )
    assert command["position"] == pytest.approx(anchor_ee)
    assert command["velocity"] == pytest.approx([0.0, 0.0, 0.0])


def test_real_relative_teleop_ignores_adapter_request():
    policy = teleop_servo_workspace_policy(
        mode="real",
        target_mapping_mode="relative",
        control_profile="teleop_servo",
        adapter_requested=True,
    )
    assert policy["workspace_check_frame"] == "fk_frame"
    assert policy["adapter_applied"] is False


def test_workspace_inside_zero_delta_is_not_rejected():
    anchor_ee = [-0.0035, 0.32, 0.025]
    workspace_min = [-0.10, 0.20, 0.00]
    workspace_max = [0.10, 0.40, 0.10]
    assert is_target_within_workspace(anchor_ee, workspace_min, workspace_max)
    assert all(
        margin >= 0.0
        for margin in workspace_axis_margins(
            anchor_ee, workspace_min, workspace_max)
    )


def test_servo_zero_delta_keeps_measured_joint_command():
    result = limit_joint_servo_command(
        target_rad=[0.1, 0.2, -0.3, 0.4, 0.0],
        measured_rad=[0.1, 0.2, -0.3, 0.4, 0.0],
        previous_velocity_radps=[0.0, 0.0, 0.0, 0.0, 0.0],
        horizon_sec=0.2,
        dt_sec=0.1,
        max_velocity_radps=0.15,
        max_acceleration_radps2=0.50,
    )
    assert result["position"] == pytest.approx([0.1, 0.2, -0.3, 0.4, 0.0])
    assert result["velocity"] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


def test_servo_joint_command_respects_velocity_limit():
    result = limit_joint_servo_command(
        target_rad=[1.0, 0.0, 0.0, 0.0, 0.0],
        measured_rad=[0.0, 0.0, 0.0, 0.0, 0.0],
        previous_velocity_radps=[0.0, 0.0, 0.0, 0.0, 0.0],
        horizon_sec=0.2,
        dt_sec=1.0,
        max_velocity_radps=0.15,
        max_acceleration_radps2=10.0,
    )
    assert max(abs(value) for value in result["velocity"]) <= 0.15


def test_velocity_servo_joint_velocity_respects_limit():
    result = limit_joint_velocity_command(
        desired_velocity_radps=[1.0, -1.0, 0.2, 0.0, 0.0],
        previous_velocity_radps=[0.0, 0.0, 0.0, 0.0, 0.0],
        dt_sec=1.0,
        max_velocity_radps=0.05,
        max_acceleration_radps2=10.0,
    )
    assert result["velocity"] == pytest.approx([0.05, -0.05, 0.05, 0.0, 0.0])


def test_velocity_servo_cartesian_deadzone_can_command_zero_velocity():
    result = limit_cartesian_velocity(
        desired_velocity_mps=[0.0, 0.0, 0.0],
        previous_velocity_mps=[0.0, 0.0, 0.0],
        dt_sec=0.05,
        max_velocity_mps=0.05,
        max_acceleration_mps2=0.20,
    )
    assert result["velocity"] == pytest.approx([0.0, 0.0, 0.0])


def test_servo_palm_timeout_requests_hold_reason():
    assert servo_stop_reason(
        palm_timed_out=True,
        joint_state_timed_out=False,
        deadman_active=True,
    ) == "palm_pose_timeout"


def test_servo_target_outside_workspace_is_rejected():
    assert not is_target_within_workspace(
        target=[0.5, 0.0, 0.2],
        workspace_min=[0.0, -0.1, 0.0],
        workspace_max=[0.4, 0.1, 0.4],
    )


def test_servo_command_joint_order_is_canonical():
    assert list(DEFAULT_ARM_JOINT_NAMES) == [
        "Joint_1",
        "Joint_2",
        "Joint_3",
        "Joint_4",
        "Joint_5",
    ]


def test_real_teleop_trajectory_uses_known_good_format():
    msg = make_real_teleop_joint_trajectory(
        positions=[0.1, 0.2, -0.3, 0.4, 0.0],
        time_from_start_sec=1.0,
        joint_names=DEFAULT_ARM_JOINT_NAMES,
    )
    point = msg.points[0]
    assert msg.header.stamp.sec == 0
    assert msg.header.stamp.nanosec == 0
    assert list(msg.joint_names) == list(DEFAULT_ARM_JOINT_NAMES)
    assert len(point.positions) == 5
    assert list(point.velocities) == []
    assert list(point.accelerations) == []
    assert list(point.effort) == []
    assert point.time_from_start.sec == 1
    assert point.time_from_start.nanosec == 0


def test_trajectory_buffer_waypoint_count_and_times():
    buffer = make_joint_trajectory_buffer(
        target_rad=[0.2, 0.0, 0.0, 0.0, 0.0],
        measured_rad=[0.0, 0.0, 0.0, 0.0, 0.0],
        previous_velocity_radps=[0.0, 0.0, 0.0, 0.0, 0.0],
        horizon_sec=1.0,
        point_dt_sec=0.1,
        max_velocity_radps=0.15,
        max_acceleration_radps2=0.50,
    )
    assert len(buffer["waypoints"]) == 10
    assert buffer["times"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5,
         0.6, 0.7, 0.8, 0.9, 1.0])
    assert all(
        later > earlier
        for earlier, later in zip(buffer["times"], buffer["times"][1:])
    )
    assert all(len(waypoint) == 5 for waypoint in buffer["waypoints"])


def test_trajectory_buffer_respects_velocity_and_acceleration_limits():
    max_velocity = 0.15
    max_acceleration = 0.50
    buffer = make_joint_trajectory_buffer(
        target_rad=[1.0, -1.0, 0.4, 0.0, 0.0],
        measured_rad=[0.0, 0.0, 0.0, 0.0, 0.0],
        previous_velocity_radps=[0.0, 0.0, 0.0, 0.0, 0.0],
        horizon_sec=1.0,
        point_dt_sec=0.1,
        max_velocity_radps=max_velocity,
        max_acceleration_radps2=max_acceleration,
    )
    velocities = buffer["velocities"]
    assert all(
        abs(value) <= max_velocity + 1e-9
        for velocity in velocities
        for value in velocity
    )
    previous_velocity = [0.0, 0.0, 0.0, 0.0, 0.0]
    previous_time = 0.0
    for velocity, time_value in zip(velocities, buffer["times"]):
        dt = time_value - previous_time
        acceleration = [
            (velocity[idx] - previous_velocity[idx]) / dt
            for idx in range(5)
        ]
        assert max(abs(value) for value in acceleration) <= (
            max_acceleration + 1e-9)
        previous_velocity = velocity
        previous_time = time_value


def test_reference_buffer_waypoints_respect_limits_from_reference_state():
    max_velocity = 0.15
    max_acceleration = 0.50
    buffer = predict_joint_reference_buffer(
        reference_rad=[0.01, 0.0, 0.0, 0.0, 0.0],
        target_rad=[0.2, -0.1, 0.05, 0.0, 0.0],
        reference_velocity_radps=[0.04, 0.0, 0.0, 0.0, 0.0],
        horizon_sec=0.30,
        point_dt_sec=0.05,
        integration_dt_sec=0.02,
        max_velocity_radps=max_velocity,
        max_acceleration_radps2=max_acceleration,
    )
    previous_velocity = [0.04, 0.0, 0.0, 0.0, 0.0]
    previous_time = 0.0
    for waypoint, velocity, time_value in zip(
            buffer["waypoints"], buffer["velocities"], buffer["times"]):
        assert len(waypoint) == 5
        assert max(abs(value) for value in velocity) <= (
            max_velocity + 1e-9)
        dt = time_value - previous_time
        acceleration = [
            (velocity[idx] - previous_velocity[idx]) / dt
            for idx in range(5)
        ]
        assert max(abs(value) for value in acceleration) <= (
            max_acceleration + 1e-9)
        previous_velocity = velocity
        previous_time = time_value


def test_real_teleop_buffer_trajectory_uses_known_good_format():
    buffer = make_joint_trajectory_buffer(
        target_rad=[0.2, 0.1, -0.1, 0.05, 0.0],
        measured_rad=[0.0, 0.0, 0.0, 0.0, 0.0],
        previous_velocity_radps=[0.0, 0.0, 0.0, 0.0, 0.0],
        horizon_sec=1.0,
        point_dt_sec=0.1,
        max_velocity_radps=0.15,
        max_acceleration_radps2=0.50,
    )
    msg = make_real_teleop_buffer_joint_trajectory(
        waypoints=buffer["waypoints"],
        times_from_start_sec=buffer["times"],
        joint_names=DEFAULT_ARM_JOINT_NAMES,
    )
    assert msg.header.stamp.sec == 0
    assert msg.header.stamp.nanosec == 0
    assert list(msg.joint_names) == list(DEFAULT_ARM_JOINT_NAMES)
    assert len(msg.points) == 10
    previous_nsec = 0
    for point in msg.points:
        assert len(point.positions) == 5
        assert list(point.velocities) == []
        assert list(point.accelerations) == []
        assert list(point.effort) == []
        current_nsec = (
            point.time_from_start.sec * 1_000_000_000 +
            point.time_from_start.nanosec
        )
        assert current_nsec > previous_nsec
        previous_nsec = current_nsec


def test_reference_tick_respects_velocity_and_acceleration_limits():
    q_reference = [0.0, 0.0, 0.0, 0.0, 0.0]
    q_velocity = [0.0, 0.0, 0.0, 0.0, 0.0]
    q_target = [1.0, -1.0, 0.5, 0.0, 0.0]
    dt = 0.02
    max_velocity = 0.15
    max_acceleration = 0.50

    for _ in range(20):
        previous_velocity = q_velocity
        step = integrate_joint_reference(
            reference_rad=q_reference,
            target_rad=q_target,
            previous_velocity_radps=previous_velocity,
            dt_sec=dt,
            max_velocity_radps=max_velocity,
            max_acceleration_radps2=max_acceleration,
        )
        q_reference = step["position"]
        q_velocity = step["velocity"]
        assert max(abs(value) for value in q_velocity) <= (
            max_velocity + 1e-9)
        acceleration = [
            (q_velocity[idx] - previous_velocity[idx]) / dt
            for idx in range(5)
        ]
        assert max(abs(value) for value in acceleration) <= (
            max_acceleration + 1e-9)


def test_reference_buffer_is_continuous_between_publishes():
    q_reference = [0.0, 0.0, 0.0, 0.0, 0.0]
    q_velocity = [0.0, 0.0, 0.0, 0.0, 0.0]
    q_target = [1.0, 0.0, 0.0, 0.0, 0.0]
    reference_dt = 0.02
    point_dt = 0.05
    update_dt = 0.10
    max_velocity = 0.15
    max_acceleration = 0.50
    first = predict_joint_reference_buffer(
        reference_rad=q_reference,
        target_rad=q_target,
        reference_velocity_radps=q_velocity,
        horizon_sec=0.30,
        point_dt_sec=point_dt,
        integration_dt_sec=reference_dt,
        max_velocity_radps=max_velocity,
        max_acceleration_radps2=max_acceleration,
    )

    elapsed = 0.0
    while elapsed < update_dt - 1e-12:
        dt = min(reference_dt, update_dt - elapsed)
        step = integrate_joint_reference(
            q_reference,
            q_target,
            q_velocity,
            dt,
            max_velocity,
            max_acceleration,
        )
        q_reference = step["position"]
        q_velocity = step["velocity"]
        elapsed += dt

    second = predict_joint_reference_buffer(
        reference_rad=q_reference,
        target_rad=q_target,
        reference_velocity_radps=q_velocity,
        horizon_sec=0.30,
        point_dt_sec=point_dt,
        integration_dt_sec=reference_dt,
        max_velocity_radps=max_velocity,
        max_acceleration_radps2=max_acceleration,
    )
    assert first["times"][2] == pytest.approx(update_dt + point_dt)
    assert second["waypoints"][0] == pytest.approx(
        first["waypoints"][2], abs=1e-4)


def test_reference_resets_only_when_tracking_error_exceeds_threshold():
    q_measured = [0.0, 0.0, 0.0, 0.0, 0.0]
    q_reference_ok = [0.079, 0.0, 0.0, 0.0, 0.0]
    q_reference_bad = [0.081, 0.0, 0.0, 0.0, 0.0]
    assert reference_tracking_error(
        q_reference_ok, q_measured) == pytest.approx(0.079)
    assert not should_reset_reference_for_tracking_error(
        q_reference_ok, q_measured, 0.08)
    assert should_reset_reference_for_tracking_error(
        q_reference_bad, q_measured, 0.08)


def test_tracking_error_small_relative_to_threshold_does_not_reset():
    q_measured = [0.0, 0.0, 0.0, 0.0, 0.0]
    q_reference = [0.002, 0.0, 0.0, 0.0, 0.0]
    assert reference_tracking_error(
        q_reference, q_measured) == pytest.approx(0.002)
    assert not should_reset_reference_for_tracking_error(
        q_reference, q_measured, 0.12)


def test_reference_servo_50hz_uses_20ms_integration_step():
    step = integrate_joint_reference(
        reference_rad=[0.0, 0.0, 0.0, 0.0, 0.0],
        target_rad=[1.0, 0.0, 0.0, 0.0, 0.0],
        previous_velocity_radps=[0.0, 0.0, 0.0, 0.0, 0.0],
        dt_sec=1.0 / 50.0,
        max_velocity_radps=0.15,
        max_acceleration_radps2=0.50,
    )
    assert step["velocity"][0] == pytest.approx(0.01)
    assert step["position"][0] == pytest.approx(0.0002)


def test_timeout_and_deadman_invalidate_reference_after_hold():
    assert should_invalidate_reference("palm_pose_timeout")
    assert should_invalidate_reference("deadman_inactive")
    assert not should_invalidate_reference("joint_state_timeout")
    assert not should_invalidate_reference("accepted")


def test_timeout_hold_trajectory_is_one_point_known_good_format():
    assert servo_stop_reason(
        palm_timed_out=True,
        joint_state_timed_out=False,
        deadman_active=True,
    ) == "palm_pose_timeout"
    q_measured = [0.1, 0.2, -0.3, 0.4, 0.0]
    msg = make_real_teleop_joint_trajectory(
        positions=q_measured,
        time_from_start_sec=0.25,
        joint_names=DEFAULT_ARM_JOINT_NAMES,
    )
    assert msg.header.stamp.sec == 0
    assert msg.header.stamp.nanosec == 0
    assert len(msg.points) == 1
    assert msg.points[0].positions == pytest.approx(q_measured)
    assert list(msg.points[0].velocities) == []


def test_forward_position_command_data_len_and_canonical_order():
    q_reference = [0.1, 0.2, -0.3, 0.4, 0.5]
    msg = make_forward_position_command(
        positions=q_reference,
        joint_names=DEFAULT_ARM_JOINT_NAMES,
    )
    assert len(msg.data) == 5
    assert list(DEFAULT_ARM_JOINT_NAMES) == [
        "Joint_1",
        "Joint_2",
        "Joint_3",
        "Joint_4",
        "Joint_5",
    ]
    assert msg.data == pytest.approx(q_reference)


def test_forward_position_rejects_noncanonical_joint_order():
    swapped = ["Joint_2", "Joint_1", "Joint_3", "Joint_4", "Joint_5"]
    with pytest.raises(ValueError):
        make_forward_position_command(
            positions=[0.1, 0.2, -0.3, 0.4, 0.5],
            joint_names=swapped,
        )


def test_forward_position_command_rejects_nan_and_inf():
    for invalid_value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            validate_joint_positions(
                [0.0, invalid_value, -0.3, 0.4, 0.0],
                DEFAULT_ARM_JOINT_NAMES,
            )


def test_forward_position_command_rejects_joint_limit_violation():
    with pytest.raises(ValueError):
        make_forward_position_command(
            positions=[3.5, 0.2, -0.3, 0.4, 0.0],
            joint_names=DEFAULT_ARM_JOINT_NAMES,
        )


def test_forward_position_timeout_hold_publishes_measured_once():
    q_measured = [0.1, 0.2, -0.3, 0.4, 0.0]
    node, publisher = make_fake_forward_position_node(q_measured)

    node.publish_forward_position_hold_once("palm_pose_timeout")
    assert len(publisher.messages) == 1
    assert publisher.messages[0].data == pytest.approx(q_measured)
    assert node.q_reference is None
    assert node.q_reference_valid is False
    assert node.q_reference_velocity == pytest.approx([0.0] * 5)

    node.publish_forward_position_hold_once("palm_pose_timeout")
    assert len(publisher.messages) == 1


def test_forward_position_deadman_hold_publishes_measured_once():
    q_measured = [0.1, 0.2, -0.3, 0.4, 0.0]
    node, publisher = make_fake_forward_position_node(q_measured)

    node.publish_forward_position_hold_once("deadman_inactive")
    assert len(publisher.messages) == 1
    assert publisher.messages[0].data == pytest.approx(q_measured)
    assert node.q_reference is None
    assert node.q_reference_valid is False

    node.publish_forward_position_hold_once("deadman_inactive")
    assert len(publisher.messages) == 1


def test_forward_position_publish_rejects_nan_without_publish():
    q_measured = [0.1, 0.2, -0.3, 0.4, 0.0]
    node, publisher = make_fake_forward_position_node(q_measured)
    q_reference = [0.1, float("nan"), -0.3, 0.4, 0.0]

    published = node.publish_forward_position_command(
        q_reference,
        np.array(q_measured, dtype=np.float64),
        np.array(q_measured, dtype=np.float64),
        "periodic_reference",
        "accepted",
    )

    assert not published
    assert publisher.messages == []


def test_velocity_servo_trajectory_uses_velocity_only_format():
    msg = make_velocity_joint_trajectory(
        velocities=[0.01, -0.02, 0.03, 0.0, 0.0],
        time_from_start_sec=0.05,
        joint_names=DEFAULT_ARM_JOINT_NAMES,
    )
    point = msg.points[0]
    assert msg.header.stamp.sec == 0
    assert msg.header.stamp.nanosec == 0
    assert list(msg.joint_names) == list(DEFAULT_ARM_JOINT_NAMES)
    assert list(point.positions) == []
    assert len(point.velocities) == 5
    assert point.velocities == pytest.approx([0.01, -0.02, 0.03, 0.0, 0.0])
    assert point.time_from_start.sec == 0
    assert point.time_from_start.nanosec == 50000000
