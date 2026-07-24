"""Safety helpers for Amir real-robot arm commands."""

# IK and ros2_control command side use radians.  The lower Amir transport uses
# milliradians, including the initial-position offset used by amir_driver.

import math

DEFAULT_ARM_JOINT_NAMES = (
    "Joint_1",
    "Joint_2",
    "Joint_3",
    "Joint_4",
    "Joint_5",
)
DEFAULT_ARM_JOINT_LIMITS = (
    (-2.96706, 2.96706),
    (0.0, 2.356194),
    (-2.792527, 0.0),
    (-2.094395, 1.308997),
    (-2.75762, 2.75762),
)
DEFAULT_ARM_INITIAL_POSITIONS = (
    0.0,
    math.pi / 2.0,
    -math.pi / 2.0,
    math.pi / 4.0,
    0.0,
)
DEFAULT_UNITY_TO_AMIR_AXIS_MAP = (2, 0, 1)
DEFAULT_UNITY_TO_AMIR_AXIS_SIGN = (1.0, 1.0, 1.0)
DEFAULT_UNITY_TO_AMIR_BASE_OFFSET_XYZ = (0.0, 0.0, 0.0)
DEFAULT_ARM_COMMAND_TOPIC = "/arm_controller/joint_trajectory"
DEFAULT_ARM_STATE_TOPIC = "/joint_states"
DEFAULT_GRIPPER_COMMAND_TOPIC = "/grasp_command"
DEFAULT_FORWARD_POSITION_COMMAND_TOPIC = "/forward_position_controller/commands"
SUPPORTED_COMMAND_UNITS = ("rad", "mrad")
CONTROL_PROFILE_SAFE_TRAJECTORY = "safe_trajectory"
CONTROL_PROFILE_TRAJECTORY_STREAM = "teleop_trajectory_stream"
CONTROL_PROFILE_TRAJECTORY_BUFFER = "teleop_trajectory_buffer"
CONTROL_PROFILE_VELOCITY_SERVO = "teleop_velocity_servo"
CONTROL_PROFILE_FORWARD_POSITION = "teleop_forward_position"
CONTROL_PROFILE_LEGACY_TELEOP_SERVO = "teleop_servo"
SUPPORTED_CONTROL_PROFILES = (
    CONTROL_PROFILE_SAFE_TRAJECTORY,
    CONTROL_PROFILE_TRAJECTORY_STREAM,
    CONTROL_PROFILE_TRAJECTORY_BUFFER,
    CONTROL_PROFILE_VELOCITY_SERVO,
    CONTROL_PROFILE_FORWARD_POSITION,
)
TELEOP_CONTROL_PROFILES = (
    CONTROL_PROFILE_TRAJECTORY_STREAM,
    CONTROL_PROFILE_TRAJECTORY_BUFFER,
    CONTROL_PROFILE_VELOCITY_SERVO,
    CONTROL_PROFILE_FORWARD_POSITION,
)

AMIR_DRIVER_INITIAL_POSITION_MRAD = (
    2970.0,
    2360.0,
    -2790.0,
    1310.0,
    -2760.0,
    260.0,
)


def normalize_command_unit(unit):
    normalized = str(unit).strip().lower()
    if normalized not in SUPPORTED_COMMAND_UNITS:
        raise ValueError(
            f"Unsupported command_unit={unit!r}. "
            f"Expected one of {SUPPORTED_COMMAND_UNITS}.")
    return normalized


def normalize_control_profile(profile):
    normalized = str(profile).strip().lower()
    if normalized == CONTROL_PROFILE_LEGACY_TELEOP_SERVO:
        return CONTROL_PROFILE_TRAJECTORY_STREAM
    if normalized not in SUPPORTED_CONTROL_PROFILES:
        raise ValueError(
            f"Unsupported control_profile={profile!r}. "
            f"Expected one of {SUPPORTED_CONTROL_PROFILES}.")
    return normalized


def is_teleop_control_profile(profile):
    normalized = normalize_control_profile(profile)
    return normalized in TELEOP_CONTROL_PROFILES


def is_teleop_trajectory_profile(profile):
    return normalize_control_profile(profile) in (
        CONTROL_PROFILE_TRAJECTORY_STREAM,
        CONTROL_PROFILE_TRAJECTORY_BUFFER,
    )


def is_teleop_buffer_profile(profile):
    return (
        normalize_control_profile(profile) ==
        CONTROL_PROFILE_TRAJECTORY_BUFFER)


def is_teleop_velocity_profile(profile):
    return normalize_control_profile(profile) == CONTROL_PROFILE_VELOCITY_SERVO


def is_teleop_forward_position_profile(profile):
    return normalize_control_profile(profile) == CONTROL_PROFILE_FORWARD_POSITION


def rad_to_mrad(values_rad):
    return [float(value) * 1000.0 for value in values_rad]


def mrad_to_rad(values_mrad):
    return [float(value) / 1000.0 for value in values_mrad]


def validate_joint_names(actual, expected=DEFAULT_ARM_JOINT_NAMES):
    actual_list = [str(name) for name in actual]
    expected_list = [str(name) for name in expected]
    if actual_list != expected_list:
        raise ValueError(
            "Joint name/order mismatch: "
            f"actual={actual_list}, expected={expected_list}")
    return actual_list


def validate_joint_positions(
        positions,
        joint_names=DEFAULT_ARM_JOINT_NAMES,
        joint_limits=DEFAULT_ARM_JOINT_LIMITS):
    names = validate_joint_names(joint_names, DEFAULT_ARM_JOINT_NAMES)
    values = [float(value) for value in positions]
    if len(values) != len(names):
        raise ValueError(
            "Joint position count mismatch: "
            f"positions={len(values)}, joint_names={len(names)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Joint positions contain NaN or Inf: {values}")
    if joint_limits is not None:
        limits = [
            (float(lower), float(upper))
            for lower, upper in joint_limits
        ]
        if len(limits) != len(names):
            raise ValueError(
                "Joint limit count mismatch: "
                f"limits={len(limits)}, joint_names={len(names)}")
        for idx, value in enumerate(values):
            lower, upper = limits[idx]
            if value < lower - 1e-9 or value > upper + 1e-9:
                raise ValueError(
                    "Joint position limit violation: "
                    f"{names[idx]}={value}, limit=[{lower}, {upper}]")
    return values


def validate_xyz(values, name="xyz"):
    xyz = [float(value) for value in values]
    if len(xyz) != 3:
        raise ValueError(f"{name} must have length 3: {xyz}")
    if not all(math.isfinite(value) for value in xyz):
        raise ValueError(f"{name} contains NaN or Inf: {xyz}")
    return xyz


def validate_axis_map(axis_map):
    values = [int(value) for value in axis_map]
    if len(values) != 3 or sorted(values) != [0, 1, 2]:
        raise ValueError(f"axis_map must be a permutation of [0, 1, 2]: {values}")
    return values


def validate_axis_sign(axis_sign):
    values = [float(value) for value in axis_sign]
    if len(values) != 3:
        raise ValueError(f"axis_sign must have length 3: {values}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"axis_sign contains NaN or Inf: {values}")
    return [-1.0 if value < 0.0 else 1.0 for value in values]


def unity_to_amir_base(
        unity_xyz,
        axis_map=DEFAULT_UNITY_TO_AMIR_AXIS_MAP,
        axis_sign=DEFAULT_UNITY_TO_AMIR_AXIS_SIGN,
        base_offset_xyz=DEFAULT_UNITY_TO_AMIR_BASE_OFFSET_XYZ):
    unity = validate_xyz(unity_xyz, "unity_xyz")
    mapping = validate_axis_map(axis_map)
    signs = validate_axis_sign(axis_sign)
    offset = validate_xyz(base_offset_xyz, "base_offset_xyz")
    return [
        offset[robot_axis] + signs[robot_axis] * unity[mapping[robot_axis]]
        for robot_axis in range(3)
    ]


def reorder_positions_from_joint_state(
        names, positions, expected=DEFAULT_ARM_JOINT_NAMES):
    name_to_idx = {str(name): idx for idx, name in enumerate(names)}
    missing = [name for name in expected if name not in name_to_idx]
    if missing:
        raise ValueError(f"JointState is missing required joints: {missing}")

    ordered = []
    for name in expected:
        idx = name_to_idx[name]
        if idx >= len(positions):
            raise ValueError(
                "JointState position array has no value for "
                f"{name} at index {idx}")
        ordered.append(float(positions[idx]))
    return ordered


def clamp_joint_step(target_rad, current_rad, max_step_rad):
    target = [float(value) for value in target_rad]
    current = [float(value) for value in current_rad]
    if len(target) != len(current):
        raise ValueError(
            "target/current joint count mismatch: "
            f"{len(target)} != {len(current)}")

    max_step = float(max_step_rad)
    if max_step < 0.0:
        raise ValueError(f"max_step_rad must be non-negative: {max_step_rad}")

    limited = []
    for target_value, current_value in zip(target, current):
        delta = target_value - current_value
        if delta > max_step:
            limited.append(current_value + max_step)
        elif delta < -max_step:
            limited.append(current_value - max_step)
        else:
            limited.append(target_value)
    return limited


def compute_relative_target(
        anchor_ee,
        palm_anchor,
        palm_now,
        axis_map,
        axis_sign,
        axis_gain,
        scale_xyz,
        deadzone_m=0.0):
    anchor = [float(value) for value in anchor_ee]
    palm_anchor_values = [float(value) for value in palm_anchor]
    palm_now_values = [float(value) for value in palm_now]
    axis_map_values = [int(value) for value in axis_map]
    axis_sign_values = [float(value) for value in axis_sign]
    axis_gain_values = [float(value) for value in axis_gain]
    scale_values = [float(value) for value in scale_xyz]

    for name, values in (
            ("anchor_ee", anchor),
            ("palm_anchor", palm_anchor_values),
            ("palm_now", palm_now_values),
            ("axis_map", axis_map_values),
            ("axis_sign", axis_sign_values),
            ("axis_gain", axis_gain_values),
            ("scale_xyz", scale_values)):
        if len(values) != 3:
            raise ValueError(f"{name} must have length 3: {values}")

    delta_hand = [
        palm_now_values[idx] - palm_anchor_values[idx]
        for idx in range(3)
    ]
    delta_norm = math.sqrt(sum(value * value for value in delta_hand))
    if delta_norm < max(float(deadzone_m), 0.0):
        delta_hand = [0.0, 0.0, 0.0]

    if sorted(axis_map_values) != [0, 1, 2]:
        raise ValueError("axis_map must be a permutation of [0, 1, 2]")

    delta_robot = [
        axis_sign_values[robot_axis] *
        axis_gain_values[robot_axis] *
        delta_hand[axis_map_values[robot_axis]]
        for robot_axis in range(3)
    ]
    scaled_delta = [
        scale_values[idx] * delta_robot[idx]
        for idx in range(3)
    ]
    target_delta = scaled_delta.copy()
    target_control = [
        anchor[idx] + target_delta[idx]
        for idx in range(3)
    ]
    return {
        "delta_hand": delta_hand,
        "delta_robot": delta_robot,
        "scaled_delta": scaled_delta,
        "target_control": target_control,
        "target_minus_anchor_ee": target_delta,
    }


def vector_norm(values):
    return math.sqrt(sum(float(value) * float(value) for value in values))


def limit_vector_norm(values, max_norm):
    values_list = [float(value) for value in values]
    limit = max(float(max_norm), 0.0)
    norm = vector_norm(values_list)
    if limit <= 0.0:
        return [0.0 for _ in values_list], norm > 0.0
    if norm <= limit or norm <= 1e-12:
        return values_list, False
    scale = limit / norm
    return [value * scale for value in values_list], True


def limit_cartesian_command(
        desired_position,
        previous_position,
        previous_velocity,
        dt_sec,
        max_velocity_mps,
        max_acceleration_mps2):
    dt = max(float(dt_sec), 1e-6)
    desired = [float(value) for value in desired_position]
    previous = [float(value) for value in previous_position]
    prev_velocity = [float(value) for value in previous_velocity]
    if len(desired) != 3 or len(previous) != 3 or len(prev_velocity) != 3:
        raise ValueError("Cartesian command vectors must have length 3")

    raw_velocity = [
        (desired[idx] - previous[idx]) / dt
        for idx in range(3)
    ]
    velocity_limited, velocity_was_limited = limit_vector_norm(
        raw_velocity, max_velocity_mps)
    acceleration_delta = [
        velocity_limited[idx] - prev_velocity[idx]
        for idx in range(3)
    ]
    max_delta_v = max(float(max_acceleration_mps2), 0.0) * dt
    acceleration_delta, acceleration_was_limited = limit_vector_norm(
        acceleration_delta, max_delta_v)
    command_velocity = [
        prev_velocity[idx] + acceleration_delta[idx]
        for idx in range(3)
    ]
    command_position = [
        previous[idx] + command_velocity[idx] * dt
        for idx in range(3)
    ]
    return {
        "position": command_position,
        "velocity": command_velocity,
        "velocity_limited": velocity_was_limited,
        "acceleration_limited": acceleration_was_limited,
    }


def limit_cartesian_velocity(
        desired_velocity_mps,
        previous_velocity_mps,
        dt_sec,
        max_velocity_mps,
        max_acceleration_mps2):
    desired = [float(value) for value in desired_velocity_mps]
    previous = [float(value) for value in previous_velocity_mps]
    if len(desired) != 3 or len(previous) != 3:
        raise ValueError("Cartesian velocity vectors must have length 3")

    dt = max(float(dt_sec), 1e-6)
    velocity_limited, velocity_was_limited = limit_vector_norm(
        desired, max_velocity_mps)
    acceleration_delta = [
        velocity_limited[idx] - previous[idx]
        for idx in range(3)
    ]
    max_delta_v = max(float(max_acceleration_mps2), 0.0) * dt
    acceleration_delta, acceleration_was_limited = limit_vector_norm(
        acceleration_delta, max_delta_v)
    command_velocity = [
        previous[idx] + acceleration_delta[idx]
        for idx in range(3)
    ]
    return {
        "velocity": command_velocity,
        "velocity_limited": velocity_was_limited,
        "acceleration_limited": acceleration_was_limited,
    }


def limit_joint_servo_command(
        target_rad,
        measured_rad,
        previous_velocity_radps,
        horizon_sec,
        dt_sec,
        max_velocity_radps,
        max_acceleration_radps2):
    target = [float(value) for value in target_rad]
    measured = [float(value) for value in measured_rad]
    prev_velocity = [float(value) for value in previous_velocity_radps]
    if len(target) != len(measured) or len(target) != len(prev_velocity):
        raise ValueError("Joint command vectors must have matching lengths")

    horizon = max(float(horizon_sec), 1e-6)
    dt = max(float(dt_sec), 1e-6)
    max_velocity = max(float(max_velocity_radps), 0.0)
    max_delta_v = max(float(max_acceleration_radps2), 0.0) * dt

    raw_velocity = [
        (target[idx] - measured[idx]) / horizon
        for idx in range(len(target))
    ]
    velocity_limited = False
    acceleration_limited = False
    command_velocity = []
    for idx, raw_value in enumerate(raw_velocity):
        value = max(-max_velocity, min(max_velocity, raw_value))
        if abs(value - raw_value) > 1e-12:
            velocity_limited = True
        delta_v = value - prev_velocity[idx]
        limited_delta_v = max(-max_delta_v, min(max_delta_v, delta_v))
        if abs(limited_delta_v - delta_v) > 1e-12:
            acceleration_limited = True
        command_velocity.append(prev_velocity[idx] + limited_delta_v)

    command_position = [
        measured[idx] + command_velocity[idx] * horizon
        for idx in range(len(target))
    ]
    return {
        "position": command_position,
        "velocity": command_velocity,
        "velocity_limited": velocity_limited,
        "acceleration_limited": acceleration_limited,
    }


def reference_tracking_error(reference_rad, measured_rad):
    reference = [float(value) for value in reference_rad]
    measured = [float(value) for value in measured_rad]
    if len(reference) != len(measured):
        raise ValueError(
            "Reference/measured vectors must have matching lengths")
    if not reference:
        return 0.0
    return max(
        abs(reference[idx] - measured[idx])
        for idx in range(len(reference))
    )


def should_reset_reference_for_tracking_error(
        reference_rad, measured_rad, threshold_rad):
    threshold = max(float(threshold_rad), 0.0)
    return reference_tracking_error(reference_rad, measured_rad) > threshold


def should_invalidate_reference(stop_reason):
    return str(stop_reason) in ("palm_pose_timeout", "deadman_inactive")


def integrate_joint_reference(
        reference_rad,
        target_rad,
        previous_velocity_radps,
        dt_sec,
        max_velocity_radps,
        max_acceleration_radps2):
    reference = [float(value) for value in reference_rad]
    target = [float(value) for value in target_rad]
    previous_velocity = [float(value) for value in previous_velocity_radps]
    if (len(reference) != len(target) or
            len(reference) != len(previous_velocity)):
        raise ValueError("Joint reference vectors must have matching lengths")

    dt = max(float(dt_sec), 1e-6)
    max_velocity = max(float(max_velocity_radps), 0.0)
    max_delta_v = max(float(max_acceleration_radps2), 0.0) * dt
    next_reference = []
    next_velocity = []
    velocity_limited = False
    acceleration_limited = False

    for idx, q_value in enumerate(reference):
        error = target[idx] - q_value
        if abs(error) <= 1e-12:
            next_reference.append(target[idx])
            next_velocity.append(0.0)
            continue

        desired_velocity = error / dt
        velocity = max(-max_velocity, min(max_velocity, desired_velocity))
        if abs(velocity - desired_velocity) > 1e-12:
            velocity_limited = True

        delta_v = velocity - previous_velocity[idx]
        limited_delta_v = max(-max_delta_v, min(max_delta_v, delta_v))
        if abs(limited_delta_v - delta_v) > 1e-12:
            acceleration_limited = True
        velocity = previous_velocity[idx] + limited_delta_v

        step = velocity * dt
        if abs(step) > abs(error) and step * error > 0.0:
            step = error
            velocity = step / dt
        next_reference.append(q_value + step)
        next_velocity.append(velocity)

    return {
        "position": next_reference,
        "velocity": next_velocity,
        "velocity_limited": velocity_limited,
        "acceleration_limited": acceleration_limited,
    }


def predict_joint_reference_buffer(
        reference_rad,
        target_rad,
        reference_velocity_radps,
        horizon_sec,
        point_dt_sec,
        integration_dt_sec,
        max_velocity_radps,
        max_acceleration_radps2):
    horizon = max(float(horizon_sec), 1e-6)
    point_dt = max(float(point_dt_sec), 1e-6)
    integration_dt = max(float(integration_dt_sec), 1e-6)
    point_count = max(1, int(math.ceil(horizon / point_dt - 1e-12)))
    times = [
        min((idx + 1) * point_dt, horizon)
        for idx in range(point_count)
    ]
    if times[-1] < horizon - 1e-12:
        times.append(horizon)

    q_current = [float(value) for value in reference_rad]
    v_current = [float(value) for value in reference_velocity_radps]
    target = [float(value) for value in target_rad]
    if len(q_current) != len(target) or len(q_current) != len(v_current):
        raise ValueError("Joint buffer vectors must have matching lengths")

    current_time = 0.0
    waypoints = []
    velocities = []
    velocity_limited = False
    acceleration_limited = False

    for sample_time in times:
        while current_time < sample_time - 1e-12:
            dt = min(integration_dt, sample_time - current_time)
            step = integrate_joint_reference(
                q_current,
                target,
                v_current,
                dt,
                max_velocity_radps,
                max_acceleration_radps2)
            q_current = step["position"]
            v_current = step["velocity"]
            velocity_limited = (
                velocity_limited or step["velocity_limited"])
            acceleration_limited = (
                acceleration_limited or step["acceleration_limited"])
            current_time += dt
        waypoints.append(q_current[:])
        velocities.append(v_current[:])

    return {
        "waypoints": waypoints,
        "times": times,
        "velocities": velocities,
        "velocity_limited": velocity_limited,
        "acceleration_limited": acceleration_limited,
    }


def make_joint_trajectory_buffer(
        target_rad,
        measured_rad,
        previous_velocity_radps,
        horizon_sec,
        point_dt_sec,
        max_velocity_radps,
        max_acceleration_radps2):
    return predict_joint_reference_buffer(
        measured_rad,
        target_rad,
        previous_velocity_radps,
        horizon_sec,
        point_dt_sec,
        point_dt_sec,
        max_velocity_radps,
        max_acceleration_radps2)


def limit_joint_velocity_command(
        desired_velocity_radps,
        previous_velocity_radps,
        dt_sec,
        max_velocity_radps,
        max_acceleration_radps2):
    desired = [float(value) for value in desired_velocity_radps]
    previous = [float(value) for value in previous_velocity_radps]
    if len(desired) != len(previous):
        raise ValueError("Joint velocity vectors must have matching lengths")

    dt = max(float(dt_sec), 1e-6)
    max_velocity = max(float(max_velocity_radps), 0.0)
    max_delta_v = max(float(max_acceleration_radps2), 0.0) * dt
    velocity_limited = False
    acceleration_limited = False
    command_velocity = []
    for idx, raw_value in enumerate(desired):
        value = max(-max_velocity, min(max_velocity, raw_value))
        if abs(value - raw_value) > 1e-12:
            velocity_limited = True
        delta_v = value - previous[idx]
        limited_delta_v = max(-max_delta_v, min(max_delta_v, delta_v))
        if abs(limited_delta_v - delta_v) > 1e-12:
            acceleration_limited = True
        command_velocity.append(previous[idx] + limited_delta_v)
    return {
        "velocity": command_velocity,
        "velocity_limited": velocity_limited,
        "acceleration_limited": acceleration_limited,
    }


def is_target_within_workspace(target, workspace_min, workspace_max):
    target_values = [float(value) for value in target]
    minimum = [float(value) for value in workspace_min]
    maximum = [float(value) for value in workspace_max]
    if (len(target_values) != len(minimum) or
            len(target_values) != len(maximum)):
        raise ValueError("Workspace vectors must have matching lengths")
    return all(
        minimum[idx] <= target_values[idx] <= maximum[idx]
        for idx in range(len(target_values))
    )


def workspace_axis_margins(target, workspace_min, workspace_max):
    target_values = [float(value) for value in target]
    minimum = [float(value) for value in workspace_min]
    maximum = [float(value) for value in workspace_max]
    if (len(target_values) != len(minimum) or
            len(target_values) != len(maximum)):
        raise ValueError("Workspace vectors must have matching lengths")
    return [
        min(
            target_values[idx] - minimum[idx],
            maximum[idx] - target_values[idx])
        for idx in range(len(target_values))
    ]


def teleop_servo_workspace_policy(
        mode, target_mapping_mode, control_profile, adapter_requested=False):
    try:
        profile = normalize_control_profile(control_profile)
    except ValueError:
        profile = str(control_profile).strip().lower()
    is_real_relative_teleop = (
        str(mode).strip().lower() == "real" and
        str(target_mapping_mode).strip().lower() == "relative" and
        profile in TELEOP_CONTROL_PROFILES
    )
    if is_real_relative_teleop:
        return {
            "workspace_check_frame": "fk_frame",
            "adapter_applied": False,
        }
    return {
        "workspace_check_frame": "configured_target_frame",
        "adapter_applied": bool(adapter_requested),
    }


def servo_stop_reason(palm_timed_out, joint_state_timed_out, deadman_active):
    if joint_state_timed_out:
        return "joint_state_timeout"
    if palm_timed_out:
        return "palm_pose_timeout"
    if not deadman_active:
        return "deadman_inactive"
    return ""


def ros2_control_rad_to_motor_mrad(
        values_rad,
        initial_position_mrad=AMIR_DRIVER_INITIAL_POSITION_MRAD):
    values = [float(value) for value in values_rad]
    offsets = [float(value) for value in initial_position_mrad]
    if len(values) > len(offsets):
        raise ValueError(
            "Cannot convert "
            f"{len(values)} joints with only {len(offsets)} offsets")
    return [
        values[idx] * 1000.0 - offsets[idx]
        for idx in range(len(values))
    ]


def motor_mrad_to_ros2_control_rad(
        values_mrad,
        initial_position_mrad=AMIR_DRIVER_INITIAL_POSITION_MRAD):
    values = [float(value) for value in values_mrad]
    offsets = [float(value) for value in initial_position_mrad]
    if len(values) > len(offsets):
        raise ValueError(
            "Cannot convert "
            f"{len(values)} joints with only {len(offsets)} offsets")
    return [
        (values[idx] + offsets[idx]) / 1000.0
        for idx in range(len(values))
    ]


def has_timed_out(last_time_sec, now_sec, timeout_sec):
    if last_time_sec is None:
        return True
    timeout = max(float(timeout_sec), 0.0)
    return (float(now_sec) - float(last_time_sec)) > timeout


def should_send_command(mode, dry_run, joint_state_received, palm_timed_out):
    normalized_mode = str(mode).strip().lower()
    if palm_timed_out:
        return False
    if normalized_mode == "real":
        return (not bool(dry_run)) and bool(joint_state_received)
    if normalized_mode == "gazebo":
        return True
    return False
