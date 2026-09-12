"""Synthetic adapter: explicit time and mock units, no ROS or sleeps."""

from .local_hold_state_estimator import NormalizedHoldInputs, OptionalLoad, Signal


def mock_inputs(timestamp_ns, close=False, position=0.5, distance=0.05,
                selected_id=3, load=None):
    """Create fresh synthetic inputs; thresholds require hardware calibration."""
    def signal(value):
        return Signal(value=value, valid=True, timestamp_ns=timestamp_ns)

    return NormalizedHoldInputs(
        timestamp_ns=timestamp_ns,
        close_intent=signal(close),
        gripper_actual_position=signal(position),
        bottle_ee_distance=signal(distance),
        selected_object_id=signal(selected_id),
        load_detected=OptionalLoad(load is not None, signal(load) if load is not None else None))
