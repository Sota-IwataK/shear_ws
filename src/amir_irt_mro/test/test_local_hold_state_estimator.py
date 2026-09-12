"""P1-03B-01B T1–T10 and boundary/failure regression tests; synthetic only."""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path

import pytest

from amir_irt_mro.local_hold_mock import mock_inputs
from amir_irt_mro.local_hold_state_estimator import (
    HoldConfig, LocalHoldStateEstimator, OptionalLoad, Signal,
)


MS = 1_000_000


def json_safe(value):
    """Preserve invalid numeric test evidence as strings in strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


@pytest.fixture(scope='session')
def trace():
    """Optionally save every injected input/output as a research trace."""
    records = []
    yield records
    if os.environ.get('HOLD_TRACE_PATH'):
        Path(os.environ['HOLD_TRACE_PATH']).write_text(
            ''.join(json.dumps(json_safe(row), allow_nan=False) + '\n' for row in records))


@pytest.fixture
def estimator(trace, request):
    """Record results without wrapping or changing estimator semantics."""
    class RecordedEstimator(LocalHoldStateEstimator):
        def update(self, inputs):
            state = super().update(inputs)
            trace.append({'test': request.node.name,
                          'input': asdict(inputs), 'output': asdict(state)})
            return state
    return RecordedEstimator()


def step(estimator, ms, **kwargs):
    return estimator.update(mock_inputs(ms * MS, **kwargs))


def holding(estimator):
    step(estimator, 0)
    step(estimator, 10, close=True)
    state = step(estimator, 210, close=True)
    assert state.holds_object
    return state


def committed(estimator):
    holding(estimator)
    state = step(estimator, 610, close=True)
    assert state.held_object_id == 3
    return state


def test_t1_open(estimator):
    state = step(estimator, 0)
    assert state.valid and not state.holds_object
    assert state.held_object_id is None


@pytest.mark.parametrize('position,distance', [(1.0, 0.05), (0.5, 1.0), (1.0, 1.0)])
def test_t2_empty_close(estimator, position, distance):
    step(estimator, 0)
    for ms in (10, 210, 410):
        state = step(estimator, ms, close=True, position=position, distance=distance, load=True)
        assert state.valid and not state.holds_object


def test_t3_successful_hold(estimator):
    step(estimator, 0)
    step(estimator, 10, close=True)
    assert not step(estimator, 200, close=True).holds_object
    assert step(estimator, 210, close=True).holds_object


def test_t4_candidate_latch(estimator):
    step(estimator, 0, selected_id=3)
    assert step(estimator, 10, close=True).grasp_candidate_object_id == 3
    state = step(estimator, 110, close=True, selected_id=4)
    assert state.grasp_candidate_object_id == 3
    state = step(estimator, 210, close=True, selected_id=4)
    assert state.holds_object and state.grasp_candidate_object_id == 3


def test_t5_commit(estimator):
    holding(estimator)
    assert step(estimator, 600, close=True, selected_id=4).held_object_id is None
    state = step(estimator, 610, close=True, selected_id=4)
    assert state.held_object_id == 3


def test_t6_close_release(estimator):
    committed(estimator)
    assert step(estimator, 620).holds_object
    assert step(estimator, 719).holds_object
    state = step(estimator, 720)
    assert state.valid and not state.holds_object
    assert state.held_object_id is None and state.grasp_candidate_object_id is None
    assert step(estimator, 730, close=True, selected_id=4).grasp_candidate_object_id == 4


def test_t7_bottle_separation(estimator):
    committed(estimator)
    step(estimator, 620, close=True, distance=1.0)
    assert step(estimator, 919, close=True, distance=1.0).holds_object
    state = step(estimator, 920, close=True, distance=1.0)
    assert state.valid and not state.holds_object
    assert state.held_object_id is None and state.grasp_candidate_object_id is None


def test_t8_transient_rejection(estimator):
    step(estimator, 0)
    for start in (10, 210, 410, 610):
        assert not step(estimator, start, close=True).holds_object
        assert not step(estimator, start + 190, close=True).holds_object
        assert not step(estimator, start + 199, close=True, distance=1.0).holds_object


@pytest.mark.parametrize('name', [
    'close_intent', 'gripper_actual_position', 'bottle_ee_distance', 'selected_object_id'])
@pytest.mark.parametrize('failure', [
    'stale', 'invalid', 'missing', 'future', 'bad_stamp', 'regression'])
def test_t9_required_signal_failure(estimator, name, failure):
    committed(estimator)
    inputs = mock_inputs(620 * MS, close=True)
    signal = getattr(inputs, name)
    updates = {
        'stale': replace(signal, timestamp_ns=0),
        'invalid': replace(signal, valid=False),
        'missing': None,
        'future': replace(signal, timestamp_ns=621 * MS),
        'bad_stamp': replace(signal, timestamp_ns=float('nan')),
        'regression': replace(signal, timestamp_ns=609 * MS),
    }
    state = estimator.update(replace(inputs, **{name: updates[failure]}))
    assert not state.valid and not state.holds_object
    assert state.held_object_id is None and state.grasp_candidate_object_id is None
    assert name in state.invalid_reason


@pytest.mark.parametrize('name', ['gripper_actual_position', 'bottle_ee_distance'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), None, True])
def test_t9_nonfinite(estimator, name, value):
    committed(estimator)
    inputs = mock_inputs(620 * MS, close=True)
    state = estimator.update(replace(inputs, **{name: Signal(value, True, 620 * MS)}))
    assert not state.valid and not state.holds_object


@pytest.mark.parametrize('stamp', [600 * MS, -1, float('nan'), float('inf'), True])
def test_t9_clock_failure(estimator, stamp):
    committed(estimator)
    state = estimator.update(replace(mock_inputs(620 * MS, close=True), timestamp_ns=stamp))
    assert not state.valid and not state.holds_object
    assert state.held_object_id is None
    assert 'timestamp' in state.invalid_reason


def test_t10_load_unavailable(estimator):
    state = holding(estimator)
    assert state.valid and state.holds_object
    assert not state.load_detected_available and state.load_detected is None


@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('kind,duration', [
    ('hold', 200), ('commit', 400), ('close', 100), ('bottle', 300)])
def test_exact_nanosecond_boundaries(estimator, delta, kind, duration):
    if kind == 'hold':
        step(estimator, 0)
        start = 10
        step(estimator, start, close=True)
    elif kind == 'commit':
        holding(estimator)
        start = 210
    else:
        committed(estimator)
        start = 620
        step(estimator, start, close=kind != 'close', distance=1.0 if kind == 'bottle' else 0.05)
    state = estimator.update(mock_inputs(
        (start + duration) * MS + delta, close=kind != 'close',
        distance=1.0 if kind == 'bottle' else 0.05))
    if kind == 'hold':
        assert state.holds_object == (delta >= 0)
    elif kind == 'commit':
        assert (state.held_object_id == 3) == (delta >= 0)
    else:
        assert state.holds_object == (delta < 0)


@pytest.mark.parametrize('kind', ['close', 'bottle'])
def test_held_transient_does_not_relatch_or_release(estimator, kind):
    committed(estimator)
    step(estimator, 620, close=kind != 'close', distance=1.0 if kind == 'bottle' else 0.05)
    state = step(estimator, 700, close=True, selected_id=4)
    assert state.holds_object and state.held_object_id == 3
    assert state.grasp_candidate_object_id == 3
    assert step(estimator, 950, close=True, selected_id=4).holds_object


def test_optional_invalid_load_does_not_invalidate(estimator):
    committed(estimator)
    for load in (OptionalLoad(True, None), OptionalLoad(True, Signal(True, False, 620 * MS)),
                 OptionalLoad(True, Signal(float('nan'), True, 620 * MS)),
                 OptionalLoad(True, Signal(True, True, 621 * MS))):
        state = estimator.update(replace(mock_inputs(620 * MS, close=True), load_detected=load))
        assert state.valid and state.holds_object and not state.load_detected_available


def test_update_gap_cannot_confirm_unsampled_hold(estimator):
    holding(estimator)
    state = step(estimator, 1000, close=True)
    assert not state.valid and not state.holds_object and 'update_gap' in state.invalid_reason


def test_recovery_and_restart_need_observed_edge_for_id(estimator):
    committed(estimator)
    estimator.update(replace(mock_inputs(620 * MS), bottle_ee_distance=None))
    state = step(estimator, 630, close=True, selected_id=4)
    assert state.valid and not state.holds_object and state.grasp_candidate_object_id is None
    state = step(estimator, 830, close=True, selected_id=4)
    assert state.holds_object and state.held_object_id is None
    fresh = LocalHoldStateEstimator()
    assert fresh.update(mock_inputs(830 * MS, close=True)).held_object_id is None
    step(estimator, 840)
    step(estimator, 940)
    assert step(estimator, 950, close=True, selected_id=4).grasp_candidate_object_id == 4


def test_intermediate_loss_after_hold_uses_frozen_release_rules(estimator):
    committed(estimator)
    state = step(estimator, 620, close=True, position=1.0)
    assert not state.main_condition and not state.intermediate_stop
    assert state.holds_object  # Only the two specified release debounces apply.


@pytest.mark.parametrize('position,distance', [(0.2, 0.1), (0.8, 0.1)])
def test_inclusive_evidence_bounds(estimator, position, distance):
    state = step(estimator, 0, close=True, position=position, distance=distance)
    assert state.intermediate_stop and state.bottle_near_ee


def test_configuration_changes_behavior():
    estimator = LocalHoldStateEstimator(HoldConfig(
        hold_confirm_sec=0.01, held_id_commit_sec=0.02,
        close_release_sec=0.03, bottle_release_sec=0.04,
        gripper_intermediate_min=10, gripper_intermediate_max=20, bottle_near_threshold=2))
    step(estimator, 0, position=15, distance=1)
    step(estimator, 1, close=True, position=15, distance=1)
    assert step(estimator, 11, close=True, position=15, distance=1).holds_object
    assert step(estimator, 31, close=True, position=15, distance=1).held_object_id == 3
    step(estimator, 32, position=15, distance=1)
    assert not step(estimator, 62, position=15, distance=1).holds_object


@pytest.mark.parametrize('kwargs', [
    {'hold_confirm_sec': -1}, {'bottle_near_threshold': float('nan')},
    {'gripper_intermediate_min': 2, 'gripper_intermediate_max': 1},
    {'max_update_gap_sec': 0}, {'distance_max_age_sec': float('inf')},
    {'close_release_sec': 0.0000000001}])
def test_bad_config_rejected(kwargs):
    with pytest.raises(ValueError):
        HoldConfig(**kwargs)


def test_freshness_exact_boundary_and_repeated_clock(estimator):
    committed(estimator)
    inputs = mock_inputs(620 * MS, close=True)
    # Accepted source is 610 ms; exactly 500 ms old is still usable.
    inputs = replace(inputs, timestamp_ns=1110 * MS)
    inputs = replace(inputs, gripper_actual_position=Signal(0.5, True, 610 * MS))
    # Other signals must also be fresh, and update gap remains 500 ms.
    state = estimator.update(inputs)
    assert state.valid and state.holds_object
    assert estimator.update(inputs) == state
    state = estimator.update(replace(inputs, timestamp_ns=1110 * MS + 1))
    assert not state.valid and 'gripper_actual_position' in state.stale_inputs


def test_valid_no_selection_and_bad_id(estimator):
    step(estimator, 0, selected_id=None)
    step(estimator, 10, close=True, selected_id=None)
    state = step(estimator, 210, close=True, selected_id=None)
    assert state.valid and state.holds_object and state.grasp_candidate_object_id is None
    assert not step(estimator, 220, close=True, selected_id=-1).valid


def test_transition_time_only_changes_on_semantic_transition(estimator):
    state = holding(estimator)
    assert state.last_transition_time_ns == 210 * MS
    assert step(estimator, 220, close=True).last_transition_time_ns == 210 * MS
    state = step(estimator, 610, close=True)
    assert state.last_transition_time_ns == 610 * MS


@pytest.mark.parametrize('name', [
    'close_intent', 'gripper_actual_position', 'bottle_ee_distance', 'selected_object_id'])
def test_stale_is_reported_for_each_required_input(estimator, name):
    inputs = mock_inputs(501 * MS, close=True)
    inputs = replace(inputs, **{name: replace(getattr(inputs, name), timestamp_ns=0)})
    state = estimator.update(inputs)
    assert not state.valid and not state.holds_object
    assert name in state.stale_inputs and f'{name}:stale' in state.invalid_reason


def test_optional_stale_load_is_unavailable(estimator):
    inputs = mock_inputs(501 * MS, close=True)
    state = estimator.update(replace(
        inputs, load_detected=OptionalLoad(True, Signal(True, True, 0))))
    assert state.valid and not state.load_detected_available
    assert state.load_invalid_reason == 'stale'
    assert 'load' in state.stale_inputs


def test_pending_candidate_clears_on_open(estimator):
    step(estimator, 0)
    step(estimator, 10, close=True)
    assert step(estimator, 50).grasp_candidate_object_id is None
    assert step(estimator, 60, close=True, selected_id=4).grasp_candidate_object_id == 4


def test_negative_distance_is_invalid(estimator):
    assert not step(estimator, 0, distance=-0.01).valid


def test_clock_rollback_recovery_does_not_restore_timer(estimator):
    committed(estimator)
    assert not step(estimator, 0, close=True).valid
    state = step(estimator, 10, close=True)
    assert state.valid and not state.holds_object
    assert not step(estimator, 209, close=True).holds_object
    assert step(estimator, 210, close=True).holds_object


def test_zero_duration_config_is_immediate_after_valid_edge():
    estimator = LocalHoldStateEstimator(HoldConfig(
        hold_confirm_sec=0, held_id_commit_sec=0, close_release_sec=0))
    step(estimator, 0)
    state = step(estimator, 1, close=True)
    assert state.holds_object and state.held_object_id == 3
    assert not step(estimator, 2).holds_object
