"""Lossless supported-range mapping without a ROS node, cache or clock."""

from uuid import uuid4

from amir_irt_mro.local_hold_state_estimator import LocalHoldState
from share_semantic_interfaces.msg import LocalHoldState as RosLocalHoldState


PARTICIPANT_NAMES = ('Unassigned', 'User1', 'User2', 'User3')
UINT64_MAX = (1 << 64) - 1
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
OPTIONALS = (
    ('timestamp_ns', 'source_timestamp_ns', 'source_timestamp_valid'),
    ('grasp_candidate_object_id', 'grasp_candidate_object_id', 'grasp_candidate_valid'),
    ('held_object_id', 'held_object_id', 'held_object_valid'),
    ('close_intent', 'close_intent', 'close_intent_valid'),
    ('intermediate_stop', 'intermediate_stop', 'intermediate_stop_valid'),
    ('bottle_near_ee', 'bottle_near_ee', 'bottle_near_ee_valid'),
    ('load_detected', 'load_detected', 'load_detected_valid'),
    ('bottle_ee_distance', 'bottle_ee_distance', 'bottle_ee_distance_valid'),
    ('gripper_actual_position', 'gripper_actual_position', 'gripper_actual_position_valid'),
    ('last_transition_time_ns', 'last_transition_time_ns', 'last_transition_time_valid'),
)
DIRECT = ('valid', 'holds_object', 'load_detected_available', 'main_condition',
          'invalid_reason', 'load_invalid_reason')


def validate_participant(participant):
    """Keep the user's enum names and reject unassigned senders."""
    if participant not in PARTICIPANT_NAMES[1:]:
        raise ValueError('participant_id must be User1, User2 or User3')


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('integer outside the lossless ROS transport range')
    return value


def to_message(state, participant_id, session_id, sequence):
    """Map a snapshot without correcting validity or reviving cached data."""
    validate_participant(participant_id)
    if not isinstance(session_id, str) or not session_id:
        raise ValueError('nonempty session_id required')
    _integer(sequence, 1, UINT64_MAX)
    msg = RosLocalHoldState()
    msg.participant_id, msg.session_id, msg.sequence = participant_id, session_id, sequence
    for name in DIRECT:
        setattr(msg, name, getattr(state, name))
    for source, target, flag in OPTIONALS:
        value = getattr(state, source)
        setattr(msg, flag, value is not None)
        if value is None:
            continue
        if source.endswith('_ns'):
            value = _integer(value, 0, UINT64_MAX)
        elif source.endswith('_object_id'):
            value = _integer(value, 0, INT64_MAX)
        elif source in ('bottle_ee_distance', 'gripper_actual_position'):
            # Core accepts int too; reject ints that float64 cannot preserve.
            if type(value) is int and int(float(value)) != value:
                raise ValueError('numeric value not exactly representable in float64')
            value = float(value)
        setattr(msg, target, value)
    msg.input_age_names = [name for name, _ in state.input_ages_ns]
    msg.input_age_valid = [age is not None for _, age in state.input_ages_ns]
    msg.input_ages_ns = [
        0 if age is None else _integer(age, INT64_MIN, INT64_MAX)
        for _, age in state.input_ages_ns]
    msg.stale_inputs = list(state.stale_inputs)
    return msg


def from_message(msg):
    """Recover Core state, preserving None separately from false/zero."""
    validate_participant(msg.participant_id)
    if not msg.session_id:
        raise ValueError('empty session_id')
    _integer(msg.sequence, 1, UINT64_MAX)
    if len({len(msg.input_age_names), len(msg.input_age_valid), len(msg.input_ages_ns)}) != 1:
        raise ValueError('input age arrays differ in length')
    values = {name: getattr(msg, name) for name in DIRECT}
    values.update({source: getattr(msg, target) if getattr(msg, flag) else None
                   for source, target, flag in OPTIONALS})
    values['input_ages_ns'] = tuple(
        (name, age if valid else None) for name, age, valid in
        zip(msg.input_age_names, msg.input_ages_ns, msg.input_age_valid))
    values['stale_inputs'] = tuple(msg.stale_inputs)
    return LocalHoldState(**values)


class SemanticStream:
    """One publisher lifetime; never wrap sequence into an apparently old update."""

    def __init__(self, participant_id):
        validate_participant(participant_id)
        self.participant_id = participant_id
        self.session_id = str(uuid4())
        self.sequence = 0

    def encode(self, state):
        """Consume a sequence only after successful conversion."""
        if self.sequence == UINT64_MAX:
            raise OverflowError('sequence exhausted; start a new publisher session')
        msg = to_message(state, self.participant_id, self.session_id, self.sequence + 1)
        self.sequence += 1
        return msg
