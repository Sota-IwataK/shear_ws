"""Transport-independent hold semantics; all times are injected, never wall time."""

from dataclasses import dataclass, field
from decimal import Decimal
import math
from typing import Generic, Optional, Tuple, TypeVar


T = TypeVar('T')


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _ns(seconds):
    return int(Decimal(str(seconds)) * 1_000_000_000)


@dataclass(frozen=True)
class HoldConfig:
    """Threshold defaults are synthetic only: requires hardware calibration."""

    hold_confirm_sec: float = 0.20
    close_release_sec: float = 0.10
    bottle_release_sec: float = 0.30
    held_id_commit_sec: float = 0.40
    gripper_intermediate_min: float = 0.2
    gripper_intermediate_max: float = 0.8
    bottle_near_threshold: float = 0.1
    close_max_age_sec: float = 0.5
    gripper_max_age_sec: float = 0.5
    distance_max_age_sec: float = 0.5
    selected_id_max_age_sec: float = 0.5
    load_max_age_sec: float = 0.5
    max_update_gap_sec: float = 0.5

    def __post_init__(self):
        """Reject unusable configuration instead of silently repairing it."""
        for name, value in vars(self).items():
            if not _number(value):
                raise ValueError(f'{name}: finite numeric value required')
            if name.endswith('_sec'):
                if value < 0 or Decimal(str(value)) * 1_000_000_000 != _ns(value):
                    raise ValueError(f'{name}: nonnegative whole nanoseconds required')
                if ('age' in name or 'gap' in name) and value == 0:
                    raise ValueError(f'{name}: positive duration required')
        if self.gripper_intermediate_min > self.gripper_intermediate_max:
            raise ValueError('intermediate interval is reversed')
        if self.bottle_near_threshold < 0:
            raise ValueError('distance threshold must be nonnegative')


@dataclass(frozen=True)
class Signal(Generic[T]):
    """One normalized observation in the same clock domain as the update."""

    value: T
    valid: bool
    timestamp_ns: int


@dataclass(frozen=True)
class OptionalLoad:
    """Availability is distinct from a measured false load."""

    available: bool = False
    signal: Optional[Signal[bool]] = None


@dataclass(frozen=True)
class NormalizedHoldInputs:
    """Adapter boundary; distance must already use compatible geometry/units."""

    timestamp_ns: int
    close_intent: Optional[Signal[bool]] = None
    gripper_actual_position: Optional[Signal[float]] = None
    bottle_ee_distance: Optional[Signal[float]] = None
    selected_object_id: Optional[Signal[Optional[int]]] = None
    load_detected: OptionalLoad = field(default_factory=OptionalLoad)


@dataclass(frozen=True)
class LocalHoldState:
    """Immutable result suitable for dataclasses.asdict / structured logging."""

    timestamp_ns: Optional[int]
    valid: bool
    holds_object: bool
    grasp_candidate_object_id: Optional[int]
    held_object_id: Optional[int]
    close_intent: Optional[bool]
    intermediate_stop: Optional[bool]
    bottle_near_ee: Optional[bool]
    load_detected_available: bool
    load_detected: Optional[bool]
    main_condition: bool
    bottle_ee_distance: Optional[float]
    gripper_actual_position: Optional[float]
    last_transition_time_ns: Optional[int]
    invalid_reason: str
    load_invalid_reason: str
    input_ages_ns: Tuple[Tuple[str, Optional[int]], ...]
    stale_inputs: Tuple[str, ...]


class LocalHoldStateEstimator:
    """Estimate hold from sampled evidence without producing control commands."""

    def __init__(self, config=None):
        """Start unknown; an observed open-to-close edge is needed for ID latch."""
        self.config = config if config is not None else HoldConfig()
        self._last_time = None
        self._source_times = {}
        self._last_signature = None
        self._transition = None
        self._reset_attempt()

    def _reset_attempt(self):
        self._holding = False
        self._candidate = None
        self._held = None
        self._previous_close = None
        self._main_since = None
        self._hold_since = None
        self._open_since = None
        self._far_since = None

    def _signal_error(self, name, signal, now, age_limit, kind):
        if signal is None:
            return 'missing', None
        if type(signal.timestamp_ns) is not int or signal.timestamp_ns < 0:
            return 'timestamp_invalid', None
        age = now - signal.timestamp_ns
        if age < 0:
            return 'timestamp_future', age
        if signal.timestamp_ns < self._source_times.get(name, 0):
            return 'timestamp_regression', age
        if signal.valid is not True:
            return 'invalid', age
        if age > _ns(age_limit):
            return 'stale', age
        value = signal.value
        if kind == 'bool' and type(value) is not bool:
            return 'value_invalid', age
        if kind == 'number' and not _number(value):
            return 'value_nonfinite_or_invalid', age
        if kind == 'distance' and (not _number(value) or value < 0):
            return 'distance_invalid', age
        if kind == 'id' and value is not None and (type(value) is not int or value < 0):
            return 'id_invalid', age
        return '', age

    @staticmethod
    def _since(condition, since, now):
        return (now if since is None else since) if condition else None

    @staticmethod
    def _elapsed(since, now, seconds):
        return since is not None and now - since >= _ns(seconds)

    def update(self, inputs):
        """Advance at injected nanoseconds; invalid input immediately clears hold/IDs."""
        now = inputs.timestamp_ns
        cfg = self.config
        time_ok = type(now) is int and now >= 0
        errors = []
        if not time_ok:
            errors.append('timestamp_invalid')
        elif self._last_time is not None:
            if now < self._last_time:
                errors.append('timestamp_regression')
            elif now - self._last_time > _ns(cfg.max_update_gap_sec):
                errors.append('update_gap')
        fields = (
            ('close_intent', inputs.close_intent, cfg.close_max_age_sec, 'bool'),
            ('gripper_actual_position', inputs.gripper_actual_position,
             cfg.gripper_max_age_sec, 'number'),
            ('bottle_ee_distance', inputs.bottle_ee_distance,
             cfg.distance_max_age_sec, 'distance'),
            ('selected_object_id', inputs.selected_object_id,
             cfg.selected_id_max_age_sec, 'id'),
        )
        ages, stale, accepted = [], [], []
        for name, signal, limit, kind in fields:
            error, age = (self._signal_error(name, signal, now, limit, kind)
                          if time_ok else ('clock_invalid', None))
            ages.append((name, age))
            if error:
                errors.append(f'{name}:{error}')
                if error == 'stale':
                    stale.append(name)
            else:
                accepted.append((name, signal.timestamp_ns))
        load = inputs.load_detected
        load_error, load_age = 'unavailable', None
        if load.available and time_ok:
            load_error, load_age = self._signal_error(
                'load', load.signal, now, cfg.load_max_age_sec, 'bool')
        ages.append(('load', load_age))
        if load_error == 'stale':
            stale.append('load')
        load_available = load.available is True and not load_error
        load_value = load.signal.value if load_available else None
        valid = not errors
        close = intermediate = near = position = distance = None
        main = False
        if not valid:
            self._reset_attempt()
            self._source_times.clear()
        else:
            self._source_times.update(accepted)
            if load_available:
                self._source_times['load'] = load.signal.timestamp_ns
            position = inputs.gripper_actual_position.value
            distance = inputs.bottle_ee_distance.value
            close = inputs.close_intent.value
            intermediate = cfg.gripper_intermediate_min <= position <= cfg.gripper_intermediate_max
            near = distance <= cfg.bottle_near_threshold
            main = close and intermediate and near
            if close and self._previous_close is False and not self._holding:
                self._candidate = inputs.selected_object_id.value
            self._previous_close = close
            self._main_since = self._since(main, self._main_since, now)
            self._open_since = self._since(not close, self._open_since, now)
            self._far_since = self._since(not near, self._far_since, now)
            if self._holding:
                release = (
                    self._elapsed(self._open_since, now, cfg.close_release_sec)
                    or self._elapsed(self._far_since, now, cfg.bottle_release_sec))
                if release:
                    self._holding = False
                    self._held = self._candidate = None
                    self._hold_since = self._main_since = None
            elif self._elapsed(self._main_since, now, cfg.hold_confirm_sec):
                self._holding = True
                self._hold_since = now
            if not self._holding and not close:
                self._candidate = None
            if self._holding and self._elapsed(self._hold_since, now, cfg.held_id_commit_sec):
                self._held = self._candidate
        # A clock fault starts a fresh epoch; it never restores the old attempt.
        self._last_time = now if time_ok else None
        signature = (valid, self._holding, self._candidate, self._held)
        if signature != self._last_signature:
            self._transition = now if time_ok else None
        self._last_signature = signature
        return LocalHoldState(
            now if time_ok else None, valid, self._holding, self._candidate, self._held,
            close, intermediate, near, load_available, load_value, main,
            distance, position, self._transition, ';'.join(errors), load_error,
            tuple(ages), tuple(stale))
