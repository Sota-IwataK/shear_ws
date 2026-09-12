# Estimator-generated semantics, NOT physical ground truth or ownership.
# Identity uses SharedMRParticipantId names, not an invented numeric enum.
string participant_id
# New UUID per publisher lifetime; receiver must explicitly bind a session.
string session_id
uint64 sequence
# Nanoseconds in the source clock domain; never subtract another PC's clock.
bool source_timestamp_valid
uint64 source_timestamp_ns
bool valid
bool holds_object
bool grasp_candidate_valid
int64 grasp_candidate_object_id
bool held_object_valid
int64 held_object_id
bool close_intent_valid
bool close_intent
bool intermediate_stop_valid
bool intermediate_stop
bool bottle_near_ee_valid
bool bottle_near_ee
bool load_detected_available
bool load_detected_valid
bool load_detected
bool main_condition
bool bottle_ee_distance_valid
float64 bottle_ee_distance
bool gripper_actual_position_valid
float64 gripper_actual_position
bool last_transition_time_valid
uint64 last_transition_time_ns
# Research diagnostics; do not mirror these strings into Photon Networked State.
string invalid_reason
string load_invalid_reason
# Parallel arrays, same length, age may be negative for future-stamp failures.
string[] input_age_names
bool[] input_age_valid
int64[] input_ages_ns
string[] stale_inputs
