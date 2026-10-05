# Automatic TOI candidate only. Never a Human-confirmed selection or clear request.
string participant_id
string writer_id
string writer_session
uint64 candidate_sequence
string candidate_clock_domain
uint64 candidate_timestamp_ns
bool candidate_valid
# No candidate: empty/empty/0. Object 0 is valid when candidate_valid=true.
string source_id
string session_id
uint64 object_id
uint64 registry_revision
uint64 feature_sequence
bool score_valid
float64 score
string reason
bool target_changed
# Exact Registry observation backing a valid candidate; absent on invalidation.
bool observation_valid
uint64 observation_sequence
string observation_clock_domain
uint64 observation_timestamp_ns
