# One atomic canonical TOI decision. selected=false represents no target;
# object_id=-1 is used only by the separate legacy Int32 projection.
string participant_id
string writer_id
string writer_session
uint64 feature_sequence
uint64 decision_sequence
string decision_clock_domain
uint64 decision_timestamp_ns
string receive_clock_domain
uint64 receive_timestamp_ns

bool selected
string source_id
string session_id
uint64 object_id
bool score_valid
float64 score
string reason
uint64 registry_revision
bool target_changed
