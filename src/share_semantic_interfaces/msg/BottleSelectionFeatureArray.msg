# One atomic scoring cycle from one participant and one writer lifetime.
# feature_sequence orders feature snapshots; it is not an observation sequence
# or a timestamp.
string participant_id
string writer_id
string writer_session
uint64 feature_sequence
string assembled_clock_domain
uint64 assembled_timestamp_ns

BottleSelectionFeature[] features
