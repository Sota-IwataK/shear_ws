# Full tracker identity and the exact Registry observation used by the feature
# producer. Numeric object_id alone has no canonical meaning.
string source_id
string session_id
uint64 object_id
uint64 observation_sequence
string observation_clock_domain
uint64 observation_timestamp_ns

BottleFeatureSample[] samples
