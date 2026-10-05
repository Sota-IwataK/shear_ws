# One independently timed and qualified TOI feature. Missing/invalid/fallback
# states are distinct from an actual zero value.
uint8 ACTUAL=0
uint8 MISSING=1
uint8 INVALID=2
uint8 FALLBACK=3
uint8 BOOL=0
uint8 SCALAR=1

string name
uint8 value_type
bool bool_value
float64 scalar_value
uint8 state

bool source_timestamp_valid
string source_clock_domain
uint64 source_timestamp_ns
bool computation_timestamp_valid
string computation_clock_domain
uint64 computation_timestamp_ns

string producer_id
string semantic_id
string authority
string frame_id
