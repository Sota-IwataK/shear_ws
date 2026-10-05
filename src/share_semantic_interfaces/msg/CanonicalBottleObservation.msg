# Identity transport only. This message deliberately carries no pose and does
# not certify camera-to-Unity or camera-to-robot spatial authority.
uint8 OBSERVED=0
uint8 LOST=1
uint8 REMOVED=2

string source_id
string session_id
uint64 object_id
uint8 lifecycle
uint64 observation_sequence
builtin_interfaces/Time observation_stamp
string observation_clock_domain
string frame_id
