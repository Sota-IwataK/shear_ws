# Participant request only; ROS Registry remains the identity authority.
string participant_id
string input_writer
string input_session
uint64 input_sequence
string input_clock_domain
uint64 input_timestamp_ns
bool selected
# selected=false requires empty strings and zero; selected=true allows object 0.
string source_id
string session_id
uint64 object_id
