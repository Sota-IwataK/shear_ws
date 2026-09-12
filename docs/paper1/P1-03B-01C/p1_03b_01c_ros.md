# P1-03B-01C ROS

Result: **P1-03B-01C-ROS Software PASS**.
Two packages built successfully. Direct pytest: 28 passed. Colcon including
the two-pair local DDS integration: **29 tests, 0 errors, 0 failures, 0 skipped**.

Hardware PASS: **NO**. This is a software semantic transport boundary, not
hardware binding, Unity integration, or physical ground truth.

## Architecture

NormalizedHoldInputs -> frozen LocalHoldStateEstimator -> LocalHoldState ->
mapper -> share_semantic_interfaces/msg/LocalHoldState -> ROS-TCP -> future Unity.

Pre-implementation audit: root shear_ws, HEAD
df29960a78ebdb8c20d61764f2183383771afcda. The existing 15 staged 01B additions
were captured as SHA256 hashes (index and working tree) before any implementation.
No unstaged tracked edits existed. Existing Python/ament_python/pytest conventions
were reused. No PairSemanticState or semantic interface package was found.
amir_interfaces contains driver commands/sensors, so two small dedicated packages
were added without modifying its definitions or the existing control package.

The mapper imports the generated ROS class but requires no node, executor or clock.
The publisher accepts Core NormalizedHoldInputs through publish_inputs(inputs),
with optional HoldConfig injection on construction. There are **no subscribers**.
Future raw-input binding must be a separate adapter, with its own source audit.
The only executable is explicitly named local_hold_semantic_mock; its periodic
synthetic inputs are never a hardware measurement or production calibration.

## Semantic Transport Contract

Message: `share_semantic_interfaces/msg/LocalHoldState`.
Relative topic: `local_hold_state`. ROS namespace/remapping provides routing.
QoS: reliable, volatile, depth 1 (latest semantic state, not a complete event log).
The generated CDR message is tested for serialization and deserialization.

The local ROS-TCP-Endpoint source resolves `package/class` via dynamic Python
import, so its registration name is `share_semantic_interfaces/LocalHoldState`.
This differs from the ROS CLI spelling `share_semantic_interfaces/msg/LocalHoldState`.
Unity must generate/import the corresponding C# message and register the resolved
per-user topic. No Unity code or ROS-TCP configuration is changed in this task.

## Message Fields

| Group | Fields / meaning |
|---|---|
| Routing | participant_id string, session_id string, sequence uint64 |
| Source time | source_timestamp_valid, source_timestamp_ns uint64 |
| Hold | valid, holds_object |
| Candidate | grasp_candidate_valid, grasp_candidate_object_id int64 |
| Committed ID | held_object_valid, held_object_id int64 |
| Evidence | close_intent, intermediate_stop, bottle_near_ee, load_detected, each with a valid flag |
| Load | load_detected_available is separate from load_detected_valid/value |
| Main | main_condition |
| Numeric debug | bottle_ee_distance, gripper_actual_position float64, each with valid flag |
| Transition | last_transition_time_valid, last_transition_time_ns uint64 |
| Reasons | invalid_reason, load_invalid_reason strings |
| Input ages | parallel input_age_names, input_age_valid, input_ages_ns arrays |
| Staleness | stale_inputs string array |

Optional Core values use explicit presence flags, preserving None vs false vs 0.
Unused scalar slots stay at message defaults and MUST NOT be read when absent.
All Core output fields round-trip; envelope routing metadata is retained in the
ROS message. source time/transition fit uint64 ns, IDs fit nonnegative int64,
signed ages fit int64. Out-of-range Core integers are rejected explicitly rather
than wrapped, clamped, or silently truncated. Non-exact int-to-float conversion
for numeric evidence is rejected. This is the supported transport range, not
a change to the unrestricted Python Core API.

valid=false passes through unchanged, along with holds=false and cleared IDs
from the frozen Core. Mapper has no previous-state cache and never upgrades valid.
Research strings are for diagnostics; future Photon Networked State should use
compact semantic fields, not mirror the full debug payload.

## Timestamp / Sequence Policy

Core timestamp_ns is preserved exactly; no conversion to receive time or restamping.
Source timestamp belongs to the source process clock domain. The mock uses local
monotonic_ns. Never subtract another PC's timestamp or compare two publishers'
source times. Unity separately records receive time / Fusion tick.

sequence starts at 1 and strictly increases for each successfully encoded snapshot.
Gaps are allowed (QoS/delivery); encode failures do not consume a sequence.
At uint64 maximum the stream raises OverflowError instead of wrapping.
Each publisher creates a UUID session_id, stable for its lifetime and renewed
on restart. No existing session_id design was found in the audited repository.

Receiver contract for future Unity:

1. Bind expected participant name and resolved topic. Reject cross-pair messages.
2. Explicitly bind session_id on connection/reconnection. Reset sequence tracking
   only when accepting that new session; delayed packets from old sessions must
   not switch the active session back.
3. Accept only sequence > last accepted sequence for that bound session.
4. Apply invalid immediately; never recover a cached true or infer ownership.
5. Detect transport silence using receiver-local time/tick and mark unknown.

No autonomous stale heartbeat is provided by the injectable publisher class.
Its caller must invoke the Core periodically even with missing/stale inputs;
the receiving application must also expire state on disconnect/silence. Neither
delivery nor execution scheduling is inferred from a previous holds=true message.

## Pair Identity

User-provided SharedMRParticipantId names are preserved literally:
Unassigned, User1, User2, User3. Numeric enum values are **not** assigned here.
Unity source was not present in this workspace, so its actual enum declaration
and numeric serialization remain to be verified in the Unity integration step.

Unassigned is recognized as the unconfigured state and rejected for publishing.
Required startup parameter participant_id selects User1/User2/User3. It is read-only
for the node lifetime so a parameter edit cannot silently move an existing stream
to another pair. Namespace is independent; receivers must validate both routing
and identity rather than trust namespace text alone. User1/User2 tests use separate
namespaces and remapped output names. No robot ID or alternate participant scheme
is introduced.

Example synthetic run after build (publishes semantic state only):

```bash
ros2 run share_semantic_transport local_hold_semantic_mock --ros-args \
  -p participant_id:=User1 -r __ns:=/share/users/user_1
# Change participant_id and namespace together for another pair.
# Optional topic remap: -r local_hold_state:=hold_semantics
```

## Mock Integration Tests

R1 Open, R2 successful Hold, R3 candidate latch, R4 0.40 s commit, R5 release,
R6 invalid/stale, R7 unavailable load, R8 monotonic sequence, R9 preserved source
timestamp, R10 participant configuration, R11 CDR round-trip, R12 no raw binding.
Additional checks cover uint wrap prevention, new-session reset, unknown identity,
range rejection and malformed parallel arrays.

| Test | Result |
|---|---|
| R1 Open | PASS |
| R2 Successful Hold | PASS |
| R3 Candidate latch | PASS |
| R4 Commit | PASS |
| R5 Release | PASS |
| R6 Invalid / stale | PASS |
| R7 Load unavailable | PASS |
| R8 Sequence | PASS |
| R9 Source timestamp | PASS |
| R10 User1 / User2 identity | PASS |
| R11 Generated message CDR round-trip | PASS |
| R12 No raw binding / actuator publishers | PASS |

DDS integration drives two real ROS publisher nodes with artificial timestamps,
checks identity/namespace/remap isolation, observes Hold/commit/release messages,
and publishes no raw inputs or actuator commands. Only DDS discovery/receipt
uses real-time deadlines; all Core debounce inputs use injected times.

## Build/Test

Build uses the frozen 01B install as underlay. If starting from a new checkout,
first build the original package using the documented 01B procedure. Never source
an older amir_irt_mro installation lacking the frozen Core.

```bash
source /opt/ros/humble/setup.bash
source install/p1_03b_01b/local_setup.bash
colcon --log-base log/p1_03b_01c build \
  --base-paths src/share_semantic_interfaces src/share_semantic_transport \
  --build-base build/p1_03b_01c --install-base install/p1_03b_01c \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/p1_03b_01c/local_setup.bash
colcon --log-base log/p1_03b_01c test \
  --base-paths src/share_semantic_interfaces src/share_semantic_transport \
  --build-base build/p1_03b_01c --install-base install/p1_03b_01c
colcon test-result --test-result-base build/p1_03b_01c --verbose
python3 -m pytest src/share_semantic_transport/test/test_transport.py -q
```

For DDS tests, use a local-only test discovery context, not the robot runtime.
Test-specific ROS environment settings are shell configuration and are not
embedded into production code. Results are stored in evidence/ in this directory.

The successful colcon test invocation used these shell-only settings:

```bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=173
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOG_DIR=/tmp/p1_03b_01c_ros_test
```

DDS socket access was permitted outside the sandbox for this local integration
test. The only application topics published were the two remapped semantic test
outputs. No hardware graph, command topic, controller or ROS-TCP server was used.
Interface package has no independent behavior tests; the transport package's
29 tests include generated interface serialization and real DDS receipt.
Flake8 on the new Python package passed (max line length 99).

## Existing Control Path Impact

No existing source/config/launch/entry point was edited. The new node creates one
semantic publisher and no subscriptions, clients, actions or actuator publisher.
Neither existing driver interfaces nor palm/base/gripper control paths are used.
The unrelated photon_bottle_identifier tree is untouched and unstaged.

## Frozen 01B Integrity

All 15 pre-existing staged blobs, their working-tree contents and the index hash
are compared to the pre-implementation baseline. No git add/reset/restore is used.
01C files remain unstaged. Integrity results are saved separately under 01C,
never into frozen 01B documentation/evidence.

Final verification: all 15 staged blobs and the index hash are unchanged;
all 880 pre-existing tracked/untracked regular files retain their SHA256.
The frozen file working-tree diff is empty. See evidence/frozen_integrity.json.

## Remaining Hardware Binding

Runtime source audit, actual feedback, object identity association, TF/common
frame, timestamp freshness and threshold calibration remain unresolved. An input
adapter must handle these before real-robot use. Raw command values must never be
treated as actual measurements; Photon ownership is not physical hold evidence.

## Next Step

Unity-side PairSemanticState integration: generated C# message, exact enum-name
mapping, pair/session binding, duplicate/old packet rejection, receive-time/tick
expiry and compact Networked State. Hardware validation is a separate later step.
