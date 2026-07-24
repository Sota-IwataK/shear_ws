# Amir real robot diagnosis

This note is for ROS 2 Humble real-hardware checks before enabling any arm
motion from `/palm_pose`.

## Confirmed real graph snapshot

The real robot was reported with these arm-related topics:

| Topic | Expected role | Message type status |
| --- | --- | --- |
| `/arm_controller/joint_trajectory` | ros2_control JTC input | Must confirm live; repo expects `trajectory_msgs/msg/JointTrajectory` |
| `/arm_controller/state` | JTC state | Must confirm live |
| `/dynamic_joint_states` | ros2_control state | Must confirm live |
| `/joint_states` | Arm/gripper state | Must confirm live; repo expects `sensor_msgs/msg/JointState` |
| `/encoder_pub` | Low-level Amir encoder state | Must confirm live; repo msg is `amir_interfaces/msg/AmirSensor` |
| `/motor_sub` | Low-level Amir motor command candidate | Must confirm live before enabling bridge |
| `/grasp_command` | Gripper command candidate | Must confirm live; repo uses `std_msgs/msg/String` |
| `/gripper_state` | Gripper state candidate | Must confirm live; repo uses `std_msgs/msg/Bool` |

## Diagnosis commands

Run these on the real robot before `dry_run:=false`.

```bash
ros2 topic info -v /arm_controller/joint_trajectory
ros2 topic info -v /motor_sub
ros2 topic info -v /encoder_pub
ros2 topic info -v /grasp_command
ros2 topic type /arm_controller/joint_trajectory
ros2 topic type /motor_sub
ros2 topic type /encoder_pub
ros2 topic type /grasp_command
ros2 node list
```

Controller and hardware checks:

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
ros2 control list_controller_types
ros2 topic list
ros2 param get /controller_manager robot_description
```

For velocity servo, confirm that every arm joint exposes command velocity and
that the active controller accepts velocity trajectory points:

```bash
ros2 control list_hardware_interfaces | grep -E 'Joint_[1-5]/velocity'
ros2 control list_controllers
ros2 topic info -v /arm_controller/joint_trajectory
ros2 topic echo --once /arm_controller/state
```

Do not set `velocity_servo_hardware_confirmed:=true` until this is verified on
the running robot.

Forward controller check, only if it is later added:

```bash
ros2 control load_controller --set-state inactive forward_position_controller
ros2 control list_controllers
ros2 topic info -v /forward_position_controller/commands
ros2 topic type /forward_position_controller/commands
ros2 control switch_controllers \
  --deactivate arm_controller \
  --activate forward_position_controller \
  --strict
ros2 control list_controllers
```

Expected input type for
`position_controllers/JointGroupPositionController` is
`std_msgs/msg/Float64MultiArray`.  Keep `arm_controller` and
`forward_position_controller` from being active at the same time because both
claim the arm joints' position command interfaces.

## Unit policy

| Path | Topic | Message | Unit |
| --- | --- | --- | --- |
| IK internal | none | numpy arrays | rad |
| Joint limits/FK/step limit | none | numpy arrays | rad |
| Gazebo JTC | `/arm_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | rad |
| Real JTC | `/arm_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory` | rad |
| Real hardware interface | `/motor_sub` | `amir_interfaces/msg/AmirCmd` | mrad, after driver offset |
| Real encoder | `/encoder_pub` | `amir_interfaces/msg/AmirSensor` | mrad, before driver offset |

Do not send mrad values to `/arm_controller/joint_trajectory`.  The real
`amir_driver` converts ros2_control rad commands to `/motor_sub` mrad.

## Gazebo run

```bash
source install/setup.bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=gazebo \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p publish_rate_hz:=30.0 \
  -p trajectory_time_sec:=0.35
```

Confirm:

```bash
ros2 topic info -v /arm_controller/joint_trajectory
ros2 topic echo --once /joint_states
ros2 topic hz /arm_controller/joint_trajectory
```

## Real dry-run

`dry_run:=true` is the default in `mode:=real`.  It logs planned rad commands
but does not publish to `/arm_controller/joint_trajectory`.

```bash
source install/setup.bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p dry_run:=true \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p gripper_command_topic:=/grasp_command \
  -p command_unit:=rad \
  -p publish_rate_hz:=5.0 \
  -p max_joint_step_rad:=0.005 \
  -p trajectory_time_sec:=2.0
```

Confirm no real command is published:

```bash
ros2 topic echo /arm_controller/joint_trajectory
ros2 topic echo --once /joint_states
```

## Real JTC enable after dry-run

Confirm the controller is active:

```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces
```

Start palm IK with real publishing:

```bash
source install/setup.bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p dry_run:=false \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p publish_rate_hz:=5.0 \
  -p max_joint_step_rad:=0.005 \
  -p trajectory_time_sec:=2.0
```

The first `/palm_pose` only sets the anchor.  No command is published for that
cycle.

## Real teleop trajectory stream

This mode keeps the known-good real JTC message shape:
`header.stamp=0`, `positions` only, and `velocities=[]`.

First dry-run check:

```bash
source install/setup.bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p control_profile:=teleop_trajectory_stream \
  -p target_mapping_mode:=relative \
  -p dry_run:=true \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p servo_rate_hz:=5.0 \
  -p servo_horizon_sec:=0.25 \
  -p max_cartesian_velocity_mps:=0.05 \
  -p max_joint_velocity_radps:=0.15
```

Enable only after the dry-run logs show
`known_good_format=true`, `header_stamp={'sec': 0, 'nanosec': 0}`,
and `velocities_length=0`:

```bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p control_profile:=teleop_trajectory_stream \
  -p target_mapping_mode:=relative \
  -p dry_run:=false \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p servo_rate_hz:=5.0 \
  -p servo_horizon_sec:=0.25 \
  -p max_cartesian_velocity_mps:=0.05 \
  -p max_joint_velocity_radps:=0.15
```

## Real teleop trajectory buffer

This mode keeps the same `/arm_controller/joint_trajectory` JTC path and the
same known-good message shape, but sends a rolling horizon with multiple
position-only waypoints. It does not require `forward_position_controller`.

First dry-run check:

```bash
source install/setup.bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p control_profile:=teleop_trajectory_buffer \
  -p target_mapping_mode:=relative \
  -p dry_run:=true \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p reference_servo_rate_hz:=50.0 \
  -p reference_tracking_error_threshold_rad:=0.08 \
  -p trajectory_buffer_update_rate_hz:=10.0 \
  -p trajectory_buffer_horizon_sec:=0.30 \
  -p trajectory_buffer_point_dt_sec:=0.05 \
  -p max_cartesian_velocity_mps:=0.05 \
  -p max_joint_velocity_radps:=0.15 \
  -p max_joint_acceleration_radps2:=0.50
```

Enable only after dry-run logs show `known_good_format=true`,
`q_reference`, `tracking_error_rad`, `waypoint_count=6`,
`publish_reason=first_buffer` or `target_updated`, and
no workspace/IK rejection:

```bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p control_profile:=teleop_trajectory_buffer \
  -p target_mapping_mode:=relative \
  -p dry_run:=false \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p reference_servo_rate_hz:=50.0 \
  -p reference_tracking_error_threshold_rad:=0.08 \
  -p trajectory_buffer_update_rate_hz:=10.0 \
  -p trajectory_buffer_horizon_sec:=0.30 \
  -p trajectory_buffer_point_dt_sec:=0.05 \
  -p max_cartesian_velocity_mps:=0.05 \
  -p max_joint_velocity_radps:=0.15 \
  -p max_joint_acceleration_radps2:=0.50
```

## Real teleop velocity servo

This mode publishes velocity-only `JointTrajectory` points.  It is gated by
`velocity_servo_hardware_confirmed:=true`; without that flag, `dry_run:=false`
is still suppressed.

Start with dry-run at low joint velocity:

```bash
source install/setup.bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p control_profile:=teleop_velocity_servo \
  -p target_mapping_mode:=relative \
  -p dry_run:=true \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p servo_rate_hz:=20.0 \
  -p velocity_servo_time_from_start_sec:=0.05 \
  -p max_cartesian_velocity_mps:=0.03 \
  -p max_joint_velocity_radps:=0.05 \
  -p velocity_servo_hardware_confirmed:=false
```

Only after controller velocity acceptance is confirmed, run a small-motion test:

```bash
ros2 run amir_irt_mro amir_palm_ik_node --ros-args \
  -p mode:=real \
  -p control_profile:=teleop_velocity_servo \
  -p target_mapping_mode:=relative \
  -p dry_run:=false \
  -p palm_pose_topic:=/palm_pose \
  -p arm_command_topic:=/arm_controller/joint_trajectory \
  -p arm_state_topic:=/joint_states \
  -p command_unit:=rad \
  -p servo_rate_hz:=20.0 \
  -p velocity_servo_time_from_start_sec:=0.05 \
  -p max_cartesian_velocity_mps:=0.03 \
  -p max_joint_velocity_radps:=0.05 \
  -p velocity_servo_hardware_confirmed:=true
```

## Safe one-joint micro test

Use the current `/joint_states` values and change only one joint by at most
`0.005` rad.  Replace the numbers below with the current positions before
publishing.

```bash
ros2 topic echo --once /joint_states

ros2 topic pub --once /arm_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
"{joint_names: [Joint_1, Joint_2, Joint_3, Joint_4, Joint_5],
  points: [{positions: [0.005, 1.000, -1.000, 0.000, 0.000],
            time_from_start: {sec: 2, nanosec: 0}}]}"
```

Do not use the example numbers directly on hardware unless they match the
current robot state except for the intended tiny joint delta.

## Low-level bridge, only after type confirmation

Use this only if `/motor_sub` is confirmed as the actual motor command topic
and its live type is `amir_interfaces/msg/AmirCmd`.

Dry-run:

```bash
ros2 run amir_irt_mro amir_real_command_bridge --ros-args \
  -p input_trajectory_topic:=/arm_controller/joint_trajectory \
  -p motor_command_topic:=/motor_sub \
  -p motor_state_topic:=/encoder_pub \
  -p command_unit:=rad \
  -p dry_run:=true \
  -p max_joint_step_rad:=0.005
```

Publish enabled:

```bash
ros2 run amir_irt_mro amir_real_command_bridge --ros-args \
  -p input_trajectory_topic:=/arm_controller/joint_trajectory \
  -p motor_command_topic:=/motor_sub \
  -p motor_state_topic:=/encoder_pub \
  -p confirmed_motor_command_type:=amir_interfaces/msg/AmirCmd \
  -p command_unit:=rad \
  -p dry_run:=false \
  -p max_joint_step_rad:=0.005
```

## Gripper check

```bash
ros2 topic type /grasp_command
ros2 topic type /gripper_state
ros2 topic echo /gripper_state
ros2 topic pub --once /grasp_command std_msgs/msg/String "{data: open}"
ros2 topic pub --once /grasp_command std_msgs/msg/String "{data: close}"
```

If `/grasp_command` is not `std_msgs/msg/String`, do not use `amir_gripper`
until a small typed bridge is added.

## Emergency stop and JTC recovery

Stop palm IK or any bridge first:

```bash
pkill -f amir_palm_ik_node
pkill -f amir_real_command_bridge
```

Confirm no command publishers remain:

```bash
ros2 topic info -v /arm_controller/joint_trajectory
ros2 topic info -v /motor_sub
```

If controllers were switched later, return to JTC:

```bash
ros2 control switch_controllers \
  --deactivate forward_position_controller \
  --activate arm_controller

ros2 control list_controllers
```
