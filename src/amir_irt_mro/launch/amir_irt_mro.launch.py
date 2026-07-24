#!/usr/bin/env python3
"""
amir_irt_mro 統合launchファイル
全ノードを一括起動する
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # 引数
    use_sim = LaunchConfiguration('use_sim', default='true')
    mode = LaunchConfiguration('mode', default='sim')
    real_unit = LaunchConfiguration('real_unit', default='mrad')
    real_output = LaunchConfiguration('real_output', default='joint_trajectory')
    deadman_required = LaunchConfiguration('deadman_required', default='true')
    deadman_topic = LaunchConfiguration('deadman_topic', default='/deadman_enabled')
    palm_pose_timeout_sec = LaunchConfiguration('palm_pose_timeout_sec', default='0.3')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    start_initial_position = LaunchConfiguration('start_initial_position', default='false')
    start_trajectory_node = LaunchConfiguration('start_trajectory_node', default='true')
    start_palm_ik = LaunchConfiguration('start_palm_ik', default='false')
    palm_pose_topic = LaunchConfiguration('palm_pose_topic', default='/palm_pose')
    control_center_topic = LaunchConfiguration(
        'control_center_topic', default='/palm_pose_control_center_world')
    arm_command_topic = LaunchConfiguration(
        'arm_command_topic', default='/arm_controller/joint_trajectory')
    arm_state_topic = LaunchConfiguration('arm_state_topic', default='/joint_states')
    gripper_command_topic = LaunchConfiguration(
        'gripper_command_topic', default='/grasp_command')
    command_unit = LaunchConfiguration('command_unit', default='rad')
    control_profile = LaunchConfiguration('control_profile', default='safe_trajectory')
    trajectory_time_sec = LaunchConfiguration('trajectory_time_sec', default='2.0')
    max_joint_step_rad = LaunchConfiguration('max_joint_step_rad', default='0.005')
    publish_rate_hz = LaunchConfiguration('publish_rate_hz', default='5.0')
    dry_run = LaunchConfiguration('dry_run', default='true')
    servo_rate_hz = LaunchConfiguration('servo_rate_hz', default='5.0')
    servo_horizon_sec = LaunchConfiguration('servo_horizon_sec', default='0.25')
    reference_servo_rate_hz = LaunchConfiguration(
        'reference_servo_rate_hz', default='50.0')
    reference_tracking_error_threshold_rad = LaunchConfiguration(
        'reference_tracking_error_threshold_rad', default='0.08')
    trajectory_buffer_update_rate_hz = LaunchConfiguration(
        'trajectory_buffer_update_rate_hz', default='10.0')
    trajectory_buffer_horizon_sec = LaunchConfiguration(
        'trajectory_buffer_horizon_sec', default='0.30')
    trajectory_buffer_point_dt_sec = LaunchConfiguration(
        'trajectory_buffer_point_dt_sec', default='0.05')
    max_cartesian_velocity_mps = LaunchConfiguration(
        'max_cartesian_velocity_mps', default='0.05')
    max_joint_velocity_radps = LaunchConfiguration(
        'max_joint_velocity_radps', default='0.15')
    max_cartesian_acceleration_mps2 = LaunchConfiguration(
        'max_cartesian_acceleration_mps2', default='0.20')
    max_joint_acceleration_radps2 = LaunchConfiguration(
        'max_joint_acceleration_radps2', default='0.50')
    velocity_servo_hardware_confirmed = LaunchConfiguration(
        'velocity_servo_hardware_confirmed', default='false')
    velocity_servo_time_from_start_sec = LaunchConfiguration(
        'velocity_servo_time_from_start_sec', default='0.05')
    deadman_enabled = LaunchConfiguration('deadman_enabled', default='false')
    deadman_timeout_sec = LaunchConfiguration('deadman_timeout_sec', default='0.5')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim',
            default_value='true',
            description='Gazebo simulation mode (true) or real robot (false)'),
        DeclareLaunchArgument(
            'mode',
            default_value='sim',
            description='Control mode: sim, real, or dry_run'),
        DeclareLaunchArgument(
            'real_unit',
            default_value='mrad',
            description='Real AmirCmd unit: mrad or rad'),
        DeclareLaunchArgument(
            'real_output',
            default_value='joint_trajectory',
            description='Real output path: joint_trajectory or amir_cmd'),
        DeclareLaunchArgument(
            'deadman_required',
            default_value='true',
            description='Require /deadman_enabled true before sending commands'),
        DeclareLaunchArgument(
            'deadman_topic',
            default_value='/deadman_enabled',
            description='std_msgs/Bool deadman topic'),
        DeclareLaunchArgument(
            'palm_pose_timeout_sec',
            default_value='0.3',
            description='Stop command output when /palm_pose is stale'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'),
        DeclareLaunchArgument(
            'start_initial_position',
            default_value='false',
            description='Start initial-position node. Default false for safety.'),
        DeclareLaunchArgument(
            'start_trajectory_node',
            default_value='true',
            description='Start legacy amir_trajectory_node. Set false when using palm IK.'),
        DeclareLaunchArgument(
            'start_palm_ik',
            default_value='false',
            description='Start amir_palm_ik_node explicitly. Default false.'),
        DeclareLaunchArgument(
            'palm_pose_topic',
            default_value='/palm_pose',
            description='Palm pose input topic for palm IK.'),
        DeclareLaunchArgument(
            'control_center_topic',
            default_value='/palm_pose_control_center_world',
            description='Palm control center topic for palm IK.'),
        DeclareLaunchArgument(
            'arm_command_topic',
            default_value='/arm_controller/joint_trajectory',
            description='Arm JointTrajectory command topic.'),
        DeclareLaunchArgument(
            'arm_state_topic',
            default_value='/joint_states',
            description='Arm JointState topic used before real commands.'),
        DeclareLaunchArgument(
            'gripper_command_topic',
            default_value='/grasp_command',
            description='Gripper command topic.'),
        DeclareLaunchArgument(
            'command_unit',
            default_value='rad',
            description='Palm IK debug command unit: rad or mrad. JTC output remains rad.'),
        DeclareLaunchArgument(
            'control_profile',
            default_value='safe_trajectory',
            description=(
                'Palm IK profile: safe_trajectory, '
                'teleop_trajectory_stream, teleop_trajectory_buffer, '
                'or teleop_velocity_servo.')),
        DeclareLaunchArgument(
            'trajectory_time_sec',
            default_value='2.0',
            description='Palm IK trajectory time. Real mode is clamped to >=2.0.'),
        DeclareLaunchArgument(
            'max_joint_step_rad',
            default_value='0.005',
            description='Palm IK per-cycle joint step limit. Real mode is capped at 0.01.'),
        DeclareLaunchArgument(
            'publish_rate_hz',
            default_value='5.0',
            description='Palm IK publish rate. Real mode is capped at 10Hz.'),
        DeclareLaunchArgument(
            'dry_run',
            default_value='true',
            description='Real palm IK dry-run. Set false only after diagnostics.'),
        DeclareLaunchArgument(
            'servo_rate_hz',
            default_value='5.0',
            description=(
                'Teleop loop rate. trajectory_stream uses 5-10Hz; '
                'velocity_servo requires >=20Hz.')),
        DeclareLaunchArgument(
            'servo_horizon_sec',
            default_value='0.25',
            description='teleop_trajectory_stream JointTrajectory horizon.'),
        DeclareLaunchArgument(
            'reference_servo_rate_hz',
            default_value='50.0',
            description='teleop_trajectory_buffer internal reference integration rate.'),
        DeclareLaunchArgument(
            'reference_tracking_error_threshold_rad',
            default_value='0.08',
            description='Reset q_reference only when measured/reference max error exceeds this.'),
        DeclareLaunchArgument(
            'trajectory_buffer_update_rate_hz',
            default_value='10.0',
            description='teleop_trajectory_buffer publish update rate.'),
        DeclareLaunchArgument(
            'trajectory_buffer_horizon_sec',
            default_value='0.30',
            description='teleop_trajectory_buffer rolling horizon.'),
        DeclareLaunchArgument(
            'trajectory_buffer_point_dt_sec',
            default_value='0.05',
            description='teleop_trajectory_buffer waypoint interval.'),
        DeclareLaunchArgument(
            'max_cartesian_velocity_mps',
            default_value='0.05',
            description='Teleop Cartesian velocity limit.'),
        DeclareLaunchArgument(
            'max_joint_velocity_radps',
            default_value='0.15',
            description='Teleop joint velocity limit. Use 0.05 first for velocity_servo.'),
        DeclareLaunchArgument(
            'max_cartesian_acceleration_mps2',
            default_value='0.20',
            description='Teleop Cartesian acceleration limit.'),
        DeclareLaunchArgument(
            'max_joint_acceleration_radps2',
            default_value='0.50',
            description='Teleop joint acceleration limit.'),
        DeclareLaunchArgument(
            'velocity_servo_hardware_confirmed',
            default_value='false',
            description=(
                'Allow teleop_velocity_servo real publish only after '
                'hardware/controller velocity diagnostics pass.')),
        DeclareLaunchArgument(
            'velocity_servo_time_from_start_sec',
            default_value='0.05',
            description='Velocity-only JointTrajectory point horizon.'),
        DeclareLaunchArgument(
            'deadman_enabled',
            default_value='false',
            description='Initial deadman state. Prefer publishing /deadman_enabled.'),
        DeclareLaunchArgument(
            'deadman_timeout_sec',
            default_value='0.5',
            description='teleop_servo deadman timeout.'),

        # 1. 初期姿勢設定（最初に起動）
        Node(
            package='amir_irt_mro',
            executable='amir_initial_position',
            name='amir_initial_position',
            output='screen',
            condition=IfCondition(start_initial_position),
            parameters=[{
                'use_sim_time': use_sim_time,
                'auto_send': False,
                'deadman_required': deadman_required,
                'deadman_topic': deadman_topic,
            }],
        ),

        # 2. アーム軌道制御（2秒後）
        TimerAction(period=2.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='amir_trajectory_node',
                name='amir_trajectory_node',
                output='screen',
                condition=IfCondition(start_trajectory_node),
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'use_sim': use_sim,
                    'mode': mode,
                    'real_unit': real_unit,
                    'real_output': real_output,
                    'arm_trajectory_topic': arm_command_topic,
                    'joint_state_topic': arm_state_topic,
                    'deadman_required': deadman_required,
                    'deadman_topic': deadman_topic,
                    'palm_pose_timeout_sec': palm_pose_timeout_sec,
                }],
            ),
        ]),

        # 2b. PalmPose relative IK. Disabled by default to avoid two arm publishers.
        TimerAction(period=2.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='amir_palm_ik_node',
                name='amir_palm_ik_node',
                output='screen',
                condition=IfCondition(start_palm_ik),
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'mode': mode,
                    'palm_pose_topic': palm_pose_topic,
                    'control_center_topic': control_center_topic,
                    'arm_command_topic': arm_command_topic,
                    'arm_state_topic': arm_state_topic,
                    'gripper_command_topic': gripper_command_topic,
                    'command_unit': command_unit,
                    'control_profile': control_profile,
                    'trajectory_time_sec': trajectory_time_sec,
                    'max_joint_step_rad': max_joint_step_rad,
                    'publish_rate_hz': publish_rate_hz,
                    'dry_run': dry_run,
                    'servo_rate_hz': servo_rate_hz,
                    'servo_horizon_sec': servo_horizon_sec,
                    'reference_servo_rate_hz': reference_servo_rate_hz,
                    'reference_tracking_error_threshold_rad':
                        reference_tracking_error_threshold_rad,
                    'trajectory_buffer_update_rate_hz':
                        trajectory_buffer_update_rate_hz,
                    'trajectory_buffer_horizon_sec':
                        trajectory_buffer_horizon_sec,
                    'trajectory_buffer_point_dt_sec':
                        trajectory_buffer_point_dt_sec,
                    'max_cartesian_velocity_mps': max_cartesian_velocity_mps,
                    'max_joint_velocity_radps': max_joint_velocity_radps,
                    'max_cartesian_acceleration_mps2':
                        max_cartesian_acceleration_mps2,
                    'max_joint_acceleration_radps2':
                        max_joint_acceleration_radps2,
                    'velocity_servo_hardware_confirmed':
                        velocity_servo_hardware_confirmed,
                    'velocity_servo_time_from_start_sec':
                        velocity_servo_time_from_start_sec,
                    'deadman_enabled': deadman_enabled,
                    'deadman_timeout_sec': deadman_timeout_sec,
                    'deadman_topic': deadman_topic,
                }],
            ),
        ]),

        # 3. グリッパー制御（2秒後）
        TimerAction(period=2.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='amir_gripper',
                name='amir_gripper',
                output='screen',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'grasp_command_topic': gripper_command_topic,
                    'gripper_state_topic': '/gripper_state',
                    'joint_state_topic': arm_state_topic,
                }],
            ),
        ]),

        # 4. ベース移動（2秒後）
        TimerAction(period=2.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='amir_base_move',
                name='amir_base_move',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ]),

        # 5. 経路計画（3秒後）
        TimerAction(period=3.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='amir_path_planner',
                name='amir_path_planner',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ]),

        # 6. 座標変換（3秒後）
        TimerAction(period=3.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='amir_affine_transform',
                name='amir_affine_transform',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ]),

        # 7. 競合検出（4秒後）
        TimerAction(period=4.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='conflict_detector',
                name='conflict_detector',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ]),

        # 8. 情報フィルタリング（4秒後）
        TimerAction(period=4.0, actions=[
            Node(
                package='amir_irt_mro',
                executable='info_filter',
                name='info_filter',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ]),
    ])
