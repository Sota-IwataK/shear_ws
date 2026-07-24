from pathlib import Path

import yaml


SRC_ROOT = Path(__file__).resolve().parents[2]
DRIVER_CONFIG = (
    SRC_ROOT /
    "humbleble_ws/amir740_ros/amir_driver/config/amir_ros2_controllers.yaml"
)
DRIVER_LAUNCH = (
    SRC_ROOT /
    "humbleble_ws/amir740_ros/amir_driver/launch/amir_bringup.launch.py"
)
SUPER_MOVEIT_LAUNCH = (
    SRC_ROOT /
    "humbleble_ws/amir740_ros/amir_bringup/launch/super_amir_moveit.launch.py"
)
AMIR_MOVEIT_LAUNCH = (
    SRC_ROOT /
    "humbleble_ws/amir740_ros/amir_bringup/launch/amir_moveit.launch.py"
)


def forward_spawner_segment(text):
    start = text.index("forward_position_controller_spawner")
    end = text.index("gripper_controller_spawner", start)
    return text[start:end]


def load_driver_config():
    with DRIVER_CONFIG.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_forward_position_controller_is_configured_in_real_driver_yaml():
    data = load_driver_config()
    manager_params = data["controller_manager"]["ros__parameters"]
    assert manager_params["update_rate"] == 60
    assert manager_params["arm_controller"]["type"] == (
        "joint_trajectory_controller/JointTrajectoryController")
    assert manager_params["gripper_controller"]["type"] == (
        "position_controllers/GripperActionController")
    assert manager_params["joint_state_broadcaster"]["type"] == (
        "joint_state_broadcaster/JointStateBroadcaster")
    assert manager_params["forward_position_controller"]["type"] == (
        "position_controllers/JointGroupPositionController")

    forward_params = data["forward_position_controller"]["ros__parameters"]
    assert forward_params["joints"] == [
        "Joint_1",
        "Joint_2",
        "Joint_3",
        "Joint_4",
        "Joint_5",
    ]


def test_arm_controller_interfaces_are_unchanged():
    arm_params = load_driver_config()["arm_controller"]["ros__parameters"]
    assert arm_params["joints"] == [
        "Joint_1",
        "Joint_2",
        "Joint_3",
        "Joint_4",
        "Joint_5",
    ]
    assert arm_params["command_interfaces"] == ["position", "velocity"]
    assert arm_params["state_interfaces"] == ["position", "velocity"]


def test_real_launches_read_driver_controller_yaml_and_spawn_inactive():
    for path in (DRIVER_LAUNCH, SUPER_MOVEIT_LAUNCH, AMIR_MOVEIT_LAUNCH):
        text = path.read_text(encoding="utf-8")
        segment = forward_spawner_segment(text)
        assert "amir_ros2_controllers.yaml" in text
        assert "forward_position_controller" in segment
        assert "--inactive" in segment
        assert '"-p"' in segment
        assert "--controller-manager" in segment
        assert (
            "robot_controllers" in segment
            or "amir_controllers" in segment
        )
