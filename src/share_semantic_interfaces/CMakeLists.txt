cmake_minimum_required(VERSION 3.8)
project(share_semantic_interfaces)
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LocalHoldState.msg"
  "msg/BottleFeatureSample.msg"
  "msg/BottleSelectionFeature.msg"
  "msg/BottleSelectionFeatureArray.msg"
  "msg/CanonicalBottleSelection.msg"
  "msg/CanonicalBottleCandidate.msg"
  "msg/ParticipantTargetIntent.msg"
  "msg/CanonicalBottleObservation.msg"
  "msg/ParticipantHumanSnapshot.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
