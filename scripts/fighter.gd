extends Node3D
class_name Fighter
@export_subgroup("Rigidbodies")
@export var head : RigidBody3D
@export var neck : RigidBody3D
@export var torso : RigidBody3D
@export var pelvis : RigidBody3D
@export var left_upper_leg : RigidBody3D
@export var right_upper_leg : RigidBody3D
@export var left_lower_leg : RigidBody3D
@export var right_lower_leg : RigidBody3D
@export var left_foot : RigidBody3D
@export var right_foot : RigidBody3D
@export var left_upper_arm : RigidBody3D
@export var right_upper_arm : RigidBody3D
@export var left_lower_arm : RigidBody3D
@export var right_lower_arm : RigidBody3D
@export var left_fist : RigidBody3D
@export var right_fist : RigidBody3D
@export_subgroup("Joints")
@export var head_neck : Generic6DOFJoint3D
@export var torso_neck : Generic6DOFJoint3D
@export var torso_left_upper_arm : Generic6DOFJoint3D
@export var torso_right_upper_arm : Generic6DOFJoint3D
@export var torso_pelvis : Generic6DOFJoint3D
@export var pelvis_left_upper_leg : Generic6DOFJoint3D
@export var pelvis_right_upper_leg : Generic6DOFJoint3D
@export var left_upper_arm_left_lower_arm : Generic6DOFJoint3D
@export var right_upper_arm_right_lower_arm : Generic6DOFJoint3D
@export var left_upper_leg_left_lower_leg : Generic6DOFJoint3D
@export var right_upper_leg_right_lower_leg : Generic6DOFJoint3D
@export var left_lower_arm_left_fist : Generic6DOFJoint3D
@export var right_lower_arm_right_fist : Generic6DOFJoint3D
@export var left_lower_leg_left_foot : Generic6DOFJoint3D
@export var right_lower_leg_right_foot : Generic6DOFJoint3D

var ffnn : FeedForwardNeuralNetwork

enum body_part {
	HEAD,
	TORSO,
	PELVIS,
	LEFT_UPPER_LEG,
	RIGHT_UPPER_LEG,
	LEFT_LOWER_LEG,
	RIGHT_LOWER_LEG,
	LEFT_FOOT,
	RIGHT_FOOT,
	LEFT_UPPER_ARM,
	RIGHT_UPPER_ARM,
	LEFT_LOWER_ARM,
	RIGHT_LOWER_ARM,
	LEFT_FIST,
	RIGHT_FIST
}
enum body_connection {
	HEAD_NECK,
	TORSO_NECK,
	TORSO_LEFT_UPPER_ARM,
	TORSO_RIGHT_UPPER_ARM,
	TORSO_PELVIS,
	PELVIS_LEFT_UPPER_LEG,
	PELVIS_RIGHT_UPPER_LEG,
	LEFT_UPPER_ARM_LEFT_LOWER_ARM,
	RIGHT_UPPER_ARM_RIGHT_LOWER_ARM,
	LEFT_UPPER_LEG_LEFT_LOWER_LEG,
	RIGHT_UPPER_LEG_RIGHT_LOWER_LEG,
	LEFT_LOWER_ARM_LEFT_FIST,
	RIGHT_LOWER_ARM_RIGHT_FIST,
	LEFT_LOWER_LEG_LEFT_FOOT,
	RIGHT_LOWER_LEG_RIGHT_FOOT
}
func _ready() -> void:
	pass
	
func init_ffnn(ffnn_layers : PackedInt32Array = [3, 5, 4, 2], multiplier_weights_and_biases : float = 0.2) -> void:
	ffnn = FeedForwardNeuralNetwork.new()
	ffnn.generate_random_network(ffnn_layers, multiplier_weights_and_biases)
	ffnn.init_compute_pipeline()
	
func reset_body() -> void:
	pass
	
func get_input() -> PackedFloat32Array:
	return []

func use_output(output : PackedFloat32Array) -> void:
	pass
	
func get_body_part(body_part_enum : body_part) -> RigidBody3D:
	match body_part_enum:
		body_part.HEAD:
			return head
		body_part.TORSO:
			return torso
		_:
			return torso # default to torso if unknown body part
