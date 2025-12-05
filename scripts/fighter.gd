extends Node3D

var ffnn : FeedForwardNeuralNetwork

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
