extends Node3D

var ffnn : FeedForwardNeuralNetwork

func _ready() -> void:
	ffnn = FeedForwardNeuralNetwork.new()
	ffnn.generate_random_network([6,40,60,3])
	ffnn.init_compute_pipeline() # must be called once after generating the network
	var output : PackedFloat32Array = ffnn.feed_forward([-1, -0.5, -0.3, 0.7, 1, 0.2])
	print("output: ", output)
