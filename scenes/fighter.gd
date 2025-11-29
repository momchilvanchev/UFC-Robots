extends Node3D

var ffnn : FeedForwardNeuralNetwork

func _ready() -> void:
	ffnn = FeedForwardNeuralNetwork.new()
	ffnn.generate_random_network([6,4,2])
	ffnn.init_compute_pipeline() # must be called once after generating the network
	print("________")
	print(ffnn.feed_forward([3, 2, 4, 1, 8, 2]))
