extends Node
class_name EvolutionManager

@export var meta_mutation : float = 0

var ffnn1 : FeedForwardNeuralNetwork
var ffnn2 : FeedForwardNeuralNetwork

var generations : int = 10000
func _ready() -> void:
	ffnn1 = FeedForwardNeuralNetwork.new()
	ffnn2 = FeedForwardNeuralNetwork.new()
	ffnn1.generate_random_network([3, 6, 5, 4, 3, 1])
	ffnn2.generate_random_network([3, 6, 5, 4, 3, 1])
	ffnn1.init_compute_pipeline()
	ffnn2.init_compute_pipeline()
	var start_fitness : float
	var end_fitness : float
	var max_fitness : float = 0
	var max_fitness_generation : int = 0
	for g in range(generations):
		var input : PackedFloat32Array = generate_random_input_array(3)
		var ffnn1_output : PackedFloat32Array = ffnn1.feed_forward(input)
		var ffnn2_output : PackedFloat32Array = ffnn2.feed_forward(input)
		var target_output : float = 0.69
		var ffnn1_fitness : float = 1.0 - (abs(ffnn1_output[0] - target_output) / 2.0)
		var ffnn2_fitness : float = 1.0 - (abs(ffnn2_output[0] - target_output) / 2.0)
		print("--- generation ", g + 1, " ---")
		#print("ffnn1: output: ", ffnn1_output[0], "; fitness: ", ffnn1_fitness)
		#print("ffnn2: output: ", ffnn2_output[0], "; fitness: ", ffnn2_fitness)
		if ffnn1_fitness > ffnn2_fitness:
			ffnn2.copy_from(ffnn1, false)
			ffnn2.mutate_network(0.01, 0.05)
			if g == 0:
				start_fitness = ffnn1_fitness
			elif g == (generations - 1):
				end_fitness = ffnn1_fitness
			
			if ffnn1_fitness > max_fitness:
				max_fitness = ffnn1_fitness
				max_fitness_generation = g + 1
		elif ffnn2_fitness > ffnn1_fitness:
			ffnn1.copy_from(ffnn2, false)
			ffnn1.mutate_network(0.01, 0.05)
			if g == 0:
				start_fitness = ffnn2_fitness
			elif g == (generations - 1):
				end_fitness = ffnn2_fitness
			
			if ffnn2_fitness > max_fitness:
				max_fitness = ffnn2_fitness
				max_fitness_generation = g + 1
		else: # ffnn1_fitness == ffnn2_fitness
			ffnn1.mutate_network(0.01, 0.05)
	print("--- Training Over ---")
	print("start fitness: ", start_fitness)
	print("end fitness: ", end_fitness)
	print("fitness improvement: ", (end_fitness - start_fitness))
	print("max fitness: ", max_fitness, " at generation: ", max_fitness_generation)
func generate_random_input_array(length : int = 5) -> PackedFloat32Array:
	var random_input_array : PackedFloat32Array
	for i in range(length):
		random_input_array.append(randfn(-1, 1))
	return random_input_array
