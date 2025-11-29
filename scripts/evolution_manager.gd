extends Node
class_name EvolutionManager

@export var meta_mutation : float = 0
@export var mutation_rate : float = 0.01
@export var mutation_chance : float = 0.2
var ffnn1 : FeedForwardNeuralNetwork
var ffnn2 : FeedForwardNeuralNetwork

var max_generations : int = 10000
var target_max_fitness : float = 0.999
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
	var last_generation : int = 0
	var target_output : float = 0.676921
	for g in range(max_generations):
		last_generation = g + 1
		var input : PackedFloat32Array = generate_random_input_array(3)
		#print("input: ", input)
		var ffnn1_output : PackedFloat32Array = ffnn1.feed_forward(input)
		var ffnn2_output : PackedFloat32Array = ffnn2.feed_forward(input)
		var ffnn1_fitness : float = 1.0 - (abs(ffnn1_output[0] - target_output) / 2.0)
		var ffnn2_fitness : float = 1.0 - (abs(ffnn2_output[0] - target_output) / 2.0)
		#print("--- generation ", g + 1, " ---")
		#print("ffnn1: output: ", ffnn1_output[0], "; fitness: ", ffnn1_fitness)
		#print("ffnn2: output: ", ffnn2_output[0], "; fitness: ", ffnn2_fitness)
		if ffnn1_fitness > ffnn2_fitness:
			ffnn2.copy_from(ffnn1, false)
			ffnn2.mutate_network(0.2, 0.05)
			end_fitness = ffnn1_fitness # when the training stops, the last time it was modified will be the end fitness
			if g == 0:
				start_fitness = ffnn1_fitness
			
			if ffnn1_fitness > max_fitness:
				max_fitness = ffnn1_fitness
				max_fitness_generation = g + 1
				print("new max: ", max_fitness, "; generation: ", (g+1))
				if max_fitness >= target_max_fitness:
					break
		elif ffnn2_fitness > ffnn1_fitness:
			ffnn1.copy_from(ffnn2, false)
			ffnn1.mutate_network(mutation_rate, mutation_chance)
			end_fitness = ffnn2_fitness # when the training stops, the last time it was modified will be the end fitness
			if g == 0:
				start_fitness = ffnn2_fitness
			
			if ffnn2_fitness > max_fitness:
				max_fitness = ffnn2_fitness
				max_fitness_generation = g + 1
				print("new max: ", max_fitness, "; generation: ", (g+1))
				if max_fitness >= target_max_fitness:
					break
		else: # ffnn1_fitness == ffnn2_fitness
			ffnn1.mutate_network(mutation_rate, mutation_chance)
			if ffnn1_fitness > max_fitness:
				max_fitness = ffnn1_fitness
				max_fitness_generation = g + 1
				print(max_fitness)
				if max_fitness >= target_max_fitness:
					break
	print("--- Training Over ---")
	print("start fitness: ", start_fitness)
	print("end fitness: ", end_fitness)
	print("fitness improvement: ", (end_fitness - start_fitness))
	print("max fitness: ", max_fitness, " at generation: ", max_fitness_generation)
	print("stopped at generation: ", last_generation)
	print("target max fitness: ", target_max_fitness)
	print("max generations: ", max_generations)
func generate_random_input_array(length : int = 5) -> PackedFloat32Array:
	var random_input_array : PackedFloat32Array
	for i in range(length):
		random_input_array.append(randf_range(-1, 1))
	return random_input_array
