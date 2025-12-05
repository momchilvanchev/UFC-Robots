extends Node
class_name EvolutionManager
@export var fighter_scene : PackedScene
@export var meta_mutation : float = 0
@export var mutation_rate : float = 0.2
@export var mutation_chance : float = 0.2
@export var ffnn_layers : PackedInt32Array = [3, 5, 4, 2]
@export var rounds_per_generation : int = 1
## When calling generate_random_network() it will be passed as a multiplier.
## Lower values make the weights and biases smaller, usually making the network more stable
## While higher values make the network more 'adventurous' and usually takes more generations to stabilise
@export var multiplier_weights_and_biases : float = 0.2
var ffnn1 : FeedForwardNeuralNetwork
var ffnn2 : FeedForwardNeuralNetwork

var max_generations : int = 10000
var target_max_fitness : float = 0.999
func _ready() -> void:
	var fighter1 = fighter_scene.instantiate()
	fighter1.position.y = 5
	fighter1.position.z = 5
	fighter1.rotation.x = 10
	add_child(fighter1)
	return
	ffnn1 = FeedForwardNeuralNetwork.new()
	ffnn2 = FeedForwardNeuralNetwork.new()
	ffnn1.generate_random_network(ffnn_layers, multiplier_weights_and_biases)
	ffnn2.generate_random_network(ffnn_layers, multiplier_weights_and_biases)
	ffnn1.init_compute_pipeline()
	ffnn2.init_compute_pipeline()
	var start_fitness : float
	var end_fitness : float
	var max_fitness : float = 0
	var max_fitness_generation : int = 0
	var min_fitness : float = INF
	var min_fitness_generation : int = 0
	var last_generation : int = 0
	for g in range(max_generations):
		last_generation = g + 1
		var input : PackedFloat32Array = generate_random_input_array(ffnn_layers[0])
		var target_output : float = 0
		for input_float in input:
			target_output += input_float
		target_output /= len(input)
		var ffnn1_fitness_rounds_sum : float
		var ffnn2_fitness_rounds_sum : float
		for r in range(rounds_per_generation):
			var ffnn1_output : PackedFloat32Array = ffnn1.feed_forward(input)
			var ffnn2_output : PackedFloat32Array = ffnn2.feed_forward(input)
			var ffnn1_round_fitness : float = 1.0 - (abs(ffnn1_output[0] - target_output) / 2.0)
			var ffnn2_round_fitness : float = 1.0 - (abs(ffnn2_output[0] - target_output) / 2.0)
			ffnn1_fitness_rounds_sum += ffnn1_round_fitness
			ffnn2_fitness_rounds_sum += ffnn2_round_fitness
		var ffnn1_fitness : float = ffnn1_fitness_rounds_sum / rounds_per_generation # averaged
		var ffnn2_fitness : float = ffnn2_fitness_rounds_sum / rounds_per_generation # averaged
		if ffnn1_fitness >= ffnn2_fitness:
			print(ffnn1_fitness)
			ffnn2.copy_from(ffnn1, false)
			ffnn2.mutate_network(mutation_rate, mutation_chance)
			end_fitness = ffnn1_fitness # when the training stops, the last time it was modified will be the end fitness
			if g == 0:
				start_fitness = ffnn1_fitness
			
			if ffnn1_fitness > max_fitness:
				max_fitness = ffnn1_fitness
				max_fitness_generation = g + 1
				#print("new max: ", max_fitness, "; generation: ", (g+1))
				if max_fitness >= target_max_fitness:
					break
			if ffnn1_fitness < min_fitness:
				min_fitness = ffnn1_fitness
				min_fitness_generation = g + 1
		elif ffnn2_fitness > ffnn1_fitness:
			print(ffnn2_fitness)
			ffnn1.copy_from(ffnn2, false)
			ffnn1.mutate_network(mutation_rate, mutation_chance)
			end_fitness = ffnn2_fitness # when the training stops, the last time it was modified will be the end fitness
			if g == 0:
				start_fitness = ffnn2_fitness
			
			if ffnn2_fitness > max_fitness:
				max_fitness = ffnn2_fitness
				max_fitness_generation = g + 1
				#print("new max: ", max_fitness, "; generation: ", (g+1))
				if max_fitness >= target_max_fitness:
					break
			if ffnn2_fitness < min_fitness:
				min_fitness = ffnn2_fitness
				min_fitness_generation = g + 1
	print("--- Training Over ---")
	print("start fitness: ", start_fitness)
	print("end fitness: ", end_fitness)
	print("fitness improvement: ", (end_fitness - start_fitness))
	print("min fitness: ", min_fitness, " at generation: ", min_fitness_generation)
	print("max fitness: ", max_fitness, " at generation: ", max_fitness_generation)
	print("stopped at generation: ", last_generation)
	print("target max fitness: ", target_max_fitness)
	print("rounds per generation: ", rounds_per_generation)
	print("max generations: ", max_generations)
func generate_random_input_array(length : int = 5, min : float = -1, max : float = 1) -> PackedFloat32Array:
	var random_input_array : PackedFloat32Array
	for i in range(length):
		random_input_array.append(randf_range(min, max))
	return random_input_array
