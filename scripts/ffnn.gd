extends Node
### Simple feed forward neural network implementation
### Note: All inputs and outputs are normalised between -1 and 1
### It is that your inputs are between -1 and 1
class_name FeedForwardNeuralNetwork

# weights[layer][neuron][incoming_weight]
var weights : Array = []
# biases[layer][neuron]
var biases : Array = []

var layers : PackedInt32Array

var previous_layer_activations_buffer : RID
var current_layer_activations_buffer : RID
var current_layer_biases_buffer : RID
var current_layer_weights_buffer : RID # Represented by a flattened array

var previous_layer_activations_uniform : RDUniform
var current_layer_activations_uniform : RDUniform
var current_layer_biases_uniform : RDUniform
var current_layer_weights_uniform : RDUniform

var uniform_set : RID

func feed_forward(input : PackedFloat32Array) -> PackedFloat32Array:
	if input.size() != layers[0]:
		push_error("Input size does not match input layer size.")
		return input
	var input_copy : PackedFloat32Array = input.duplicate()
	input_copy.insert(0, input_copy.size()) # Add the input size to the first index of the previous_layer_activations_buffer
	#print("input: ", input)
	# --- Copy input into previous_layer_activations_buffer ---
	var input_bytes : PackedByteArray = input_copy.to_byte_array()
	Global.rendering_device.buffer_update(
		previous_layer_activations_buffer,
		0,                     # offset
		input_bytes.size(),    # size in bytes
		input_bytes            # data
	)
	var final_output : PackedFloat32Array
	## Note: Code below is a bit hard to understand and maintain,
	## owing to the fact that biases[] and weights[] do not contain the input layer
	## however layers[] does
	for layer_index in range(0, layers.size() - 1): # -1 because input layer does not get computed and isn't included in biases[] and weights[]
		var current_layer_biases_bytes : PackedByteArray = PackedFloat32Array(biases[layer_index]).to_byte_array()
		Global.rendering_device.buffer_update(
			current_layer_biases_buffer,
			0,
			current_layer_biases_bytes.size(),
			current_layer_biases_bytes
		)
		var _3d_layer_weights_array : Array = weights[layer_index]
		var flattened_weights : PackedFloat32Array = PackedFloat32Array()
		for neuron_weights in _3d_layer_weights_array:
			for weight in neuron_weights:
				flattened_weights.append(weight)
		var current_layer_weights_bytes : PackedByteArray = flattened_weights.to_byte_array()
		Global.rendering_device.buffer_update(
			current_layer_weights_buffer,
			0,
			current_layer_weights_bytes.size(),
			current_layer_weights_bytes
		)
		# --- Dispatch compute shader ---
		var compute_list := Global.rendering_device.compute_list_begin()
		Global.rendering_device.compute_list_bind_compute_pipeline(compute_list, Global.pipeline)
		Global.rendering_device.compute_list_bind_uniform_set(compute_list, uniform_set, 0) # set = 0
		Global.rendering_device.compute_list_dispatch(compute_list, layers[layer_index + 1], 1, 1)
		Global.rendering_device.compute_list_end()
		Global.rendering_device.submit()
		Global.rendering_device.sync() # wait until done
		var current_layer_activations_bytes : PackedByteArray = Global.rendering_device.buffer_get_data(current_layer_activations_buffer)
		var current_layer_activations : PackedFloat32Array = current_layer_activations_bytes.to_float32_array()
		current_layer_activations[0] = layers[layer_index + 1] # set the first index to show the now previous_layer_activations array length
		#print("Layer ", layer_index, ": compute output: ", current_layer_activations)
		var new_previous_layer_activations_bytes = current_layer_activations.to_byte_array()
		Global.rendering_device.buffer_update(
			previous_layer_activations_buffer,
			0,
			new_previous_layer_activations_bytes.size(),
			new_previous_layer_activations_bytes
		)
		if layer_index == layers.size() - 2: # If final loop, set final_output
			final_output = current_layer_activations.slice(1, layers[layer_index + 1] + 1)
		
	return final_output
func init_compute_pipeline() -> void:
	# Determine maximum layer size (for activations/biases)
	var max_layer_size := 0
	for layer in layers:
		if layer > max_layer_size:
			max_layer_size = layer

	# account for stored length at index 0 => +1 element
	var activation_elements := max_layer_size + 1
	var activation_bytes := activation_elements * 4  # 4 bytes per float
	var empty_data := PackedByteArray()
	empty_data.resize(activation_bytes)

	# Allocate activation & bias buffers with properly sized initial data
	previous_layer_activations_buffer = Global.rendering_device.storage_buffer_create(activation_bytes, empty_data)
	current_layer_activations_buffer = Global.rendering_device.storage_buffer_create(activation_bytes, empty_data)
	current_layer_biases_buffer = Global.rendering_device.storage_buffer_create(activation_bytes, empty_data)

	# Compute maximum number of weights across all layers
	var max_weights := 0
	for i in range(1, layers.size()):
		var prev_size := layers[i - 1]
		var curr_size := layers[i]
		var weights_in_layer := prev_size * curr_size
		if weights_in_layer > max_weights:
			max_weights = weights_in_layer

	# Allocate weights buffer
	var max_weights_bytes := max_weights * 4  # 4 bytes per float
	var empty_weights := PackedByteArray()
	empty_weights.resize(max_weights_bytes)
	current_layer_weights_buffer = Global.rendering_device.storage_buffer_create(max_weights_bytes, empty_weights)
	
	# Uniforms
	previous_layer_activations_uniform = RDUniform.new()
	current_layer_activations_uniform = RDUniform.new()
	current_layer_biases_uniform = RDUniform.new()
	current_layer_weights_uniform = RDUniform.new()
	
	previous_layer_activations_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	current_layer_activations_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	current_layer_biases_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	current_layer_weights_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	
	previous_layer_activations_uniform.binding = 0
	current_layer_activations_uniform.binding = 1
	current_layer_biases_uniform.binding = 2
	current_layer_weights_uniform.binding = 3
	
	previous_layer_activations_uniform.add_id(previous_layer_activations_buffer)
	current_layer_activations_uniform.add_id(current_layer_activations_buffer)
	current_layer_biases_uniform.add_id(current_layer_biases_buffer)
	current_layer_weights_uniform.add_id(current_layer_weights_buffer)
	
	# Uniform set
	uniform_set = Global.rendering_device.uniform_set_create(
		[previous_layer_activations_uniform,
		current_layer_activations_uniform,
		current_layer_biases_uniform,
		current_layer_weights_uniform],
		Global.shader,
		0 # set = 0
	)
	
func generate_random_network(layers_in : PackedInt32Array, multiplier : float = 0.2) -> void:
	if layers_in.size() < 3:
		push_error("Need at least 3 layers: input, hidden, output.")
		return
	
	layers = layers_in
	weights.clear()
	biases.clear()

	for layer_index in range(layers.size()):
		if layer_index == 0:
			# Input layer → no weights or biases
			continue

		var layer_weights : Array = []
		var layer_biases : Array = []

		var previous_count = layers[layer_index - 1]
		var current_count = layers[layer_index]

		for _neuron in range(current_count):
			# random bias
			layer_biases.append(randf_range(-1, 1) * multiplier)

			# weights for this neuron
			var neuron_weights : Array = []
			for _w in range(previous_count):
				neuron_weights.append(randf_range(-1, 1) * multiplier)

			layer_weights.append(neuron_weights)

		weights.append(layer_weights)
		biases.append(layer_biases)

	#print("weights:", weights)
	#print("biases:", biases)
	#print("layers: ", layers)


func mutate_network(mutation_rate : float = 0.01, mutation_chance: float = 0.1) -> void:
	for bias_layer_index in range(biases.size()):
		for bias_index in range(biases[bias_layer_index].size()):
			var mutate_probability : float = randf()
			if mutate_probability < mutation_chance:
				var mutation_delta = randf_range(-1, 1) * mutation_rate
				biases[bias_layer_index][bias_index] += mutation_delta
				biases[bias_layer_index][bias_index] = clamp(biases[bias_layer_index][bias_index], -1, 1)
	for weights_layer_index in range(weights.size()):
		for neuron_weights_layer_index in range(weights[weights_layer_index].size()):
			for weight_index in range(weights[weights_layer_index][neuron_weights_layer_index].size()):
				var mutate_probability : float = randf()
				if mutate_probability < mutation_chance:
					var mutation_delta = randf_range(-1, 1) * mutation_rate
					weights[weights_layer_index][neuron_weights_layer_index][weight_index] += mutation_delta
					weights[weights_layer_index][neuron_weights_layer_index][weight_index] = clamp(weights[weights_layer_index][neuron_weights_layer_index][weight_index], -1, 1)
	#print("mutated biases: ", biases)
	#print("mutated weights: ", weights)
	return

func copy_from(other: FeedForwardNeuralNetwork, do_compute_pipeline_initialisation : bool = false) -> void:
	var architecture_changed := layers != other.layers
	
	# --- Copy layers ---
	layers = other.layers.duplicate()

	# --- Deep copy biases (2D array) ---
	biases = []
	for bias_layer in other.biases:
		biases.append(bias_layer.duplicate())

	# --- Deep copy weights (3D array) ---
	weights = []
	for weight_layer in other.weights:
		var new_weight_layer : Array = []
		for neuron_weights in weight_layer:
			new_weight_layer.append(neuron_weights.duplicate())
		weights.append(new_weight_layer)
	if do_compute_pipeline_initialisation or architecture_changed:
		init_compute_pipeline()
