extends Node
### Simple feed forward neural network implementation
### Note: All inputs and outputs are normalised between -1 and 1
### It is expected of you to normalise your inputs between -1 and 1
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

var uniform_set

func feed_forward(input : PackedFloat32Array) -> PackedFloat32Array:
	if input.size() != layers[0]:
		push_error("Input size does not match input layer size.")
		return input
	
	# --- Copy input into previous_layer_activations_buffer ---
	var input_bytes := input.to_byte_array()
	Global.rendering_device.buffer_update(
		previous_layer_activations_buffer,
		0,                     # offset
		input_bytes.size(),    # size in bytes
		input_bytes            # data
	)	
	var current_layer_activations : PackedFloat32Array
	var layer_index : int = 0
	while layer_index < layers.size():
		if layer_index != 0:
			var previous_layer_activations_bytes : PackedByteArray = current_layer_activations.to_byte_array()
			Global.rendering_device.buffer_update(
				previous_layer_activations_buffer,
				0,
				previous_layer_activations_bytes.size(),
				previous_layer_activations_bytes
			)
		var current_layer_biases_bytes : PackedByteArray = PackedFloat32Array(biases[layer_index]).to_byte_array()
		Global.rendering_device.buffer_update(
			current_layer_biases_buffer,
			0,
			current_layer_biases_bytes.size(),
			current_layer_biases_bytes
		)
		var _2d_weights_array : Array = weights[layer_index]
		var flattened_weights : PackedFloat32Array = PackedFloat32Array()
		
		#var flattened_weights : PackedFloat32Array
	return input

func init_compute_pipeline() -> void:
	# Determine maximum layer size (for activations/biases)
	var max_layer_size := 0
	for layer in layers:
		if layer > max_layer_size:
			max_layer_size = layer
	var max_layer_size_bytes := max_layer_size * 4
	var empty_data := PackedByteArray()
	empty_data.resize(max_layer_size_bytes)

	# Allocate activation & bias buffers
	previous_layer_activations_buffer = Global.rendering_device.storage_buffer_create(max_layer_size_bytes, empty_data)
	current_layer_activations_buffer = Global.rendering_device.storage_buffer_create(max_layer_size_bytes, empty_data)
	current_layer_biases_buffer = Global.rendering_device.storage_buffer_create(max_layer_size_bytes, empty_data)

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
	
func generate_random_network(layers_in : PackedInt32Array) -> void:
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
			layer_biases.append(randf_range(-1, 1))

			# weights for this neuron
			var neuron_weights : Array = []
			for _w in range(previous_count):
				neuron_weights.append(randf_range(-1, 1))

			layer_weights.append(neuron_weights)

		weights.append(layer_weights)
		biases.append(layer_biases)

	#print("weights:", weights)
	print("biases:", biases)
	#print("layers: ", layers)


func mutate_network(mutation_rate : float) -> void:
	return
