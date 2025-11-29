#[compute]
#version 450

layout(local_size_x = 1) in;

layout(set = 0, binding = 0, std430) restrict buffer PreviousLayerActivations {
	float activations[]; // activations[0] holds the layer size
}
previous_layer_activations;

layout(set = 0, binding = 1, std430) restrict buffer CurrentLayerActivations {
	float activations[]; // activations[0] holds the layer size
}
current_layer_activations;

layout(set = 0, binding = 2, std430) restrict buffer CurrentLayerBiases {
	float biases[];
}
current_layer_biases;

layout(set = 0, binding = 3, std430) restrict buffer CurrentLayerWeights {
	float weights[]; // Flattened: (neuron_index * previous_layer_length + weight_index)
}
current_layer_weights;

void main() {
	uint id = gl_GlobalInvocationID.x + 1;
	current_layer_activations.activations[id] = 0; // clean up from previous compute invocations
	// Read layer size (stored at index 0)
	int previous_layer_length = int(previous_layer_activations.activations[0]);

	// Weighted sum
	float sum = 0.0;
	for (int i = 0; i < previous_layer_length; i++) {
		// Shift by +1 because activations[0] is length
		float previous_activation = previous_layer_activations.activations[i + 1];
		float weight = current_layer_weights.weights[id * previous_layer_length + i];
		sum += previous_activation * weight;
	}

	// Add bias
	sum += current_layer_biases.biases[id];

	current_layer_activations.activations[id] = tahn(sum);
}
