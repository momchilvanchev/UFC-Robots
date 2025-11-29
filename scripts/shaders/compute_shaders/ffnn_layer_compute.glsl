#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 1) in;

layout(set = 0, binding = 0, std430) restrict buffer PreviousLayerActivations {
	float activations[];
}
previous_layer_activations;

layout(set = 0, binding = 1, std430) restrict buffer CurrentLayerActivations {
	float activations[];
}
current_layer_activations;

layout(set = 0, binding = 2, std430) restrict buffer CurrentLayerBiases {
	float biases[];
}
current_layer_biases;

layout(set = 0, binding = 3, std430) restrict buffer CurrentLayerWeights {
	float weights[]; // Flattened array
}
current_layer_weights;


void main() {
	uint id = gl_GlobalInvocationID.x;
	current_layer_activations.activations[id] += current_layer_biases.biases[id];
}
