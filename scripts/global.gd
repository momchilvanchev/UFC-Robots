extends Node

### Autoload:
### This node is instantiated once at game startup and persists for the
### entire runtime. It is never freed, and its data is globally accessible
### from any script or scene.
var rendering_device: RenderingDevice
var shader_file : Resource
var shader_spirv: RDShaderSPIRV
var shader : RID
var pipeline : RID
func _ready() -> void:
	rendering_device = create_new_rendering_device()
	print(rendering_device)
	shader_file = load("res://scripts/shaders/compute_shaders/ffnn_layer_compute.glsl")
	print(shader_file)
	shader_spirv = shader_file.get_spirv()
	print(shader_spirv)
	shader = rendering_device.shader_create_from_spirv(shader_spirv)
	print(shader)
	pipeline = rendering_device.compute_pipeline_create(shader)
	print("- Global Singleton Loaded")

func create_new_rendering_device() -> RenderingDevice:
	return RenderingServer.create_local_rendering_device()
