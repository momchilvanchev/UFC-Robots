extends Node3D

@export var main_camera : Camera3D
@export var min_max_fov : Vector2i = Vector2i(30, 100)
@export var octagon_mesh_instance : MeshInstance3D
var octagon_camera_positions : PackedVector3Array
var octagon_mesh : CylinderMesh
var camera_octagon_position_index : int = 0
func _ready() -> void:
	octagon_mesh = octagon_mesh_instance.mesh
	var angle : float = 360 / 8.0
	var octagon_radius : float = octagon_mesh.top_radius
	var angle_offset : float = rad_to_deg(octagon_mesh_instance.global_rotation.y)
	var inner_angle : float = angle + angle_offset
	var height : float = 10
	for i in range(8):
		var x : float = octagon_mesh_instance.global_position.x + octagon_radius * cos(inner_angle + i * (2 * PI / 8))
		var z : float = octagon_mesh_instance.global_position.z + octagon_radius * sin(inner_angle + i * (2 * PI / 8))
		octagon_camera_positions.append(Vector3(x, height + octagon_mesh_instance.global_position.y, z))
		# calculate each point
	main_camera.position = octagon_camera_positions[camera_octagon_position_index]
func _process(delta: float) -> void:
	var target_look_position
	if Global.fighter1 != null:
		target_look_position = Global.fighter1.get_body_part(Global.fighter1.body_part.HEAD).global_position
	else:
		target_look_position = octagon_mesh_instance.global_position
	# make camera face the camera_look_dir
	main_camera.look_at(target_look_position, Vector3.UP)
func _input(event: InputEvent) -> void:
	if event is InputEventKey:
		if event.pressed:
			match event.as_text():
				"Left":
					print("left")
					camera_octagon_position_index += 1
					move_camera_position_index(1)
					main_camera.position = octagon_camera_positions[camera_octagon_position_index]
				"Right":
					print("right")
					move_camera_position_index(-1)
					main_camera.position = octagon_camera_positions[camera_octagon_position_index]
func move_camera_position_index(amount : int = 1) -> void:
	camera_octagon_position_index += amount
	if camera_octagon_position_index > 7:
		camera_octagon_position_index = 0
	elif camera_octagon_position_index < 0:
		camera_octagon_position_index = 7
