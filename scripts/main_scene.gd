extends Node3D

@export var map_scene_link : PackedScene
@export var fighter_scene_link : PackedScene

var map_scene : Node3D
var fighter1 : Node3D
var fighter2 : Node3D

func _ready() -> void:
	map_scene = map_scene_link.instantiate()
	#fighter1 = fighter_scene_link.instantiate()
	#fighter1.position.y = 5
	#fighter1.position.z = 5
	#fighter2 = fighter_scene_link.instantiate()
	add_child(map_scene)
	#add_child(fighter1)
	#add_child(fighter2)
