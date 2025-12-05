extends Node3D

func _input(event: InputEvent) -> void:
	if event is InputEventKey:
		if event.pressed and not event.echo:
			match event.as_text():
				"Left":
					print("switch to previous camera")
				"Right":
					print("switch to next camera")
