class Substance:
    def __init__(self, name, movement_type, mass, color):
        self.name = name
        self.movement_type = movement_type
        self.mass = mass
        self.color = color

class Reaction:
    def __init__(self, inputs, outputs, weight):
        self.inputs = inputs
        self.outputs = outputs
        self.weight = weight


substance_objects = [Substance('water', 'fluid', 0.5, [67, 100, 176]),
                     #Note: removing empty will cause script to crash
                     Substance('empty', 'fluid', 0.0, [0,0,0]),
                     Substance('sand', 'powder', 1.0, [138, 131, 66]),
                     Substance('dust', 'powder', 0.1, [186, 186, 184]),
                     Substance('ember', 'powder', 0.6, [235, 79, 52]),
                     Substance('fish', 'fluid', 0.5, [174, 52, 235]),
                     Substance('dry_fish', 'fluid', 0.5, [171, 122, 196]),
                     Substance('steam', 'fluid', -0.3, [84, 120, 176]),
                     Substance('oil', 'fluid', 0.2, [51, 31, 23]),
                     Substance('gas', 'fluid', -0.5, [128, 156, 128]),
                     Substance('fire', 'fluid', -0.5, [252, 73, 3]),
                     Substance('torch', 'fixed', 1.0, [168, 90, 50]),
                     Substance('spout', 'fixed', 1.0, [44, 19, 237]),
                     Substance('drain', 'fixed', 1.0, [81, 76, 117]),
                     #Note: removing stone will cause script to crash,
                     #adding interactions with it will break boundary behaviour
                     Substance('stone', 'fixed', 1.0, [87, 89, 94]),
                     Substance('plant', 'fixed', 1.0, [3, 252, 69])]

reaction_objects = [Reaction(('plant','water'), ('plant', 'plant'), 0.1),
                    Reaction(('fish','water'), ('water', 'fish'), 10),
                    Reaction(('dry_fish','water'), ('fish', 'water'), 10),
                    Reaction(('dry_fish','empty'), ('empty', 'dry_fish'), 10),
                    Reaction(('plant','fire'), ('ember', 'fire'), 0.1),
                    Reaction(('gas','fire'), ('fire', 'fire'), 1.0),
                    Reaction(('water','fire'), ('steam', 'empty'), 1.0),
                    Reaction(('oil','fire'), ('fire', 'fire'), 1.0),
                    Reaction(('dust','fire'), ('fire', 'fire'), 10),
                    Reaction(('dust','ember'), ('fire', 'ember'), 10),
                    Reaction(('ember', 'empty'), ('ember', 'fire'),0.1),
                    Reaction(('ember', 'plant'), ('ember', 'ember'),0.1),
                    Reaction(('ember', 'oil'), ('ember', 'fire'),0.1),
                    Reaction(('ember', 'gas'), ('ember', 'fire'),0.1),
                    Reaction(('ember', 'dust'), ('ember', 'fire'),10),
                    Reaction(('ember', 'water'), ('ember', 'steam'),0.1),
                    Reaction(('torch', 'empty'), ('torch', 'fire'),0.1),
                    Reaction(('torch', 'plant'), ('torch', 'fire'),0.1),
                    Reaction(('torch', 'oil'), ('torch', 'fire'),0.1),
                    Reaction(('torch', 'gas'), ('torch', 'fire'),0.1),
                    Reaction(('torch', 'dust'), ('torch', 'fire'),10),
                    Reaction(('torch', 'water'), ('torch', 'steam'),0.1),
                    Reaction(('spout', 'empty'), ('spout', 'water'),0.1),
                    Reaction(('drain', 'water'), ('drain', 'empty'),0.1),
                    Reaction(('fire',), ('empty',), 0.03),
                    Reaction(('ember',), ('empty',), 0.01),
                    Reaction(('steam',), ('water',), 0.01),
                    Reaction(('fish',), ('dry_fish',), 0.03),
                    Reaction(('dry_fish',), ('dust',), 0.03)]
