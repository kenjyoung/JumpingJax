import jax as jx
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
import argparse
import time

from substances import substance_objects, reaction_objects

import imageio

import pygame
# This file attempts to implement simple falling sand game rules on GPU

world_size = 500
#this is is simple way to ensure all index partition entries are the same length which makes it convenient to store as a jnp array
world_size = 6*(world_size//6)

# partition cell indices into sets which can operate in parallel without collision, note this assumes the world map is padded by an extra cell all around and excludes boundaries
movement_index_partitions = []
#tile layout:
#1,2,3#
#4,5,6#
# j/n is vertical i/m is horizontal
for n in range(2):
    for m in range(3):
        p = [(j+1) * (world_size+2) + i+1 for i in range(n, world_size, 2) for j in range(m, world_size, 3)]
        movement_index_partitions += [p]
movement_index_partitions = jnp.asarray(movement_index_partitions)

m = jnp.zeros((world_size+2,world_size+2))
for i, p in enumerate(movement_index_partitions):
    m = m.at[jnp.unravel_index(p, (world_size+2, world_size+2))].set(i+1)

binary_reaction_index_partitions = []\
#tile layout:
#1,2,3,4,5,6#
#5,6,1,2,3,4#
#3,4,5,6,1,2#
# j/n is vertical i/m is horizontal
for m in range(6):
    p = sum([[(j+1) * (world_size+2) + i+1 for j in range((m+2*i)%6, world_size, 6)] for i in range(world_size)],[])
    binary_reaction_index_partitions += [p]
binary_reaction_index_partitions = jnp.asarray(binary_reaction_index_partitions)

#unary reactions only effect a single cell so we can perform them all at once
unary_reaction_index_partitions = [jnp.asarray([(j+1) * (world_size+2) + i+1 for j in range(world_size) for i in range(world_size)])]

screen_size = (1000, 1000)

pygame.init()
display = pygame.display.set_mode(screen_size)
font = pygame.font.Font('freesansbold.ttf', 24)

parser = argparse.ArgumentParser()
parser.add_argument("--record", "-r", action='store_true', default=False)
parser.add_argument("--seed", "-s", type=int, default=0)
args = parser.parse_args()

movement_types = {'fixed': 0, 'powder': 1, 'fluid': 2, 'left_fluid': 3, 'right_fluid': 4}

substances = {s.name:i for i, s in enumerate(substance_objects)}
substance_names = [s.name for s in substance_objects]
num_substances = len(substances)
color_array = jnp.asarray([jnp.asarray(s.color) for s in substance_objects], dtype=jnp.uint8)

properties = {'movement': 0, 'mass': 1}
substance_properties = [{"movement":movement_types[s.movement_type], "mass": s.mass} for s in substance_objects]
properties_array = jnp.asarray([[sp[p] for p in properties] for sp in substance_properties])

binary_substance_reactions = []
for s in substance_objects:
    binary_substance_reactions += [[]]
    for t in substance_objects:
        binary_substance_reactions[-1] += [[]]
        for r in reaction_objects:
            if(len(r.inputs)==2 and r.inputs[0]==s.name and r.inputs[1]==t.name):
                binary_substance_reactions[-1][-1]+=[r]

binary_reaction_weights = jnp.asarray([[sum([r.weight for r in binary_substance_reactions[i][j]]) for j in range(num_substances)] for i in range(num_substances)])

unary_substance_reactions = []
for s in substance_objects:
    unary_substance_reactions += [[]]
    for r in reaction_objects:
        if(r.inputs[0]==s.name and len(r.inputs)==1):
            unary_substance_reactions[-1]+=[r]

def move_strings_to_nums(m):
    return [jnp.asarray([move[n] for n in p]) for p in m]


move = {'n': 0, 'l': 1, 'ld': 2, 'd': 3, 'rd': 4, 'r': 5, 'ru': 6, 'u': 7,'lu':8}

# moves in same inner list have same priority and will be randomized
powder_moves = move_strings_to_nums([['d'], ['ld', 'rd']])
default_fluid_moves = move_strings_to_nums([['d'], ['ld', 'rd'], ['l', 'r']])
left_fluid_moves = move_strings_to_nums([['d'], ['l'], ['ld', 'rd'], ['r']])
right_fluid_moves = move_strings_to_nums([['d'], ['r'], ['ld', 'rd'], ['l']])
fixed_moves = move_strings_to_nums([])

relative_position_array = jnp.asarray([(0, 0), (-1, 0), (-1, -1), (0, -1), (1, -1), (1,0), (1, 1), (0, 1), (-1,1)])

key = jx.random.PRNGKey(args.seed)

substance_map = jnp.full((world_size, world_size), substances['empty'], dtype=int)

#stores fluid momentum (-1=left, 1=right or 0=none) for each cell
momentum_map = jnp.zeros((world_size, world_size), dtype=int)


def padded_map(_map, pad_val):
    padded_shape = (_map.shape[0] + 2, _map.shape[1] + 2) if _map.ndim == 2 else (_map.shape[0] + 2, _map.shape[1] + 2, _map.shape[2:])
    padded = jnp.full(padded_shape, pad_val, dtype=_map.dtype)
    padded = padded.at[1:-1, 1:-1].set(_map)
    return padded
padded_map = jit(padded_map)

def get_select_move_function(moves):
    def select_move(index, key, density_map, fixed_map, substance_map):
        selection_made = jnp.asarray(False)
        m = move['n']
        for group in moves:
            key, subkey = jx.random.split(key)
            candidate_move = jx.random.choice(subkey, group)
            candidate_index = index - relative_position_array[candidate_move]
            m = jnp.where(
                jnp.logical_and(
                    jnp.logical_and(
                        jnp.logical_not(fixed_map[candidate_index[0],candidate_index[1]]),
                        density_map[candidate_index[0],candidate_index[1]] < density_map[index[0],index[1]]
                    ),
                    jnp.logical_not(selection_made)
                ),
                candidate_move,
                m
            )
            selection_made = jnp.logical_or(selection_made,
                                            jnp.logical_and(
                                                jnp.logical_not(
                                                    fixed_map[candidate_index[0],candidate_index[1]]),
                                                density_map[candidate_index[0],candidate_index[1]] < density_map[index[0],index[1]]
                                            ),
                                            )
        dest_index = index - relative_position_array[m]
        dest_substance = substance_map[dest_index[0],dest_index[1]]
        return index, dest_substance, dest_index, substance_map[index[0],index[1]], m
    return jit(select_move)

select_fixed_move = get_select_move_function(fixed_moves)
select_powder_move = get_select_move_function(powder_moves)
select_left_fluid_move = get_select_move_function(left_fluid_moves)
select_right_fluid_move = get_select_move_function(right_fluid_moves)
select_default_fluid_move = get_select_move_function(default_fluid_moves)

def select_move(flat_index, key, density_map, fixed_map, substance_map, movement_map):
    index = jnp.asarray(jnp.unravel_index(flat_index, density_map.shape))

    return jx.lax.switch(movement_map[index[0],index[1]],
                         (select_fixed_move, 
                          select_powder_move, 
                          select_default_fluid_move,
                          select_left_fluid_move,
                          select_right_fluid_move), index, key, density_map, fixed_map, substance_map)
select_moves = jit(vmap(select_move, in_axes=(0, 0, None, None, None, None)))

def get_binary_reaction_function(substance_1, substance_2):
    def select_binary_reaction(index, key, substance_map):
        outputs_1 = jnp.asarray([substances[r.outputs[0]] for r in binary_substance_reactions[substance_1][substance_2]])
        outputs_2 = jnp.asarray([substances[r.outputs[1]] for r in binary_substance_reactions[substance_1][substance_2]])
        weights = jnp.asarray([r.weight for r in binary_substance_reactions[substance_1][substance_2]])
        key, subkey = jx.random.split(key)
        if(len(weights)>0):
            selection = jx.random.choice(subkey,len(weights), p=weights/jnp.sum(weights))
            local_substance = outputs_1[selection]
            dest_substance = outputs_2[selection]
        else:
            local_substance = substance_1
            dest_substance = substance_2
        return local_substance, dest_substance
    return jit(select_binary_reaction)

binary_reaction_functions = tuple(tuple(get_binary_reaction_function(i, j) for j in range(num_substances)) for i in range(num_substances))

def get_binary_pair_function(substance):
    def select_binary_pair(index, key, substance_map):
        candidate_move_array = jnp.array([relative_position_array[move['u']],
                                          relative_position_array[move['r']],
                                          relative_position_array[move['d']],
                                          relative_position_array[move['l']]])
        dest_indices = index+candidate_move_array
        dest_substances = substance_map[dest_indices[:,0],dest_indices[:,1]]

        #No reaction gets a default weight of 1.0
        weights = jnp.concatenate((binary_reaction_weights[substance][dest_substances],jnp.array([1.0])))

        key, subkey = jx.random.split(key)
        selection = jx.random.choice(subkey, len(candidate_move_array)+1, p=weights/jnp.sum(weights))
        dest_index = jnp.where(jnp.equal(selection,4), index, dest_indices[selection])

        output_1, output_2 = jx.lax.switch(dest_substances[selection], binary_reaction_functions[substance], index, key, substance_map)

        #seperately handle case where no reaction is selected
        local_substance = jnp.where(selection==4, substance, output_1)
        dest_substance = jnp.where(selection==4, substance, output_2)

        return index, local_substance, dest_index, dest_substance
    return jit(select_binary_pair)
binary_pair_functions = tuple(get_binary_pair_function(i) for i in range(num_substances))

def select_binary_reaction(flat_index, key, substance_map):
    index = jnp.asarray(jnp.unravel_index(flat_index, substance_map.shape))
    return jx.lax.switch(substance_map[index[0],index[1]], binary_pair_functions, index, key, substance_map)
select_binary_reactions = jit(vmap(select_binary_reaction, in_axes=(0, 0, None)))

def get_unary_reaction_function(substance):
    def select_unary_reaction(index, key, substance_map):
        #No reaction gets a default weight of 1.0
        outputs = jnp.asarray([substances[r.outputs[0]] for r in unary_substance_reactions[substance]]+[substance])
        weights = jnp.asarray([r.weight for r in unary_substance_reactions[substance]]+[1.0])
        key, subkey = jx.random.split(key)
        local_substance = jx.random.choice(subkey,outputs, p=weights/jnp.sum(weights))
        return index, local_substance
    return jit(select_unary_reaction)

unary_reaction_functions = tuple(get_unary_reaction_function(i) for i in range(num_substances))

def select_unary_reaction(flat_index, key, substance_map):
    index = jnp.asarray(jnp.unravel_index(flat_index, substance_map.shape))
    return jx.lax.switch(substance_map[index[0],index[1]],unary_reaction_functions, index, key, substance_map)
select_unary_reactions = jit(vmap(select_unary_reaction, in_axes=(0, 0, None)))


def step_sim(key, substance_map, momentum_map):
    # pad substance map with a solid boundary to prevent movement offscreen
    #TODO: this is sub ideal since it depends on stone being defined, and probably inert to avoid boundary weirdness
    substance_map = padded_map(substance_map, substances['stone'])
    momentum_map = padded_map(momentum_map, 0)
    key, subkey = jx.random.split(key)
    permuted_movement_index_partitions = jx.random.permutation(subkey, movement_index_partitions)
    key, subkey = jx.random.split(key)
    permuted_binary_reaction_index_partitions = jx.random.permutation(subkey,binary_reaction_index_partitions)

    def movement_loop_func(substance_map_momentum_map_and_key, indices):
        substance_map, momentum_map, key = substance_map_momentum_map_and_key
        density_map = jnp.take(properties_array[:, properties['mass']], substance_map)
        movement_map = jnp.take(properties_array[:, properties['movement']], substance_map).astype(int)
        left_fluid_map = jnp.where(movement_map==movement_types['fluid'], momentum_map==-1, False)
        right_fluid_map = jnp.where(movement_map==movement_types['fluid'], momentum_map==1, False)
        movement_map = jnp.where(left_fluid_map, movement_types['left_fluid'], movement_map)
        movement_map = jnp.where(right_fluid_map, movement_types['right_fluid'], movement_map)
        fixed_map = jnp.take(properties_array[:, properties['movement']], substance_map) == movement_types['fixed']

        key, subkey = jx.random.split(key)
        subkeys = jx.random.split(subkey, len(indices))
        start_indices, start_substances, end_indices, end_substances, executed_move = select_moves(indices, subkeys, density_map, fixed_map, substance_map, movement_map)

        inds = jnp.concatenate((start_indices, end_indices))
        subs = jnp.concatenate((start_substances, end_substances))

        substance_map = substance_map.at[inds[:, 0], inds[:, 1]].set(subs)

        momentum_map = momentum_map.at[end_indices[:,0], end_indices[:,1]].set(jnp.where(
            executed_move == move['l'], -1, jnp.where(executed_move == move['r'], 1, momentum_map[start_indices[:, 0], start_indices[:, 1]])))
        return (substance_map, momentum_map, key), None

    def binary_reaction_loop_func(substance_map_and_key, indices):
        substance_map, momentum_map, key = substance_map_and_key
        key, subkey = jx.random.split(key)
        subkeys = jx.random.split(subkey, len(indices))
        start_indices, start_substances, end_indices, end_substances = select_binary_reactions(indices, subkeys, substance_map)

        inds = jnp.concatenate((start_indices, end_indices))
        subs = jnp.concatenate((start_substances, end_substances))

        substance_map = substance_map.at[inds[:, 0], inds[:, 1]].set(subs)
        return (substance_map, momentum_map, key), None

    def unary_reaction_loop_func(substance_map_and_key, indices):
        substance_map, momentum_map, key = substance_map_and_key
        key, subkey = jx.random.split(key)
        subkeys = jx.random.split(subkey, len(indices))
        inds, subs = select_unary_reactions(indices, subkeys, substance_map)
        substance_map = substance_map.at[inds[:, 0], inds[:, 1]].set(subs)
        return (substance_map, momentum_map, key), None


    substance_map_and_key, _ = jx.lax.scan(movement_loop_func, (substance_map, momentum_map,key), permuted_movement_index_partitions)
    substance_map_and_key, _ = jx.lax.scan(binary_reaction_loop_func, substance_map_and_key, permuted_binary_reaction_index_partitions)
    substance_map_and_key, _ = unary_reaction_loop_func(substance_map_and_key, unary_reaction_index_partitions[0])
    substance_map, momentum_map, key = substance_map_and_key
    return substance_map[1:-1,1:-1], momentum_map[1:-1,1:-1]
step_sim = jit(step_sim, static_argnames=('steps',))

#User Interface Stuff
#================
def substance_map_to_img(substance_map):
    img = jnp.take(color_array, substance_map, axis=0)
    return img
substance_map_to_img = jit(substance_map_to_img)


def display_img(img):
    surf = pygame.surfarray.make_surface(np.asarray(img))
    surf = pygame.transform.scale(surf, screen_size)
    display.blit(surf, (0, 0))


selected_substance = 0
brush_size = 1
def place_pixels(x, y, substance_map, substance, brush_size):
    update =jnp.full((2*brush_size, 2*brush_size), substance, dtype=int)
    substance_map = jx.lax.dynamic_update_slice(substance_map, update, (x, y))
    return substance_map
place_pixels = jit(place_pixels, static_argnames=('brush_size',))


def handle_user_input(substance_map, selected_substance, brush_size):
    text = font.render("current substance: "+substance_names[selected_substance]+" | brush_size: "+str(brush_size), True, (255, 255, 255))
    display.blit(text, (0, 0))
    events = pygame.event.get()
    quit = False
    if(pygame.mouse.get_pressed()[0]):
        x, y = pygame.mouse.get_pos()
        x*=world_size/screen_size[0]
        y*=world_size/screen_size[1]
        x = int(x)
        y = int(y)
        x_start = jnp.clip(x-brush_size, 0, world_size-2*brush_size)
        y_start = jnp.clip(y-brush_size, 0, world_size-2*brush_size)
        substance_map = place_pixels(x_start,y_start, substance_map, selected_substance, brush_size)
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                selected_substance = (selected_substance+1)%num_substances
            elif event.key == pygame.K_RIGHT:
                selected_substance = (selected_substance-1)%num_substances
            elif event.key == pygame.K_UP:
                brush_size = min(brush_size*2,32)
            elif event.key == pygame.K_DOWN:
                brush_size = max(brush_size//2,1)
            elif event.key == pygame.K_q:
                quit = True

    return substance_map, selected_substance, brush_size, quit


#Simulation Loop
#===============
avg_fps = 0.0
quit = False
if(args.record):
    recorder = imageio.get_writer('gameplay.gif', mode='I')
print("Compiling...",end="\r")
while not quit:
    start = time.time()
    img = substance_map_to_img(substance_map)
    if(args.record):
        recorder.append_data(np.transpose(np.asarray(img),(1,0,2)))
    display_img(img)
    substance_map, selected_substance, brush_size, quit = handle_user_input(substance_map, selected_substance, brush_size)
    pygame.display.update()

    key, subkey = jx.random.split(key)
    substance_map, momentum_map = step_sim(subkey, substance_map, momentum_map)

    avg_fps = 0.99*avg_fps+0.01*(1/(time.time() - start))
    print("fps: "+str(avg_fps),end="\r")

if(args.record):
    recorder.close()
