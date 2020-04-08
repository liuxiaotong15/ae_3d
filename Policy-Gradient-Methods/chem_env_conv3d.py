import random
import numpy as np

from gym.spaces.box import Box

seed = 1234
random.seed(seed)
np.random.seed(seed)

stt_sz = 50

observation_space = Box(low=np.array([-1]*stt_sz), high=np.array([1]*stt_sz), dtype=np.float32) 
action_space = Box(low=np.array([-1]*3), high=np.array([1]*3), dtype=np.float32)

# The answer of morse cluster: doye.chem.ox.ac.uk/jon/structures/Morse/tables.html
global_min_energy = [0] * 80
std_ans_morse_clst = \
    [0, -1, -3, -6,  -9.044930, -12.487810, -16.207580, -19.327420, -23.417190, -27.473283, # 1-10
    -31.521880, -36.400278, -42.439863, -45.619277, -49.748409, -53.845835, -57.941386, -62.689245, -68.492285, -72.507782, #11-20
    -76.529139, -81.136735, -86.735494, -90.685398, -95.127899, -100.549598, -104.745275, -108.997831, -114.145949, -118.432844, #21-30
    -122.857743, -127.771395, -132.287431, -136.797544, -141.957188, -147.381965, -151.891203, -157.477108, -163.481990, -167.993097, #31-40
    -172.526828, -177.680222, -183.092699, -187.626292, -192.954739, -199.177751, -203.704178, -209.044000, -215.253702, -219.820229, #41-50
    -225.391240, -231.615013, -237.834976, -244.058174, -250.286609, -253.922955, -258.041717, -263.410755, -267.945226, -273.341243, #51-60
    -278.726626, -283.183002, -288.560948, -293.931716, -298.392345, -303.763297, -309.130322, -314.374880, -319.819905, -325.887749, #61-70
    -331.588748, -336.121753, -341.266253, -346.610834, -351.472365, -356.372708, -361.727086, -367.0722648, -372.832290, -378.333471, #71-80
]

state_voxels = np.zeros((1, stt_sz, stt_sz, stt_sz))

def atoms2voxels(atoms):
    # 50*50*50 voxel returned
    pass

def reset():
    global state_voxels
    observation_space = Box(low=np.array(
        [-1]*stt_sz), high=np.array([1]*stt_sz),
        dtype=np.float32)  # don't care about observation_space low and high
    # TODO: return 1 atom at the center of the box
    return state_voxels

def render():
    pass

def step(action):
    global state_voxels
    # TODO: 1. find the xyz position
    # TODO: 2. add new atom to state
    # TODO: 3. calculate the reward of the action
    done = True
    msg = 'test ok...'
    reward = random.random()
    # print(action)
    return state_voxels, reward, done, msg
