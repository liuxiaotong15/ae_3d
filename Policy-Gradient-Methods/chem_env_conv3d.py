import itertools
import math
import random
import numpy as np

from gym.spaces.box import Box

from ase import Atoms
from ase import Atom
from ase.visualize import view
from ase.io import read, write
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.vasp import Vasp
from ase.calculators.morse import MorsePotential
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton, BFGS
from ase.constraints import FixAtoms


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
state_atoms = Atoms()
voxel_side_cnt = 50
side_len = 5

def atoms2voxels(at):
    # 50*50*50 voxel returned
    sigma = 1
    # volume = np.random.rand(voxel_side_cnt, voxel_side_cnt, voxel_side_cnt)
    volume = np.zeros((1, voxel_side_cnt, voxel_side_cnt, voxel_side_cnt), dtype=float)
    for idx in range(len(at)):
        for i, j, k in itertools.product(range(voxel_side_cnt),
                                        range(voxel_side_cnt),
                                        range(voxel_side_cnt)):
            x, y, z = i/voxel_side_cnt * side_len, j/voxel_side_cnt * side_len, k/voxel_side_cnt * side_len
            pow_sum = (x-at[idx].position[0])**2 + (y-at[idx].position[1])**2 + (z-at[idx].position[2])**2
            volume[0][i][j][k] += math.exp(-1*pow_sum/(2*sigma**2))
    return volume

def reset():
    global state_voxels, state_atoms
    state_atoms = Atoms()
    observation_space = Box(low=np.array(
        [-1]*stt_sz), high=np.array([1]*stt_sz),
        dtype=np.float32)  # don't care about observation_space low and high
    # DONE: return 1 atom at the center of the box
    state_atoms.append(Atom('Au', ( side_len * 0.5 , side_len * 0.5, side_len * 0.5 )))
    state_voxels = atoms2voxels(state_atoms)
    return state_voxels

def render():
    pass

def step(action):
    global state_voxel, state_atoms
    # TODO: 1. find the xyz position
    # print(action.shape, np.amax(action), np.sum(action))
    result = np.where(action == np.amax(action))
    # print(len(result))
    # print(result[0].shape)
    # print(result)
    x, y, z = result[1][0]/voxel_side_cnt, result[2][0]/voxel_side_cnt, result[3][0]/voxel_side_cnt
    # print(x, y, z)
    morse_calc = MorsePotential()
    state_atoms.set_calculator(morse_calc)
    orig_engy = state_atoms.get_potential_energy()
    # DONE: 2. add new atom to state
    state_atoms.append(Atom('Au', ( side_len * x, side_len * y, side_len * z )))
    # print('state atoms count: ', len(state_atoms))
    state_voxels = atoms2voxels(state_atoms)
    # DONE: 3. calculate the reward of the action
    state_atoms.set_calculator(morse_calc)
    next_engy = state_atoms.get_potential_energy()
    reward = orig_engy - next_engy
    reward = max(0, reward)
    done = False
    if len(state_atoms) == 10:
        done = True
    msg = 'test ok...'
    # reward = random.random()
    # print(action)
    return state_voxels, reward, done, msg
