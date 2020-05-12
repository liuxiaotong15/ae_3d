import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
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

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    side_len = 5
    stt_sz = 50
    max_atoms_count = 7
    
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
    
    state_voxels = np.zeros((4, stt_sz, stt_sz, stt_sz))
    state_atoms = Atoms()
    voxel_side_cnt = stt_sz
    # state_shap = 250
    def atoms2voxels(self, at):
        # 50*50*50 voxel returned
        sigma = 1
        voxel_side_cnt = self.voxel_side_cnt
        side_len = self.side_len
        # volume = np.random.rand(voxel_side_cnt, voxel_side_cnt, voxel_side_cnt)
        volume = np.zeros((4, voxel_side_cnt, voxel_side_cnt, voxel_side_cnt), dtype=float)
        for i, j, k in itertools.product(range(voxel_side_cnt),
                range(voxel_side_cnt),
                range(voxel_side_cnt)):
            # volume[0][i][j][k] = i/voxel_side_cnt
            # volume[1][i][j][k] = j/voxel_side_cnt
            # volume[2][i][j][k] = k/voxel_side_cnt
            dis_lst = []
            for idx in range(len(at)):
                x, y, z = i/voxel_side_cnt * side_len, j/voxel_side_cnt * side_len, k/voxel_side_cnt * side_len
                pow_sum = (x-at[idx].position[0])**2 + (y-at[idx].position[1])**2 + (z-at[idx].position[2])**2
                volume[-1][i][j][k] += math.exp(-1*pow_sum/(2*sigma**2))
                dis_lst.append(math.exp(-1*pow_sum/(2*sigma**2)))
            volume[0][i][j][k] = np.std(np.array(dis_lst))
            volume[1][i][j][k] = np.amax(np.array(dis_lst))
            volume[2][i][j][k] = np.amin(np.array(dis_lst))
        volume[-1] /= np.amax(volume[-1])
        volume[-1] = 1/(1+exp(-10*(volume[-1]-0.5)))
        if np.amax(volume[0]) > 0:
            volume[0] /= np.amax(volume[0])
        # print('max: ', np.amax(volume), 'min: ', np.amin(volume), 'mean: ', np.average(volume), 'atoms cnt: ', len(at))
        return volume

    def reset(self):
        side_len = self.side_len
        self.state_atoms = Atoms()
        # observation_space = Box(low=np.zeros((1, stt_sz, stt_sz, stt_sz)), high=np.zeros((1, stt_sz, stt_sz, stt_sz)) + max_atoms_count, dtype=np.float32) 
        # DONE: return 1 atom at the center of the box
        self.state_atoms.append(Atom('Au', ( side_len * 0.5 , side_len * 0.5, side_len * 0.5 )))
        self.state_voxels = self.atoms2voxels(self.state_atoms)
        return self.state_voxels
        # return np.zeros((self.state_shap,))


    def __init__(self, g=10.0):
        self.side_len = 5
        stt_sz = self.stt_sz
        self.viewer = None
        # high = np.ones((250,)) * self.side_len
        # low = np.zeros((250,))
        
        # high = np.array([1., 1., self.max_speed], dtype=np.float32)
        high = np.ones((4, stt_sz, stt_sz, stt_sz)) * self.max_atoms_count
        low = np.zeros((4, stt_sz, stt_sz, stt_sz))
        self.action_space = spaces.Box(low=0,  high=self.max_atoms_count , shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # depreciated: old reward function which make relations with the differential of the energy...
    # def step(self, u):
    #     done = False
    #     side_len = self.side_len
    #     x, y, z = u[0], u[1], u[2]
    #     # DONE: 2. add new atom to state
    #     morse_calc = MorsePotential()
    #     self.state_atoms.set_calculator(morse_calc)
    #     orig_engy = self.state_atoms.get_potential_energy()
    #     self.state_atoms.append(Atom('Au', ( side_len * x, side_len * y, side_len * z )))
    #     self.state_voxels = self.atoms2voxels(self.state_atoms)
    #     # DONE: 3. calculate the reward of the action
    #     self.state_atoms.set_calculator(morse_calc)
    #     next_engy = self.state_atoms.get_potential_energy()
    #     reward = orig_engy - next_engy
    #     if reward < 0.5:
    #         done = True
    #     reward = max(0, reward)
    #     if len(self.state_atoms) == self.max_atoms_count:
    #         done = True
    #     msg = 'test ok...'
    #     return self.state_voxels, reward, done, msg

    def step(self, u):
        done = False
        side_len = self.side_len
        x, y, z = u[0], u[1], u[2]
        # DONE: 2. add new atom to state
        morse_calc = MorsePotential()
        self.state_atoms.append(Atom('Au', ( side_len * x, side_len * y, side_len * z )))
        self.state_voxels = self.atoms2voxels(self.state_atoms)
        # DONE: 3. calculate the reward of the action
        atom_cnt = len(self.state_atoms)
        self.state_atoms.set_calculator(morse_calc)
        engy = self.state_atoms.get_potential_energy()
        
        # reward is between 0-1, so sigmoid as activation func is enough
        reward = (self.std_ans_morse_clst[atom_cnt-2] - engy)/(self.std_ans_morse_clst[atom_cnt-2] - self.std_ans_morse_clst[atom_cnt-1])

        if atom_cnt == self.max_atoms_count or reward < 0:
            done = True

        reward = max(0, reward)
        msg = 'test ok...'
        return self.state_voxels, reward, done, msg

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # def step(self, u):
    #     high = np.ones((250,)) * self.side_len
    #     print("step of chem env")
    #     return high, 100, False, {}

    # def reset(self):
    #     high = np.ones((250,)) * self.side_len
    #     self.state = self.np_random.uniform(low=-high, high=high)
    #     self.last_u = None
    #     # high = np.array([np.pi, 1])
    #     # self.state = self.np_random.uniform(low=-high, high=high)
    #     # self.last_u = None
    #     return high

    # def _get_obs(self):
    #     theta, thetadot = self.state
    #     return np.array([np.cos(theta), np.sin(theta), thetadot])

    # def render(self, mode='human'):
    #     pass

    # def close(self):
    #     pass

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
