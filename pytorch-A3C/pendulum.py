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

    side_len = 2 # A
    max_atoms_count = 4
    
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
    
    state_atoms = Atoms()

    def reset(self):
        side_len = self.side_len
        self.state_atoms = Atoms()
        # DONE: return 1 atom at the center of the box
        self.state_atoms.append(Atom('Au', ( side_len * 0.5 , side_len * 0.5, side_len * 0.5 )))
        return self.state_atoms
        # return np.zeros((self.state_shap,))


    def __init__(self, g=10.0):
        # self.side_len = 2
        self.viewer = None
        
        # action space is xyz output from NN.
        self.action_space = spaces.Box(low=0,  high=self.side_len, shape=(3,), dtype=np.float32)

        # observation space is an Atom list. It will be process outside. 
        # here is just a placeholder.
        high = np.ones(self.max_atoms_count)
        low = np.zeros(self.max_atoms_count)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, u):
        done = False
        side_len = self.side_len
        x, y, z = u[0], u[1], u[2]
        # DONE: 2. add new atom to state
        morse_calc = MorsePotential()
        self.state_atoms.append(Atom('Au', ( side_len * x, side_len * y, side_len * z )))
        # DONE: 3. calculate the reward of the action
        atom_cnt = len(self.state_atoms)
        self.state_atoms.set_calculator(morse_calc)
        engy = self.state_atoms.get_potential_energy()

        valid = (engy)/(self.std_ans_morse_clst[atom_cnt-1])
        try:
            reward = math.exp(-1 * engy)/math.exp(-1 * self.std_ans_morse_clst[atom_cnt-1])
        except OverflowError:
            reward = 0

        if atom_cnt == self.max_atoms_count or valid < 0:
            done = True

        msg = 'Good_luck!!!'
        return self.state_atoms, reward, done, msg

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

