import h5py
import os
import random
import math
import itertools
import numpy as np
import time
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

voxel_side_cnt = 50
side_len = 5 
seed = 1234
random.seed(seed)
np.random.seed(seed)

def generate_atoms_random():
    atoms = Atoms()
    for _ in range(5):
        atoms.append(Atom('Au', ( side_len * random.random(), side_len * random.random(), side_len * random.random())))
    return atoms

def cal_atoms_energy(at):
    morse_calc = MorsePotential()
    at.set_calculator(morse_calc)
    return at.get_potential_energy()

def atoms2array1d(at):
    sigma = 1
    # volume = np.random.rand(voxel_side_cnt, voxel_side_cnt, voxel_side_cnt)
    volume = np.zeros((voxel_side_cnt, voxel_side_cnt, voxel_side_cnt), dtype=float)
    for idx in range(len(at)):
        for i, j, k in itertools.product(range(voxel_side_cnt),
                                        range(voxel_side_cnt),
                                        range(voxel_side_cnt)):
            x, y, z = i/voxel_side_cnt * side_len, j/voxel_side_cnt * side_len, k/voxel_side_cnt * side_len
            pow_sum = (x-at[idx].position[0])**2 + (y-at[idx].position[1])**2 + (z-at[idx].position[2])**2
            volume[i][j][k] += math.exp(-1*pow_sum/(2*sigma**2))
    # import view2d
    # view2d.view2d(volume, voxel_side_cnt)
    return list(volume.reshape(-1))

def save_atom_enregy_h5(at_lst):
    idx = 0
    os.system('rm -rf ' + 'dataset_' + str(idx+1) + '.hdf5')
    f = h5py.File('dataset_' + str(idx+1) + '.hdf5', 'w')
    f.create_group('/grp1') # or f.create_group('grp1')
    f.create_dataset('dset1', compression='gzip', data=np.array(at_lst)) # or f.create_dataset('/dset1', data=data)
    f.close()

def main():
    dataset = []
    for i in range(100):
        print(i)
        at = generate_atoms_random()
        tmp = atoms2array1d(at)
        tmp.append(cal_atoms_energy(at))
        dataset.append(tmp)
        # time.sleep(0.01)
    # print(dataset)
    save_atom_enregy_h5(dataset)

# write to HDF5
'''
idx = 0
ret = []
size = 3
volume = np.random.rand(size, size, size)
print(volume)
volume = list(volume.reshape(-1))
volume.append(2.331)
os.system('rm -rf ' + 'dataset_new_' + str(idx+1) + '.hdf5')
f = h5py.File('dataset_new_' + str(idx+1) + '.hdf5', 'w')
f.create_group('/grp1') # or f.create_group('grp1')
f.create_dataset('dset1', compression='gzip', data=np.array(volume)) # or f.create_dataset('/dset1', data=data)
f.close()
'''
# read from HDF5
'''
f = h5py.File('dataset_new_' + str(idx+1) + '.hdf5', 'r')
# print('start to load data.')
dataset = f['dset1'][:]

print(dataset)
v = np.array(dataset[:-1])
v = v.reshape(size,size,size)
print(v)
print(dataset[-1])
f.close()
'''

if __name__ == '__main__':
    main()