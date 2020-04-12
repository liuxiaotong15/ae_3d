# ae_3d
try to make an autoencoder from 3d structure to 3d structure using conv3d

pip install torch gym h5py torchvision ase

some ref url:

1. autoencoder: https://blog.csdn.net/yuyangyg/article/details/80054121

2. conv3d: https://blog.csdn.net/FrontierSetter/article/details/99888787

3. 3d visualizer: https://stackoverflow.com/questions/45969974/what-is-the-most-efficient-way-to-plot-3d-array-in-python

and after run ok of the ae, I continue the work to DRL with conv3d, some ref url about DRL:

1. a good sample: https://github.com/liuxiaotong15/Reinforcement-learning

2. the previou repo is no longer maintained and move to

Algorithms from the Q learning family will be moved to https://github.com/cyoon1729/deep-Q-networks.

Algorithms from the PG family will be moved https://github.com/cyoon1729/Policy-Gradient-Methods. cause my output is continurous, so I develope based on this repo.

3. A3C from this repo: https://github.com/MorvanZhou/pytorch-A3C

I modify the 'pendulum' of gym, so make a symbol link: 

cd ../ae_venv/lib/python3.7/site-packages/gym/envs/classic_control

ln -s ~/code/ae_3d/pytorch-A3C/pendulum.py pendulum.py

cd pytorch-A3C

python xiaotong_continuous_A3C.py

and if want to re-run the real 'pendulum', we need to move it back or make a symbol link to the real file.
