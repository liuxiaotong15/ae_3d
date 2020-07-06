"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import torch
import torch.nn as nn
from utils_xiaotong import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import itertools
import gym
import math, os, time
import numpy as np
import numpy.ma as ma
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
MAX_EP = 30000000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v0')
N_S = env.max_atoms_count * 3
N_A = env.action_space.shape[0]
HIGH_A = 1
LOW_A = 0

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0)
        # 4, 25, 25, 25
        # self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        # self.fltt = 2*5*5*5

        # test_2
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=2)
        # # 4, 50, 50, 50
        # self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        # # 2, 10, 10, 10
        # self.conv3d3 = nn.Conv3d(2, 2, 2, stride=2, padding=0)

        # test_3
        self.conv3d1 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=7, stride=2, padding=3)
        # 4, 25, 25, 25
        self.conv3d2 = nn.Conv3d(8, 4, 7, stride=2, padding=3)
        # 2, 13, 13, 13
        self.conv3d3 = nn.Conv3d(4, 2, 3, stride=3, padding=2)

        self.fltt = 2*5*5*5

        self.mu1 = nn.Linear(self.fltt, 128)
        self.mu2 = nn.Linear(256, 128)
        self.mu3 = nn.Linear(128, 64)
        self.mu4 = nn.Linear(128, 3)

        self.sigma1 = nn.Linear(self.fltt, 128)
        self.sigma2 = nn.Linear(256, 128)
        self.sigma3 = nn.Linear(128, 64)
        self.sigma4 = nn.Linear(128, 3)

        self.v1 = nn.Linear(self.fltt, 128)
        self.v2 = nn.Linear(256, 64)
        self.v3 = nn.Linear(64, 32)
        self.v4 = nn.Linear(128, 1)
        set_init([self.conv3d1, self.conv3d2, self.conv3d3,
            self.mu1, self.mu2, self.mu3, self.mu4,
            self.sigma1, self.sigma2, self.sigma3, self.sigma4,
            self.v1, self.v2, self.v3, self.v4])
        
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x = F.relu(self.conv3d1(x))
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        x = F.relu(self.conv3d2(x))
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        x = F.relu(self.conv3d3(x))
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        x = torch.flatten(x, start_dim=1)
        mu = F.relu(self.mu1(x))
        # mu = F.relu(self.mu2(mu))
        # mu = F.relu(self.mu3(mu))
        mu = torch.sigmoid(self.mu4(mu))
        # mu = F.softplus(self.mu4(mu))
        # mu_pre = F.relu(self.mu_pre1(x))
        # mu_pre = torch.sigmoid(self.mu_pre4(mu_pre))
        sigma = F.relu(self.sigma1(x))
        # sigma = F.relu(self.sigma2(sigma))
        # sigma = F.relu(self.sigma3(sigma))
        sigma = F.softplus(self.sigma4(sigma)) + 0.0000001      # avoid 0
        values = F.relu(self.v1(x))
        # x = F.relu(self.v2(x))
        # x = F.relu(self.v3(x))
        values = self.v4(values)

        # mu = torch.cat((mu_pre, mu), 1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(3, ).data, sigma.view(3, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(torch.clamp(m.scale, 1e-10))  # exploration
        print('entropy: ', entropy)
        print('log_prob: ', log_prob)
        print('td: ', td.detach(), 'v_t: ', v_t, 'values: ', values, 'mu: ', mu, 'sigma: ', sigma)
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        print('total_loss: ', total_loss)
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, global_max_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.grid_cnt = 50
        self.seed = name
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.g_ep_max_r = global_max_ep_r
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        # self.lnet.load_state_dict(torch.load('./ep_10000.pth'))
        # self.env = gym.make('Pendulum-v0').unwrapped
        self.env = env 

    def atoms2voxels(self, at):
        # 50*50*50 voxel returned
        sigma_1 = 0.6
        voxel_side_cnt = self.grid_cnt
        side_len = self.env.side_len
        # volume = np.random.rand(voxel_side_cnt, voxel_side_cnt, voxel_side_cnt)
        volume = np.zeros((4, voxel_side_cnt, voxel_side_cnt, voxel_side_cnt), dtype=float)
        for i, j, k in itertools.product(range(voxel_side_cnt),
                range(voxel_side_cnt),
                range(voxel_side_cnt)):
            volume[0][i][j][k] = i/voxel_side_cnt
            volume[1][i][j][k] = j/voxel_side_cnt
            volume[2][i][j][k] = k/voxel_side_cnt
            for idx in range(len(at)):
                x, y, z = i/voxel_side_cnt * side_len, j/voxel_side_cnt * side_len, k/voxel_side_cnt * side_len
                pow_sum = (x-at[idx].position[0])**2 + (y-at[idx].position[1])**2 + (z-at[idx].position[2])**2
                volume[-1][i][j][k] += math.exp(-1*pow_sum/(2*sigma_1**2))
            # np.clip(volume, 0, self.env.max_atoms_count/2)
        return volume


    def run(self):
        total_step = 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()

            s = self.atoms2voxels(s)
            
            buffer_s, buffer_a, buffer_r = [], [], []
            r_history = []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                
                # action conversion
                # print('a: ', a, a.shape)
                # print('s: ', s.shape)
                stt_sz = self.grid_cnt
                
                a = a.clip(LOW_A, HIGH_A)

                s_, r, done, _ = self.env.step(a)
                
                s_ = self.atoms2voxels(s_)
                # s_, r, done, _ = self.env.step(a.clip(LOW_A, HIGH_A))
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                r_history.append(((r, a), a))

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    if len(buffer_a) > 0:
                        buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        g_ep_ret = record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, r_history, self.g_ep_max_r)
                        # if g_ep_ret % 1000 == 0:
                        #     torch.save(self.lnet.state_dict(), 'ep_' + str(g_ep_ret) + '.pth')
                        # for param_group in self.opt.param_groups:
                        #     param_group['lr'] = 1e-5 * (0.5 ** (g_ep_ret//20000))
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    gnet = Net(N_S, N_A)        # global network
    # gnet.load_state_dict(torch.load('./ep_10000.pth'))
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-5, betas=(0.95, 0.999), weight_decay=1e-3)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    global_max_ep_r = mp.Value('d', 0.)
    # parallel training
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count()-2)]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, global_max_ep_r, res_queue, i) for i in range(20)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
