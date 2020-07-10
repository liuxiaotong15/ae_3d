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
from scipy.optimize import minimize
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
MAX_EP = 30000000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v0')
N_S = env.max_atoms_count * 3
N_A = env.max_atoms_count - 1
HIGH_A = env.side_len
LOW_A = 0

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.mu1 = nn.Linear(self.s_dim, 128)
        self.mu2 = nn.Linear(128, 64)
        self.mu3 = nn.Linear(64, 32)
        self.mu4 = nn.Linear(32, self.a_dim)

        self.sigma1 = nn.Linear(self.s_dim, 128)
        self.sigma2 = nn.Linear(128, 64)
        self.sigma3 = nn.Linear(64, 32)
        self.sigma4 = nn.Linear(32, self.a_dim)

        self.v1 = nn.Linear(self.s_dim, 128)
        self.v2 = nn.Linear(128, 64)
        self.v3 = nn.Linear(64, 32)
        self.v4 = nn.Linear(32, 1)
        
        set_init([
            self.mu1, self.mu2, self.mu3, self.mu4,
            self.sigma1, self.sigma2, self.sigma3, self.sigma4,
            self.v1, self.v2, self.v3, self.v4])
        
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        mu = F.relu(self.mu1(x))
        mu = F.relu(self.mu2(mu))
        mu = F.relu(self.mu3(mu))
        mu = torch.sigmoid(self.mu4(mu)) * HIGH_A
        
        sigma = F.relu(self.sigma1(x))
        sigma = F.relu(self.sigma2(sigma))
        sigma = F.relu(self.sigma3(sigma))
        sigma = F.softplus(self.sigma4(sigma)) + 0.0000001      # avoid 0
        
        x = F.relu(self.v1(x))
        x = F.relu(self.v2(x))
        x = F.relu(self.v3(x))
        values = self.v4(x)

        # mu = torch.cat((mu_pre, mu), 1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(self.a_dim, ).data, sigma.view(self.a_dim, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(torch.clamp(m.scale, 1e-10))  # exploration
        # print('entropy: ', entropy)
        # print('log_prob: ', log_prob)
        # print('td: ', td.detach(), 'v_t: ', v_t, 'values: ', values, 'mu: ', mu, 'sigma: ', sigma)
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        # print('total_loss: ', total_loss)
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, global_max_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.seed = name
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.g_ep_max_r = global_max_ep_r
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        # self.lnet.load_state_dict(torch.load('./ep_10000.pth'))
        # self.env = gym.make('Pendulum-v0').unwrapped
        self.env = env 

    def atoms2xyz(self, atoms):
        xyz_lst = []
        for at in atoms:
            xyz_lst.extend(list(at.position))
        while len(xyz_lst) < self.env.max_atoms_count * 3:
            xyz_lst.append(-1 * self.env.side_len)
        return np.array([xyz_lst])/self.env.side_len + 1


    def loss(self, x, atoms, distance):
        ret = 0
        for i in range(len(atoms)):
            at = atoms[i]
            ret += (math.sqrt((at.position[0] - x[0] * HIGH_A)**2 +
                    (at.position[1] - x[1] * HIGH_A)**2 +
                    (at.position[2] - x[2] * HIGH_A)**2) - distance[i])**2
        # print(len(atoms), x * HIGH_A, distance, ret)
        return ret

    def run(self):
        total_step = 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        while self.g_ep.value < MAX_EP:
            s_atoms = self.env.reset()
            s_xyz = self.atoms2xyz(s_atoms)
            
            buffer_s, buffer_a, buffer_r = [], [], []
            r_history = []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                a = self.lnet.choose_action(v_wrap(s_xyz[None, :]))
                a = a.clip(LOW_A, HIGH_A)
                # convert distance to all atoms to xyz, 
                # maybe random choice in the begining is better, equal is bad
                xyz = np.array([0.1, 0.2, 0.3])
                bnds = ((0, 1), (0, 1), (0, 1))
                res = minimize(self.loss, xyz, args=(
                                        s_atoms, a), method='L-BFGS-B', bounds=bnds) # method='BFGS')
                if(res.success):
                    xyz = res.x
                else:
                    print('opt failure', res.message, res.x)
                    print(len(s_atoms), a, s_atoms.positions)
                    xyz = res.x

                s_, r, done, _ = self.env.step(xyz)
                # print(len(s_), r, xyz)
                
                s_atoms = s_
                s_ = self.atoms2xyz(s_)
                s_xyz = s_

                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                
                buffer_a.append(a)
                buffer_s.append(s_xyz)
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
    workers = [Worker(gnet, opt, global_ep, global_ep_r, global_max_ep_r, res_queue, i) for i in range(1)]
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
