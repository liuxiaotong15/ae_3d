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
import gym
import math, os
os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 30000000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0] # useless
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
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=7, stride=2, padding=3)
        # # 4, 25, 25, 25
        # self.conv3d2 = nn.Conv3d(4, 2, 7, stride=2, padding=3)
        # # 2, 13, 13, 13
        # self.conv3d3 = nn.Conv3d(2, 2, 3, stride=3, padding=2)

        # self.fltt = 2*5*5*5
        # test_4 all same input size conv3d
        self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=9, stride=1, padding=4)
        # 4, 25, 25, 25
        self.conv3d2 = nn.Conv3d(4, 2, 7, stride=1, padding=3)
        # 2, 13, 13, 13
        self.conv3d_3mu = nn.Conv3d(2, 1, 3, stride=1, padding=1)
        self.conv3d_3sigma = nn.Conv3d(2, 1, 3, stride=1, padding=1)
        self.conv3d_3v = nn.Conv3d(2, 1, 3, stride=1, padding=1)
        self.fltt = 1*50*50*50
        # self.mu1 = nn.Linear(self.fltt, 100)
        # self.mu2 = nn.Linear(100, a_dim)
        # self.sigma1 = nn.Linear(self.fltt, 100)
        # self.sigma2 = nn.Linear(100, a_dim)
        # self.c1 = nn.Linear(s_dim, 100)
        self.v1 = nn.Linear(self.fltt, 100)
        self.v2 = nn.Linear(100, 1)
        # set_init([self.conv3d1, self.conv3d2, self.conv3d3, self.mu1, self.mu2, self.sigma1, self.sigma2, self.v1, self.v2])
        set_init([self.conv3d1, self.conv3d2, self.conv3d_3mu, self.conv3d_3sigma, self.conv3d_3v, self.v1, self.v2])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x = F.relu(self.conv3d1(x))
        # x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        x = F.relu(self.conv3d2(x))
        # x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        mu = HIGH_A * torch.sigmoid(self.conv3d_3mu(x))
        mu = mu.view(-1).softmax(0).view(*mu.shape)
        sigma = F.softplus(self.conv3d_3sigma(x))
        # x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        # x = torch.flatten(x, start_dim=1)
        # mu = F.relu(self.mu1(x))
        # mu = HIGH_A * torch.sigmoid(self.mu2(mu))
        # sigma = F.relu(self.sigma1(x))
        # sigma = F.softplus(self.sigma2(sigma)) + 0.001      # avoid 0
        x = torch.flatten(self.conv3d_3v(x), start_dim=1)
        x = F.relu(self.v1(x))
        values = self.v2(x)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu, sigma)
        # m = self.distribution(mu.view(3, ).data, sigma.view(3, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        # https://blog.csdn.net/weixin_43116379/article/details/90674462
        # in xiaotong's understanding: under specific td, try to max the log_prob of a by changing the mu and sigma
        for i in range(td.detach().shape[0]):
            log_prob[i] *= td.detach()[i]
        # exp_v = log_prob * td.detach() + 0.005 * entropy
        exp_v = log_prob + 0.005 * entropy
        a_loss = -exp_v
        total_loss = 0
        for i in range(c_loss.shape[0]):
            total_loss += (a_loss[i] + c_loss[i]).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        # self.env = gym.make('Pendulum-v0').unwrapped
        self.env = env 

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                # if self.name == 'w0':
                #     self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a.clip(LOW_A, HIGH_A))
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                # buffer_r.append((r+8.1)/8.1)    # normalize
                buffer_r.append(r)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('forkserver')
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count()-2)]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(16)]
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
