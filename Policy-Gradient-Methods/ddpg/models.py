import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

seed = 1234
torch.manual_seed(seed)
debug = False
def print_msg(x):
    if debug:
        print(x)
    else:
        pass

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 1, 50, 50, 50
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0)
        # 4, 25, 25, 25
        # self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        # test_2
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=2)
        # # 4, 50, 50, 50
        # self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        # # 2, 10, 10, 10
        # self.conv3d3 = nn.Conv3d(2, 2, 2, stride=2, padding=0)
        # test_3
        self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=7, stride=2, padding=3)
        # 4, 25, 25, 25
        self.conv3d2 = nn.Conv3d(4, 2, 7, stride=2, padding=3)
        # 2, 13, 13, 13
        self.conv3d3 = nn.Conv3d(2, 2, 3, stride=3, padding=2)

        self.linear = nn.Linear(5 * 5 * 5 * 2, 1)

    def forward(self, x, a):
        # TODO: x = x + nn.MaxPool3d(a......) to set all small value of a to zero
        # a is 32, 1, 50, 50, 50
        a = a.detach().numpy()
        stt_sz = a.shape[2]
        for bi in range(a.shape[0]):
            if np.amax(a[bi]) > 1e-5:
                a[bi] = a[bi]/np.amax(a[bi])
            # mask = np.zeros(a.shape, dtype=np.float32)
            result = np.where(a[bi] == np.amax(a[bi]))
            # find xyz in whole action
            x1, y1, z1 = 0, 0, 0
            # if have multi max, choose the max sum around it
            max_sub_arr_sum = -100
            for idx in range(0, len(result[1])):
                i, j, k = result[1][idx], result[2][idx], result[3][idx]
                sub_arr = a[bi, 0, max(0, i-1):min(i+2, stt_sz), max(0, j-1):min(j+2, stt_sz), max(0, k-1):min(k+2, stt_sz)]
                if np.sum(sub_arr) < 1e-5 or sub_arr.size < 1e-5:
                    continue
                if np.sum(sub_arr)/sub_arr.size > max_sub_arr_sum:
                    max_sub_arr_sum = np.sum(sub_arr)/sub_arr.size
                    x1, y1, z1 = i, j, k
            for i in range(max(0, x1-1),min(x1+2, stt_sz)):
                for j in range(max(0, y1-1),min(y1+2, stt_sz)):
                    for k in range(max(0, z1-1),min(z1+2, stt_sz)):
                        x[bi][0][i][j][k] += a[bi][0][i][j][k]
                    # mask[0][i][j][k] = a[0][i][j][k]

        # a = torch.from_numpy(mask)
        # a.requires_grad_(True)
        # x = F.relu(self.conv3d1(x+a))
        x = F.relu(self.conv3d1(x))
        x = F.relu(self.conv3d2(x))
        x = F.relu(self.conv3d3(x))
        x = torch.flatten(x, start_dim=1)
        qval = self.linear(x)
      
        return qval

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # test_1
        # 1, 50, 50, 50
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0)
        # 4, 25, 25, 25
        # self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        # self.conv3dT1 = nn.ConvTranspose3d(2, 4, 5, stride=5)
        # 4, 25, 25, 25
        # self.conv3dT2 = nn.ConvTranspose3d(4, 1, 2, stride=2, padding=0)
        # 1, 50, 50, 50
        # test_2
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=2)
        # # 4, 50, 50, 50
        # self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        # # 2, 10, 10, 10
        # self.conv3d3 = nn.Conv3d(2, 2, 2, stride=2, padding=0)
        # # 2, 5, 5, 5
        # self.conv3dT1 = nn.ConvTranspose3d(2, 2, 2, stride=2)
        # # 2, 10, 10, 10
        # self.conv3dT2 = nn.ConvTranspose3d(2, 1, 5, stride=5)
        # 4, 50, 50, 50 
        # self.conv3dT3 = nn.ConvTranspose3d(4, 1, 3, stride=1, padding=2)
        # 1, 50, 50, 50

        # # test_3
        # self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=7, stride=2, padding=3)
        # # 4, 25, 25, 25
        # self.conv3d2 = nn.Conv3d(4, 2, 7, stride=2, padding=3)
        # # 2, 13, 13, 13
        # self.conv3d3 = nn.Conv3d(2, 2, 3, stride=3, padding=2)
        # # 2, 5, 5, 5
        # self.conv3dT1 = nn.ConvTranspose3d(2, 2, 2, stride=2)
        # # 2, 10, 10, 10
        # self.conv3dT2 = nn.ConvTranspose3d(2, 1, 5, stride=5)
        # # 1, 50, 50, 50
        # test_4
        self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=9, stride=1, padding=4)
        # 4, 25, 25, 25
        self.conv3d2 = nn.Conv3d(4, 2, 7, stride=1, padding=3)
        # 2, 13, 13, 13
        self.conv3d3 = nn.Conv3d(2, 1, 3, stride=1, padding=1)

    def forward(self, obs):
        print_msg(obs.shape)
        x = F.relu(self.conv3d1(obs))
        print_msg(x.shape)
        # x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        # print_msg(x.shape)
        x = F.relu(self.conv3d2(x))
        print_msg(x.shape)
        # x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        # print_msg(x.shape)
        x = F.relu(self.conv3d3(x))
        print_msg(x.shape)
        # x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        # print_msg(x.shape)
        # x = F.relu(self.conv3dT1(x))
        # print_msg(x.shape)
        # x = self.conv3dT2(x)
        # print_msg(x.shape)
        # x = self.conv3dT3(x)
        # softmax of all elements
        x = x.view(-1).softmax(0).view(*x.shape)
        
        return x
