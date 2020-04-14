import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

seed = 1234
torch.manual_seed(seed)

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
        x = F.relu(self.conv3d1(x+a))
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

        # test_3
        self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=7, stride=2, padding=3)
        # 4, 25, 25, 25
        self.conv3d2 = nn.Conv3d(4, 2, 7, stride=2, padding=3)
        # 2, 13, 13, 13
        self.conv3d3 = nn.Conv3d(2, 2, 3, stride=3, padding=2)
        # 2, 5, 5, 5
        self.conv3dT1 = nn.ConvTranspose3d(2, 2, 2, stride=2)
        # 2, 10, 10, 10
        self.conv3dT2 = nn.ConvTranspose3d(2, 1, 5, stride=5)
        # 1, 50, 50, 50

    def forward(self, obs):
        # print(obs.shape)
        x = F.relu(self.conv3d1(obs))
        # print(x.shape)
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        # print(x.shape)
        x = F.relu(self.conv3d2(x))
        # print(x.shape)
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        # print(x.shape)
        x = F.relu(self.conv3d3(x))
        # print(x.shape)
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        # print(x.shape)
        x = F.relu(self.conv3dT1(x))
        # print(x.shape)
        x = self.conv3dT2(x)
        # print(x.shape)
        # x = self.conv3dT3(x)
        # softmax of all elements
        x = x.view(-1).softmax(0).view(*x.shape)
        
        return x
