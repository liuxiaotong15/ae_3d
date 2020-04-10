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
        self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0)
        # 4, 25, 25, 25
        self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        # self.linear1 = nn.Linear(self.obs_dim, 1024)
        # self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        # self.linear3 = nn.Linear(512, 300)
        self.linear = nn.Linear(5 * 5 * 5 * 2, 1)

    def forward(self, x, a):
        # TODO: x = x + nn.MaxPool3d(a......) to set all small value of a to zero
        x = F.relu(self.conv3d1(x+a))
        x = F.relu(self.conv3d2(x))
        x = torch.flatten(x, start_dim=1)
        qval = self.linear(x)
      
        return qval

class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 1, 50, 50, 50
        self.conv3d1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0)
        # 4, 25, 25, 25
        self.conv3d2 = nn.Conv3d(4, 2, 5, stride=5, padding=0)
        self.conv3dT1 = nn.ConvTranspose3d(2, 4, 5, stride=5)
        # 4, 25, 25, 25
        self.conv3dT2 = nn.ConvTranspose3d(4, 1, 2, stride=2, padding=0)
        # 1, 50, 50, 50
       
    def forward(self, obs):
        # print(obs.shape)
        x = F.relu(self.conv3d1(obs))
        x = self.conv3d2(x)
        x = F.relu(self.conv3dT1(x))
        x = self.conv3dT2(x)
        # softmax of all elements
        x = x.view(-1).softmax(0).view(*x.shape)
        
        return x
