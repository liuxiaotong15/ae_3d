import torch
import h5py
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import numpy as np

training_type = 0 # 0: ae_training; 1: prediction_training
model_dump_name = './conv_autoencoder.pth'
side_length = 50 # * 0.1A

num_epochs = 100
batch_size = 128
learning_rate = 1e-3
 
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.5,), (0.5,))
])
 
# dataset = MNIST('./data', transform=img_transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
 
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 1, 28, 28, 28
            # 1, 50, 50, 50 
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            # 4, 25, 25, 25 
            # nn.MaxPool2d(2, stride=2),
            nn.Conv3d(4, 2, 5, stride=5, padding=0),
            nn.ReLU(True),
            # 2, 5, 5, 5
            # nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(2, 4, 5, stride=5),
            nn.ReLU(True),
            # 4, 14, 14, 14
            # 4, 25, 25, 25
            nn.ConvTranspose3d(4, 1, 2, stride=2, padding=0),
            # 1, 50, 50, 50 
            # nn.ReLU(True),
            # nn.Tanh()
        )
        self.prediction = nn.Sequential(
            nn.Linear(250, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )
 
    def forward(self, x):
        # print('input shape: ', x.shape)
        x = self.encoder(x)
        print('latent shape: ', x.shape)
        x = self.decoder(x)
        # print('output shape: ', x.shape)
        return x
 
 
model = autoencoder()#.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
                             # weight_decay=1e-5)
size = side_length
# img = torch.randn(128, 1, 28, 28, 28)
idx = 0
f = h5py.File('dataset_' + str(idx+1) + '.hdf5', 'r')
dataset = f['dset1'][:]
img = []
prpty = []
for i in range(100):
    v = np.array(dataset[i][:-1])
    v = v.reshape(1,size,size,size)
    img.append(v)
    prpty.append(dataset[i][-1])
    # print(v)
    # print(dataset[-1])
f.close()
# print(prpty)
img = torch.from_numpy(np.array(img)).to('cpu').float()
prpty = torch.from_numpy(np.array(prpty)).to('cpu').float()

if training_type == 1:
    # prpty = torch.randn(128,1)
    model.load_state_dict(torch.load(model_dump_name))
    model.eval()

for epoch in range(num_epochs):
    if training_type == 0:    
    # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
    # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    elif training_type == 1:
        latent_output = model.encoder(img)
        latent_output = torch.flatten(latent_output, start_dim=1)
        print('latent output shape in main: ', latent_output.shape)
        predict_property = model.prediction(latent_output)
        print('property output shape in main: ', predict_property.shape)
        loss = criterion(predict_property, prpty)
    # ===================backward====================
        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    else:
        pass

if training_type == 0: 
    torch.save(model.state_dict(), model_dump_name)
