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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', required = True,
        choices=['ae', 'pred'],
        help = '\'ae\' autoencoder, \
        \'pred\' prediction.')
args = parser.parse_args()      # parse_args()从指定的选项中返回一些数据

if args.mode == 'ae':
    training_type = 0 # 0: ae_training; 1: prediction_training
    model_dump_name = './conv_ae.pth'
elif args.mode == 'pred':
    training_type = 1 # 0: ae_training; 1: prediction_training
    model_dump_name = './conv_pred.pth'
else:
    training_type = -1
    model_dump_name = './conv_default.pth'
side_length = 50 # * 0.1A

patience = 10
num_epochs = 10000
batch_size = 64 # TODO: enable it! 
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
        # print('latent shape: ', x.shape)
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
print(type(dataset[:][:-1]))
print('data count: ', len(dataset))
print('max/min/avg img data: ', np.max(dataset[:, :-1]), np.min(dataset[:, :-1]), np.average(dataset[:, :-1]))
print('max/min/avg raw property data: ', np.max(dataset[:, -1]), np.min(dataset[:, -1]), np.average(dataset[:, -1]))

# TODO:divide dataset into tr/te/va

total_cnt = len(dataset)
max_p = np.max(dataset[:, -1])
for i in range(total_cnt):
    v = np.array(dataset[i][:-1])
    v = v.reshape(1,size,size,size)
    img.append(v)
    p = dataset[i][-1]
    if p > 0:
        p = p / max_p
    prpty.append(p)
    # print(v)
    # print(dataset[-1])
f.close()

print('max/min/avg modified property data: ', max(prpty), min(prpty), np.average(np.array(prpty)))
# print(prpty)
img = torch.from_numpy(np.array(img)).to('cpu').float()
prpty = torch.from_numpy(np.array(prpty)).to('cpu').float()

if training_type == 1:
    # prpty = torch.randn(128,1)
    model.load_state_dict(torch.load(model_dump_name))
    # model.eval()

patience_tmp = 0
loss_min = 10000
for epoch in range(num_epochs):
    if epoch > 500:
        lr = 1e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if training_type == 0:    
    # ===================forward=====================
        model.train()
        output = model(img[:int(0.8 * total_cnt)])
        loss = criterion(output[:int(0.8 * total_cnt)], img[:int(0.8 * total_cnt)])
    # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # ===================vali=======================
        model.eval()
        output = model(img[int(0.8*total_cnt):int(0.9*total_cnt)])
        loss = criterion(output, img[int(0.8*total_cnt):int(0.9*total_cnt)])
    # ===================log========================
        print('epoch [{}/{}], vali loss:{:.4f}, pat:{}'.format(epoch+1, num_epochs, loss.item(), patience_tmp))
        if loss.item()<loss_min:
            patience_tmp = 0
            loss_min = loss.item()
            torch.save(model.state_dict(), model_dump_name + str(epoch))
        else:
            patience_tmp += 1
    # ===================test=======================
        model.eval()
        output = model(img[int(0.9*total_cnt):])
        loss = criterion(output, img[int(0.9*total_cnt):])
    # ===================log========================
        print('epoch [{}/{}], test loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
        print('-' * 100)
    elif training_type == 1:
    # ===================forward=====================
        model.train()
        latent_output = model.encoder(img[:int(0.8*total_cnt)])
        latent_output = torch.flatten(latent_output, start_dim=1)
        # print('latent output shape in main: ', latent_output.shape)
        predict_property = model.prediction(latent_output)
        # print('property output shape in main: ', predict_property.shape)
        loss = criterion(predict_property, prpty[:int(0.8*total_cnt)])
    # ===================backward====================
        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # ===================vali=======================
        model.eval()
        latent_output = model.encoder(img[int(0.8*total_cnt):int(0.9*total_cnt)])
        latent_output = torch.flatten(latent_output, start_dim=1)
        predict_property = model.prediction(latent_output)
        loss = criterion(predict_property, prpty[int(0.8*total_cnt):int(0.9*total_cnt)])
    # ===================log========================
        print('epoch [{}/{}], vali loss:{:.4f}, pat:{}'.format(epoch+1, num_epochs, loss.item(), patience_tmp))
        if loss.item()<loss_min:
            patience_tmp = 0
            loss_min = loss.item()
            torch.save(model.state_dict(), model_dump_name + str(epoch))
        else:
            patience_tmp += 1
    # ===================test=======================
        model.eval()
        latent_output = model.encoder(img[int(0.9*total_cnt):])
        latent_output = torch.flatten(latent_output, start_dim=1)
        predict_property = model.prediction(latent_output)
        loss = criterion(predict_property, prpty[int(0.9*total_cnt):])
    # ===================log========================
        print('epoch [{}/{}], test loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
        print('-' * 100)
    else:
        pass
    if patience_tmp >= patience:
            break

# if training_type == 0: 
#     torch.save(model.state_dict(), model_dump_name)
