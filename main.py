import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

training_type = 1 # 0: ae_training; 1: prediction_training
 
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
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0),  # b, 16, 10, 10
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(4, 2, 2, stride=2, padding=0),  # b, 8, 3, 3
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(2, 4, 2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose3d(4, 1, 2, stride=2, padding=0),  # b, 8, 15, 15
            # nn.ReLU(True),
            # nn.ConvTranspose3d(1, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        self.prediction = nn.Sequential(
            nn.Linear(686, 256),
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
        print('input shape: ', x.shape)
        x = self.encoder(x)
        print('latent shape: ', x.shape)
        x = self.decoder(x)
        print('output shape: ', x.shape)
        return x
 
 
model = autoencoder()#.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

img = torch.randn(128, 1, 28, 28, 28)
prpty = torch.randn(128,1)

if training_type == 1: 
    model.load_state_dict(torch.load('./conv_autoencoder.pth'))
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
    torch.save(model.state_dict(), './conv_autoencoder.pth')
