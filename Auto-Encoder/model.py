import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as ptv


__all__ = ["DataDecayWrapper", "VAE"]


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias != None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class DataDecayWrapper(nn.Module):
    def __init__(self, model, batch_size, lamda, shape):
        super(DataDecayWrapper, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.lamda = lamda
        self.shape = shape
        if lamda != 0.0:
            self.p = torch.zeros(shape)
            self.p = torch.split(self.p, batch_size)
            self.p = [nn.Parameter(x, requires_grad=True) for x in self.p]
            self.p = nn.ParameterList(self.p)
            self.count = 0

    def forward(self, x):
        x = self.model(x)
        if self.lamda != 0.0:
            if x.size(0) == self.batch_size:
                x = x + self.lamda * self.p[self.count]
            else:
                x = x + self.lamda * ptv(self.p).view(self.shape)
        return x

    def step(self):
        if self.lamda != 0.0:
            self.count += 1
            self.count %= len(self.p)


class VAE(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, 2, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            #
            nn.Conv2d(16, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            #
            nn.Conv2d(32, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            #
            nn.Conv2d(32, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2),
            #
            nn.Conv2d(32, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, img_size // 64, img_size // 64)),
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            #
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            #
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            #
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            #
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            #
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.latent_dim = 32 * ((img_size // 64) ** 2)
        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, self.latent_dim)
        initialize_weights(self)

    def get_latent(self, x: torch.Tensor):
        mu = self.fc1(x)
        logvar = self.fc2(x)
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        x = self.get_latent(x)
        x = self.decoder(x)
        return x
