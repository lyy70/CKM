import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_CNN(nn.Module):
    def __init__(self, latent_dim, input_dim, channels):
        super(VAE_CNN, self).__init__()

        self.input_dim = input_dim
        self.channels = channels
        self.latent_dim = latent_dim

        # Encoder with three convolutional layers
        self.conv1 = nn.Conv2d(channels, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)

        self.fc_mu = nn.Linear(32 * 1 * 1, latent_dim)
        self.fc_log_var = nn.Linear(32 * 1 * 1, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 32 * 1 * 1)
        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):

        x = F.relu(self.fc_decode(z))
        x = x.view(x.size(0), 32, 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return recons, mu, log_var
