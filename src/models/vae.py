from torch import nn
from torch.nn import functional as F
import torch


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    Args:
        state_dim (int): Dimension of the input state space
        latent_dim (int): Dimension of the latent space
    """

    def __init__(self, state_dim, latent_dim=32):
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)      # Mean
        self.fc_var = nn.Linear(256, latent_dim)     # Variance
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)        # Mean of the latent distribution
        log_var = self.fc_var(x)  # Log variance of the latent distribution
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # Random noise
        return mu + eps * std           # Sample from the distribution
    
    def decode(self, z):
        x = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(x))
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def sample(self, z):
        x_reconstructed = self.decode(z)
        
    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.mse_loss(recon_x, x, reduction='sum')  # Reconstruction loss
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL divergence
        return BCE + KLD
        