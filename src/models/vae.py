import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    the VAE condenses the  (HWC) 128 x 256 x 3 (RGB) input image into
    a N-dimensional normally distributed random variable, parameterized by two variables,
    mu and logvar. Here, logvar is the logarithm of the variance of the distribution.
    We can sample from this distribution to produce a latent vector z that represents
    the current state. This is passed on to the next part of the network, the MDN-RNN.
    """
    def __init__(self, latent_dim=128, input_channels=3, input_height=128, input_width=256):
        super(VAE, self).__init__()
        
        # Store input dimensions
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        # Encoder layers
        self.encoder = nn.Sequential(
            # Layer 1: 128x256x3 -> 64x128x32
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 2: 64x128x32 -> 32x64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: 32x64x64 -> 16x32x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 4: 16x32x128 -> 8x16x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Flatten for fully connected layers
            nn.Flatten()
        )
        
        # Calculate the size after convolutions
        self.conv_output_size = 16 * 8 * 256
        
        # Latent space layers
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_var = nn.Linear(self.conv_output_size, latent_dim)
        
        # Decoder layers
        self.decoder_input = nn.Linear(latent_dim, self.conv_output_size)
        
        self.decoder = nn.Sequential(
            # Reshape to 16x8x256
            nn.Unflatten(1, (256, 8, 16)),
            
            # Layer 1: 8x16x256 -> 16x32x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 2: 16x32x128 -> 32x64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: 32x64x64 -> 64x128x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 4: 64x128x32 -> 128x256x3
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def encode(self, x):
        """
        Encode input image into latent space parameters.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from the latent space.
        
        Args:
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector back into an image.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            x_recon: Reconstructed image of shape (batch_size, channels, height, width)
        """
        x = self.decoder_input(z)
        x_recon = self.decoder(x)
        return x_recon
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            x_recon: Reconstructed image
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def mse_loss(self, reconstructed_images, original_images, mu, log_var):
        """
        Calculate the VAE loss.
        
        Args:
            reconstructed_images: Reconstructed images
            original_images: Original images
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            
        Returns:
            loss: Total VAE loss
        """
        # Reconstruction loss (MSE) - mean over batch and pixels
        recon_loss = F.mse_loss(reconstructed_images, original_images, reduction='sum')
        
        # KL divergence - mean over batch
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss with weighted KL term
        total_loss = recon_loss + kl_loss
        
        return total_loss, recon_loss, kl_loss 
    
    

    def binary_cross_entropy_loss(self, reconstructed_images, original_images, mu, log_var):
        """
        Compute the loss for a Variational Autoencoder (VAE) for image prediction.
        
        The loss is a combination of:
        1. Reconstruction loss (BCE or MSE loss)
        2. KL divergence loss for the latent space
        
        Args:
            reconstructed_image (Tensor): Reconstructed image from the decoder, expected to be in the range [0, 1].
            original_image (Tensor): Original image, also in the range [0, 1].
            mu (Tensor): The mean of the latent Gaussian distribution.
            log_var (Tensor): The log variance of the latent Gaussian distribution.

        Returns:
            loss (Tensor): The total loss (scalar tensor).
        """
        # Reconstruction loss: using binary cross-entropy loss
        # Set reduction='sum' if you want to sum over all pixels; you could also use 'mean'
        bce_loss = F.binary_cross_entropy(reconstructed_images, original_images, reduction='mean')
                
        
        # KL divergence loss: using the analytical form for Gaussian distributions
        # KL divergence for each element in the batch (summed over the latent dimensions)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss is a sum of both components
        total_loss = bce_loss + kl_loss
        
        return total_loss