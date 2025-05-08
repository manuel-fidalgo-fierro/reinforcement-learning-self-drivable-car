import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from models.device import get_device_name
import sys
import torch.nn.functional as F
import math


from mdn_rnn import MDNRNN

class MDNRNDataset(Dataset):
    def __init__(self, data_dir):
        """
        Dataset for loading MDN-RNN training data.
        
        Args:
            data_dir: Path to the directory containing .npy files with:
                     - latent vector (128)
                     - reward (1)
                     - action (3)
        """
        self.data_dir = Path(data_dir)
        self.data_files = list(self.data_dir.glob('**/*.npy'))
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single data point.
        
        Returns:
            data: Tensor containing concatenated [z, reward, action_forward, action_left_right]
                 Shape: [131] where:
                 - z: 128 dimensions
                 - reward: 1 dimension
                 - action_forward: 1 dimension
                 - action_left_right: 1 dimension
        """
        # Load data
        data = np.load(self.data_files[idx])
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        return data

def mdn_loss(mdn_params, batch, num_mixtures=5, z_dim=128, eps=1e-8):
    """
    MDN loss (NLL) for a per-dimension mixture of Gaussians.
    Thanks ChatGPT for the implementation.

    Args:
        mdn_params: Tensor, shape (B, M*3*D) where:
                    - B: batch size
                    - M: number of mixtures
                    - D: dimensionality of latent z
        batch:      Tensor, shape (B, D+1+action_dims)
                    we'll take the first D entries as the target z.
        num_mixtures: int, number of mixture components (K).
        z_dim:        int, dimensionality of latent z (D).
        eps:        float, numerical stabilizer.

    Returns:
        loss: scalar Tensor, average NLL over batch.
    """
    B = mdn_params.size(0)
    K, D = num_mixtures, z_dim

    # Pull out target z, shape (B, D)
    z_target = batch[:, :D]

    # split mdn_params into (pi, mu, log_sigma) each of length K*D along the last axis
    param_per_mixture = D * K
    pi_flat     = mdn_params[:, :           param_per_mixture]
    mu_flat     = mdn_params[:, param_per_mixture:2*param_per_mixture]
    log_sigma_flat = mdn_params[:, 2*param_per_mixture:3*param_per_mixture]

    # reshape to (B, D, K)
    pi_logits    = pi_flat.view(B, D, K)
    mu           = mu_flat.view(B, D, K)
    log_sigma    = log_sigma_flat.view(B, D, K)

    # Build actual pi (mixture weights) and sigma
    pi    = F.softmax(pi_logits, dim=-1)       # (B, D, K)
    sigma = torch.exp(log_sigma).clamp_min(eps)  # (B, D, K)

    # Compute log-prob of z_target under each Gaussian component expand target from (B,D) → (B,D,1)
    z_exp = z_target.unsqueeze(-1)

    # exponent:   -0.5 * ((z - μ)/σ)^2
    exponent = -0.5 * ((z_exp - mu) / sigma) ** 2

    # norm term: -log(σ) - 0.5*log(2π)
    log_norm = -torch.log(sigma) - 0.5 * math.log(2 * math.pi)

    # component log-lik across dims: sum over the D dims but we'll sum dims later—first keep shape (B,D,K)
    comp_log = exponent + log_norm

    # add mixture‐weight log π
    log_pi = torch.log(pi + eps)
    comp_log = comp_log + log_pi

    # log-sum-exp over K mixtures → (B, D)
    log_prob_per_dim = torch.logsumexp(comp_log, dim=-1)

    # sum across latent dimensions → (B,)
    log_prob = log_prob_per_dim.sum(dim=-1)

    # negative log-likelihood and average
    nll = -log_prob
    return nll.mean()


def train_mdn_rnn(data_dir, output_dir, num_epochs=50, batch_size=2048, learning_rate=1e-4, 
                 weight_decay=1e-5, from_checkpoint=None, input_size=131, hidden_state_size=256, num_mixtures=5, z_dim=128):
    """
    Train the MDN-RNN model on the collected data.
    
    Args:
        data_dir: Directory containing the collected data
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        from_checkpoint: Path to checkpoint to resume training from
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = get_device_name()
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = MDNRNDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = MDNRNN(input_size=input_size, hidden_state_size=hidden_state_size, num_mixtures=num_mixtures, z_dim=z_dim).to(device)
    if from_checkpoint:
        model.load_state_dict(torch.load(from_checkpoint))
        print(f"Loaded model from {from_checkpoint}")
    else:
        print("Initialized new model")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            
            # Move batch to device
            batch = batch.to(device)
            z = batch[:, :128]
            
            
            # Forward pass
            mdn_params, (_,_) = model(batch)
            #print(f"Batch shape: {batch.shape}")  # ([2048, 131])
            #print(f"mdn_params  shape: {mdn_params.shape}") # ([2048, 1921])
            #print(f"z shape: {z.shape}") # ([2048, 128])
            
            # Calculate loss
            loss = mdn_loss(mdn_params, batch) 
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / len(pbar)})
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'mdn_rnn_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    
    # Save final model
    final_model_path = output_dir / 'mdn_rnn_final.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model to {final_model_path}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MDN-RNN on collected data')
    parser.add_argument('--data_dir', type=str, default='data_mdn_rnn',
                        help='Directory containing collected data for mdn_rnn model')
    parser.add_argument('--output_dir', type=str, default='models/mdn_rnn',
                        help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--input_size', type=int, default=131,
                        help='Input size for the model')
    parser.add_argument('--hidden_state_size', type=int, default=256,
                        help='Hidden state size for the model')
    parser.add_argument('--num_mixtures', type=int, default=5,
                        help='Number of mixtures for the model')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='Dimension of the latent z for the model')
    args = parser.parse_args()
    
    train_mdn_rnn(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        from_checkpoint=args.from_checkpoint,
        input_size=args.input_size,
        hidden_state_size=args.hidden_state_size,
        num_mixtures=args.num_mixtures,
        z_dim=args.z_dim
    ) 