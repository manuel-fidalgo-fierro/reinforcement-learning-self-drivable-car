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
            data: Tensor containing concatenated [z, reward, action]
        """
        # Load data
        data = np.load(self.data_files[idx])
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        return data

def mdn_loss(mdn_params, target_z, num_mixtures=5):
    """
    Calculate the MDN loss.
    
    Args:
        mdn_params: Output from the MDN layer
        target_z: Target latent vector
        num_mixtures: Number of mixture components
        
    Returns:
        loss: Negative log likelihood loss
    """
    # Get mixture parameters
    pi, mu, sigma = MDNRNN.get_mixture_params(mdn_params)
    
    # Calculate negative log likelihood
    target_z = target_z.unsqueeze(2).expand_as(mu)
    z_dist = torch.distributions.Normal(mu, sigma)
    log_prob = z_dist.log_prob(target_z)
    log_prob = log_prob.sum(dim=-1)  # Sum over dimensions
    log_prob = log_prob + torch.log(pi)
    log_prob = torch.logsumexp(log_prob, dim=-1)  # Sum over mixtures
    loss = -log_prob.mean()
    
    return loss

def train_mdn_rnn(data_dir, output_dir, num_epochs=50, batch_size=32, learning_rate=1e-4, 
                 weight_decay=1e-5, sequence_length=16, from_checkpoint=None):
    """
    Train the MDN-RNN model on the collected data.
    
    Args:
        data_dir: Directory containing the collected data
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        sequence_length: Length of sequences for training
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
    if from_checkpoint:
        model = MDNRNN().to(device)
        model.load_state_dict(torch.load(from_checkpoint))
        print(f"Loaded model from {from_checkpoint}")
    else:
        model = MDNRNN().to(device)
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
            
            # Create sequences
            batch_size = batch.size(0)
            sequences = []
            targets = []
            
            for i in range(batch_size - sequence_length):
                seq = batch[i:i+sequence_length]
                target = batch[i+1:i+sequence_length+1, :128]  # Only predict z
                sequences.append(seq)
                targets.append(target)
            
            if not sequences:
                continue
                
            sequences = torch.stack(sequences)
            targets = torch.stack(targets)
            
            # Forward pass
            mdn_params, _ = model(sequences)
            
            # Calculate loss
            loss = mdn_loss(mdn_params, targets)
            
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
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing collected data')
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
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Length of sequences for training')
    parser.add_argument('--from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    train_mdn_rnn(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sequence_length=args.sequence_length,
        from_checkpoint=args.from_checkpoint
    ) 