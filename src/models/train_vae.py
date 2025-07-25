import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from src.models.device import get_device_name
import sys


from vae import VAE

class ImageDataset(Dataset):
    def __init__(self, data_dir):
        """
        Dataset for loading images from the data directory.
        
        Args:
            data_dir: Path to the directory containing .npy files
        """
        self.data_dir = Path(data_dir)
        self.image_files = list(self.data_dir.glob('**/*.npy'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
            Reshape from (H, W, C) to (C, H, W) format
            Dimension at index 2 in the original tensor (the channels) becomes the first dimension.
            Dimension at index 0 (height) becomes the second.
            Dimension at index 1 (width) becomes the third.
        """
        # Image will be (128, 256, 3) but color channel is BGR
        image = np.load(self.image_files[idx], allow_pickle=True)
        image = image[:, :, ::-1].copy() # Convert BGR to RGB
        
        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).float() / 255.0

        # Reshape from (H, W, C) to (C, H, W) format
        image = image.permute(2, 0, 1)
        
        return image

def train_vae(data_dir, output_dir, num_epochs=50, batch_size=32, learning_rate=1e-5, weight_decay=1e-5, form_checkpoint=None):
    """
    Train the VAE model on the collected images.
    
    Args:
        data_dir: Directory containing the collected images
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = get_device_name()
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = ImageDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    if form_checkpoint:
        model = VAE(latent_dim=128).to(device)
        model.load_state_dict(torch.load(form_checkpoint))
        print(f"Loaded model from {form_checkpoint}")
    else:
        model = VAE(latent_dim=128).to(device)
        print("Initialized new model")
    
    # Initialize optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in pbar:
            # Move batch to device
            images = batch.to(device)
            
            # Forward pass
            reconstructed_images, mu, log_var = model(images)

            # Calculate loss
            loss, _, _  = model.mse_loss(reconstructed_images, images, mu, log_var)
            
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
            checkpoint_path = output_dir / f'vae_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    
    # Save final model
    final_model_path = output_dir / 'vae_final.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model to {final_model_path}')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VAE on collected images')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing collected images')
    parser.add_argument('--output_dir', type=str, default='models/vae',
                        help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument("--form-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    
    train_vae(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        form_checkpoint=args.form_checkpoint,
    ) 