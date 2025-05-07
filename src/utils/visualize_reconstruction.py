import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import sys
from pathlib import Path


# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from models.vae import VAE
from models.device import get_device_name as get_device

def load_random_images(data_dir='data', num_images=3):
    """Load multiple random images from the collected data directory."""
    data_dir = Path(data_dir)
    image_files = list(data_dir.rglob('*.npy'))
    if not image_files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    
    # Select random images
    random_image_paths = random.sample(image_files, min(num_images, len(image_files)))
    print(f"Loading {len(random_image_paths)} random images")
    
    images_tensor = []
    images_array = []
    
    for img_path in random_image_paths:
        print(f"Loading image: {img_path}")
        
        # Load and preprocess the image
        # Image will be (128, 256, 3) but color channel is BGR
        img_array = np.load(img_path)
        img_array = img_array[:, :, ::-1].copy() # Convert to rgb
        img_array = np.array(img_array) / 255.0  # Normalize to [0, 1]
        
        # Convert to tensor and reshape to CHW
        img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)
        
        images_tensor.append(img_tensor)
        images_array.append(img_array)
    
    return images_tensor, images_array


def visualize_reconstructions(num_images=5):
    # Load random images
    img_tensors, img_arrays = load_random_images(num_images=num_images)
    
    # Load VAE model
    model, device = VAE.load_vae_model()
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 3*num_images))
    fig.suptitle('Original vs Reconstructed Images', fontsize=16)
    
    for i, (img_tensor, img_array) in enumerate(zip(img_tensors, img_arrays)):
        # Prepare input
        input_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed, _, _ = model(input_tensor)
        
        # Convert to numpy for visualization
        reconstructed = reconstructed.squeeze(0).cpu().numpy()
        reconstructed = np.transpose(reconstructed, (1, 2, 0))  # Convert from CHW to HWC
        
        # Plot original image
        axes[i, 0].imshow(img_array)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Plot reconstructed image
        axes[i, 1].imshow(reconstructed)
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_reconstructions(num_images=3)