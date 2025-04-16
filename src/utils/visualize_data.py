import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from PIL import Image
import os
import argparse

def load_and_display_images(image_files, num_images=5, save_path=None):
    """
    Load and display random images from the dataset.
    
    Args:
        image_files: List of paths to .npy files
        num_images: Number of images to display
        save_path: Optional path to save the visualization
    """
    # Select random images
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    # Create figure
    fig, axes = plt.subplots(1, len(selected_files), figsize=(20, 4))
    if len(selected_files) == 1:
        axes = [axes]
    
    # Display each image
    for ax, file_path in zip(axes, selected_files):
        # Load image
        image = np.load(file_path)
        image = image[:, :, ::-1] # Convert BGR to RGB
        H, W, C = image.shape
        # Display image
        ax.imshow(image)
        ax.set_title(f"Dims: ({H}x{W}x{C})")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def display_instance_images(instance_dir, num_images=5, save_path=None):
    """
    Display images from a specific instance directory.
    
    Args:
        instance_dir: Path to instance directory
        num_images: Number of images to display
        save_path: Optional path to save the visualization
    """
    # Get all .npy files in the instance directory
    instance_files = list(instance_dir.glob('*.npy'))
    
    # Create figure
    fig, axes = plt.subplots(1, min(num_images, len(instance_files)), figsize=(20, 4))
    if len(instance_files) == 1:
        axes = [axes]
    
    # Display each image
    for ax, file_path in zip(axes, instance_files[:num_images]):
        # Load image
        image = np.load(file_path)
        
        # Display image
        ax.imshow(image)
        ax.set_title(f"{file_path.name}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_image_statistics(image_files, sample_size=10):
    """
    Analyze basic statistics of the images.
    
    Args:
        image_files: List of paths to .npy files
        sample_size: Number of images to analyze
    """
    # Sample random images
    sample_files = random.sample(image_files, min(sample_size, len(image_files)))
    
    # Collect statistics
    shapes = []
    min_values = []
    max_values = []
    mean_values = []
    
    for file_path in sample_files:
        image = np.load(file_path)
        shapes.append(image.shape)
        min_values.append(image.min())
        max_values.append(image.max())
        mean_values.append(image.mean())
    
    # Print statistics
    print(f"Image shapes: {set(shapes)}")
    print(f"Min values: {min(min_values):.2f} - {max(min_values):.2f}")
    print(f"Max values: {min(max_values):.2f} - {max(max_values):.2f}")
    print(f"Mean values: {min(mean_values):.2f} - {max(mean_values):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize collected game data')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing collected images')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to display')
    parser.add_argument('--instance', type=int, default=None,
                        help='Instance number to display (if None, shows random images)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save visualization (if None, shows plot)')
    
    args = parser.parse_args()
    
    # Set the path to collected data
    data_dir = Path(args.data_dir)
    
    # Get all .npy files
    image_files = list(data_dir.glob('**/*.npy'))
    print(f"Found {len(image_files)} images")
    
    # Analyze image statistics
    print("\nImage Statistics:")
    analyze_image_statistics(image_files)
    
    # Display images
    if args.instance is not None:
        # Display images from specific instance
        instance_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if 0 <= args.instance < len(instance_dirs):
            print(f"\nDisplaying images from instance {args.instance}:")
            display_instance_images(instance_dirs[args.instance], 
                                  num_images=args.num_images,
                                  save_path=args.save_path)
        else:
            print(f"Instance {args.instance} not found. Available instances: 0-{len(instance_dirs)-1}")
    else:
        # Display random images
        print("\nDisplaying random images:")
        load_and_display_images(image_files, 
                              num_images=args.num_images,
                              save_path=args.save_path)

if __name__ == '__main__':
    main() 