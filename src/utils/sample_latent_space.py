import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import argparse

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)


from models.vae import VAE
from models.device import get_device_name

def sample_latent_space(model, num_samples, latent_dim, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)
        return samples

def visualize_images(samples):
    fig, axes = plt.subplots(1, len(samples), figsize=(20, 4))
    for i, ax in enumerate(axes):
        sample = samples[i].permute(1, 2, 0)
        sample = sample.cpu().numpy()
        ax.imshow(sample)
        ax.axis('off')
    plt.show()

def load_model(model_path):
    model = VAE(latent_dim=128, input_channels=3, input_height=128, input_width=256)
    model.load_state_dict(torch.load(model_path))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='models/vae/vae_final.pt')
    args = parser.parse_args()

    device = get_device_name()
    model = load_model(args.model_path).to(device)
    samples = sample_latent_space(model, args.num_samples, args.latent_dim, device)
    visualize_images(samples)

if __name__ == '__main__':
    main()
