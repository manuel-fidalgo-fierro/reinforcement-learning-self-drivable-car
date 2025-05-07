import os
import random
import argparse
from game.screenshot import get_game_state_image
import numpy as np
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from panda3d.core import loadPrcFileData
from data_collectors.data_collectors_utils import window_position
from models.vae import VAE
import torch

# Print the module name
print(f"Current module name: {__name__}")


def run_game_instance(instance_id, output_dir, num_samples, x_pos, y_pos, win_size_w=256, win_size_h=128, vae_model=None):
    """
    Run a single game instance in a separate process.
    """
    print(f"Instance {instance_id}: Creating game.")
    try:
        # Create unique output directory for this instance
        instance_dir = Path(output_dir) / f"instance_{instance_id}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the game
        from game.car_game import RacingGame
        loadPrcFileData("", f"""
        window-title Racing Game {instance_id}
        win-origin {x_pos} {y_pos}
        win-size {win_size_w} {win_size_h}
        framebuffer-multisample 1
        multisamples 2
        show-frame-rate-meter 1
        """)
        # Import needs to be here to avoid the game being launched during import time.
        game = RacingGame()
        
        
        # Collect data
        for i in range(num_samples):
            print(f"Instance {instance_id}: Starting simulation {i} of {num_samples}")
            # Generate random controls
            acceleration = random.choice([1])
            turning = random.choice([-1, 0, 1])
            
            # Set the controls
            game.setControl("acceleration", acceleration)
            game.setControl("turning", turning)
            

            # Run one frame of the game
            game.taskMgr.step()
    
            # Take a screenshot and convert to tensor
            image = get_game_state_image(game)    
            image = image[:, :, ::-1].copy() # Convert BGR to RGB
            image = torch.from_numpy(image).float() / 255.0 # Convert to tensor and normalize to [0, 1]
            image = image.permute(2, 0, 1) # Reshape from (H, W, C) to (C, H, W) format
            input_tensor = image.unsqueeze(0)  # Add batch dimension

            # calculate reward
            reward = game.get_reward()
            
            # run the vae encoder
            mu, logvar = vae_model.encode(input_tensor)
            z = vae_model.reparameterize(mu, logvar).detach().cpu().numpy()
            
            # Reshape z to 1D array and convert scalars to 1D arrays
            z = z.reshape(-1)  # Flatten to 1D array
            acceleration = np.array([acceleration])
            turning = np.array([turning])
            reward = np.array([reward])
            
            # save the latent vector, reward, and controls as a single numpy array
            data = np.concatenate([z, acceleration, turning, reward]) # 128 + 1 + 1 + 1 = 131
            data_file = instance_dir / f"data_{instance_id}_{i}.npy"
            np.save(data_file, data)

            if game.isGameOver or game.isGameWon:
                break
            
        print(f"Instance {instance_id}: Data collection complete!")
    
    finally:
        # Clean up the game instance
        if 'game' in locals():
            game.destroy()
            print(f"Instance {instance_id}: Game instance cleaned up")


def collect_data(output_dir="data_mdn_rnn", num_samples=100, num_instances=300, num_parallelism=20, vae_model=None):
    """
    Collect data using multiple game instances running in parallel.
    args:
        output_dir: The directory to save the collected data.
        num_samples: The number of samples to collect per instance.
        num_instances: The number of game instances to run.
        num_parallelism: The number of game instances to run in parallel.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    screen_w, screen_h = 1792, 1008
    win_size_w, win_size_h = 256, 128
    
    # Calculate number of full batches and remaining instances
    num_batches = num_instances // num_parallelism
    for batch in range(num_batches):
        processes = []
        x_pos, y_pos = 0, 0
        
        # Start all processes in the batch
        for i in range(num_parallelism):
            instance_id = batch * num_parallelism + i
            p = mp.Process(target=run_game_instance, 
                        args=(instance_id, output_dir, num_samples, x_pos, y_pos, win_size_w, win_size_h, vae_model))
            processes.append(p)
            p.start()
            
            # Update the window position for next instance
            x_pos, y_pos = window_position(x_pos, y_pos, win_size_w, win_size_h, screen_w, screen_h)
        
        # Wait for all processes in the batch to complete
        for p in processes:
            p.join()
        
        print(f"Completed batch {batch + 1}/{num_batches}")
    
    print(f"\nAll instances completed! Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data from the racing game")
    parser.add_argument("--output_dir", type=str, default="data_mdn_rnn",
                        help="Directory to save collected data")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to collect per instance")
    parser.add_argument("--instances", type=int, default=300,
                        help="Number of game instances to run.")
    parser.add_argument("--parallelism", type=int, default=20,
                        help="Number of game instances to run in parallel.")
    parser.add_argument("--vae-model-path", type=str, default="models/vae/vae_final.pt",
                        help="Path to the VAE model")
    parser.add_argument("--latent-dim", type=int, default=128,
                        help="Latent dimension of the VAE model")
    
    args = parser.parse_args()
    print(f"Running game instances...{args}")
    assert args.instances % args.parallelism == 0, "Number of instances must be divisible by number of parallelism"
    
    # Load model and move to CPU for multiprocessing
    vae_model, _ = VAE.load_vae_model(args.vae_model_path, args.latent_dim)
    vae_model = vae_model.cpu()  # Move model to CPU
    vae_model.eval()  # Set to evaluation mode
    
    collect_data(args.output_dir, args.samples, args.instances, args.parallelism, vae_model) 