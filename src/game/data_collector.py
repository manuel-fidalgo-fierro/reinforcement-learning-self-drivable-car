import os
import random
import argparse
from screenshot import get_game_state_image
import numpy as np
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from panda3d.core import loadPrcFileData


# Print the module name
print(f"Current module name: {__name__}")

def run_game_instance(instance_id, output_dir, num_samples, x_pos, y_pos, win_size_x=128*2, win_size_y=128):
    """
    Run a single game instance in a separate process.
    """
    print(f"Instance {instance_id}: Creating game.")
    try:
        # Create unique output directory for this instance
        instance_dir = Path(output_dir) / f"instance_{instance_id}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the game
        from car_game import RacingGame
        loadPrcFileData("", f"""
        window-title Racing Game {instance_id}
        win-origin {x_pos} {y_pos}
        win-size {win_size_x} {win_size_y}
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
    
            # Take a screenshot
            image = get_game_state_image(game)
            
            # Save the screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"sample_{instance_id}_{timestamp}.npy"
            filepath = instance_dir / filename
            
            # Save as numpy array
            np.save(filepath, image)
            
            print(f"Instance {instance_id}: Saved sample {i+1}/{num_samples} to {filepath}")
            
            # Run one frame of the game
            game.taskMgr.step()
        
        print(f"Instance {instance_id}: Data collection complete!")
    
    finally:
        # Clean up the game instance
        if 'game' in locals():
            game.destroy()
            print(f"Instance {instance_id}: Game instance cleaned up")

def collect_data(output_dir="collected_data", num_samples=100, num_instances=2):
    """
    Collect data using multiple game instances running in parallel.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create processes for each game instance
    processes = []
    x_pos = 128*2
    y_pos = 128
    for i in range(num_instances):

        p = mp.Process(target=run_game_instance, 
                      args=(i, output_dir, num_samples, (x_pos*i)%3456, (y_pos*i)%2234))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"\nAll instances completed! Data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data from the racing game")
    parser.add_argument("--output_dir", type=str, default="collected_data",
                        help="Directory to save collected data")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to collect per instance")
    parser.add_argument("--instances", type=int, default=2,
                        help="Number of game instances to run in parallel")
    
    args = parser.parse_args()
    print(f"Running game instances...{args}")
    collect_data(args.output_dir, args.samples, args.instances) 