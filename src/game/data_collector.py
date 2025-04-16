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


def run_game_instance(instance_id, output_dir, num_samples, x_pos, y_pos, win_size_w=256, win_size_h=128):
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

def window_position(x_pos, y_pos, win_size_w, win_size_h, screen_w, screen_h):
    """
    Calculate the window position for the next instance in a grid layout.
    args:
        x_pos: The current x position of the window.
        y_pos: The current y position of the window.
        win_size_w: The width of the window.
        win_size_h: The height of the window.
        screen_w: The width of the screen.
        screen_h: The height of the screen.
    returns:
        x_pos: The next x position of the window.
        y_pos: The next y position of the window.
    """
    # Calculate how many windows can fit in each row and column
    windows_per_row = screen_w // win_size_w
    windows_per_col = screen_h // win_size_h
    
    # Calculate current position in grid
    current_col = x_pos // win_size_w
    current_row = y_pos // win_size_h
    
    # Move to next position
    current_col += 1
    if current_col >= windows_per_row:
        current_col = 0
        current_row += 1
        if current_row >= windows_per_col:
            current_row = 0
    
    # Calculate new position
    x_pos = current_col * win_size_w
    y_pos = current_row * win_size_h
    
    print(f"Window position: {x_pos}, {y_pos} (Grid position: {current_col}, {current_row})")
    return x_pos, y_pos

def collect_data(output_dir="data", num_samples=100, num_instances=300, num_parallelism=20):
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
                        args=(instance_id, output_dir, num_samples, x_pos, y_pos, win_size_w, win_size_h))
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
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Directory to save collected data")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to collect per instance")
    parser.add_argument("--instances", type=int, default=300,
                        help="Number of game instances to run.")
    parser.add_argument("--parallelism", type=int, default=20,
                        help="Number of game instances to run in parallel.")
    
    args = parser.parse_args()
    print(f"Running game instances...{args}")
    assert args.instances % args.parallelism == 0, "Number of instances must be divisible by number of parallelism"
    collect_data(args.output_dir, args.samples, args.instances, args.parallelism) 