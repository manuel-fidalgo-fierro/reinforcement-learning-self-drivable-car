import random
import time
import os
from pathlib import Path
from car_game import RacingGame
from screenshot import get_game_state_image, get_compressed_state
from PIL import Image

class DataCollector:
    def __init__(self, output_dir="data/screenshots"):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save screenshots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define possible actions
        self.actions = {
            "forward": False,
            "reverse": False,
            "left": False,
            "right": False
        }
        
        # Game parameters
        self.screenshot_interval = 0.1  # Take screenshot every 0.1 seconds
        self.min_action_duration = 0.5  # Minimum time to hold an action
        self.max_action_duration = 2.0  # Maximum time to hold an action
        
    def random_action(self):
        """Generate a random action for the car."""
        # Reset all actions
        for key in self.actions:
            self.actions[key] = False
            
        # Randomly choose one or two actions
        num_actions = random.randint(1, 2)
        chosen_actions = random.sample(list(self.actions.keys()), num_actions)
        
        for action in chosen_actions:
            self.actions[action] = True
            
        return self.actions
    
    def collect_data(self, num_episodes=10):
        """
        Collect data by running the game multiple times.
        
        Args:
            num_episodes: Number of game episodes to run
        """
        print("Running episode.")
        for episode in range(num_episodes):
            print(f"Starting episode {episode + 1}/{num_episodes}")
            
            # Create game instance
            game = RacingGame()
            
            # Create episode directory
            episode_dir = self.output_dir / f"episode_{episode}"
            episode_dir.mkdir(exist_ok=True)
            
            frame_count = 0
            last_screenshot_time = 0
            last_action_time = time.time()
            current_action_duration = random.uniform(self.min_action_duration, self.max_action_duration)
            current_actions = self.random_action()
            
            while not game.isGameOver:
                current_time = time.time()
                
                # Take screenshot at regular intervals
                if current_time - last_screenshot_time >= self.screenshot_interval:
                    # Get game state image
                    img_array = get_game_state_image(game)
                    
                    # Save screenshot
                    screenshot_path = episode_dir / f"frame_{frame_count:06d}.jpg"
                    img = Image.fromarray(img_array)
                    img.save(screenshot_path, quality=85)
                    
                    frame_count += 1
                    last_screenshot_time = current_time
                
                # Update actions if duration has passed
                if current_time - last_action_time >= current_action_duration:
                    current_actions = self.random_action()
                    current_action_duration = random.uniform(self.min_action_duration, self.max_action_duration)
                    last_action_time = current_time
                
                # Apply current actions to the game
                game.keyMap = current_actions
                
                # Run one frame of the game
                game.taskMgr.step()
                
                # Small delay to prevent the game from running too fast
                time.sleep(0.01)
            
            print(f"Episode {episode + 1} completed. Collected {frame_count} frames.")
            game.destroy()
            
        print(f"Data collection completed. Total episodes: {num_episodes}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect game data for VAE training")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--output", type=str, default="data/screenshots",
                       help="Output directory for screenshots")
    
    args = parser.parse_args()
    
    collector = DataCollector(output_dir=args.output)
    collector.collect_data(num_episodes=args.episodes) 