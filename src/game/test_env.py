import numpy as np
from racing_env import RacingEnvironment

def main():
    # Create environment
    env = RacingEnvironment()
    
    # Run one episode
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("Starting episode...")
    print("Controls:")
    print("- Use arrow keys to control the car")
    print("- ESC to exit")
    
    while not done:
        # Get random action
        action = (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))
        
        # Take action
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Print info every 100 steps
        if steps % 100 == 0:
            print(f"Step {steps}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            print(f"Distance to finish: {info['distance_to_finish']:.1f}")
            print(f"Speed: {info['speed']:.1f}")
            print("---")
        
        # Render the environment
        env.render()
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main() 