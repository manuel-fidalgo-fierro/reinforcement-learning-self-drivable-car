import numpy as np
from car_game import RacingGame
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3, Vec3, NodePath, CollisionTraverser, CollisionHandlerQueue

class RacingEnvironment:
    def __init__(self):
        # Initialize the game
        self.game = RacingGame()
        self.game.setup()
        
        # Define action space
        self.action_space = {
            'acceleration': [-1, 0, 1],  # brake, nothing, accelerate
            'turning': [-1, 0, 1]        # left, straight, right
        }
        
        # Define observation space dimensions
        self.state_dim = 8  # We'll define this in get_state()
        
        # Initialize collision detection for state observation
        self.cTrav = CollisionTraverser()
        self.collisionHandler = CollisionHandlerQueue()
        
    def reset(self):
        """Reset the environment to initial state"""
        self.game = RacingGame()
        self.game.setup()
        return self.get_state()
    
    def step(self, action):
        """
        Execute one step in the environment
        action: tuple of (acceleration, turning) values
        returns: (state, reward, done, info)
        """
        # Convert action to game controls
        accel, turn = action
        self.game.controls = {
            "acceleration": accel,
            "turning": turn
        }
        
        # Run one step of the game
        self.game.update(None)
        
        # Get new state
        state = self.get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.game.isGameOver
        
        # Additional info
        info = {
            'distance_to_finish': self._get_distance_to_finish(),
            'speed': self.game.carSpeed,
            'heading': self.game.carHeading
        }
        
        return state, reward, done, info
    
    def get_state(self):
        """
        Get the current state observation
        Returns: numpy array of state values
        """
        # Get car state
        car_pos = self.game.carPos
        car_speed = self.game.carSpeed
        car_heading = self.game.carHeading
        
        # Get distance to finish line (blue plane)
        distance_to_finish = self._get_distance_to_finish()
        
        # Get track boundaries info
        track_left = 25 - car_pos.x
        track_right = 25 + car_pos.x
        track_front = 2000 - car_pos.y
        track_back = 2000 + car_pos.y
        
        # Normalize values to [-1, 1] range
        state = np.array([
            car_pos.x / 25.0,           # Normalized x position
            car_pos.y / 2000.0,         # Normalized y position
            car_speed / self.game.maxSpeed,  # Normalized speed
            car_heading / 180.0,        # Normalized heading
            distance_to_finish / 4000.0,  # Normalized distance to finish
            track_left / 25.0,          # Normalized distance to left boundary
            track_right / 25.0,         # Normalized distance to right boundary
            track_front / 2000.0        # Normalized distance to front boundary
        ])
        
        return state
    
    def _calculate_reward(self):
        """Calculate reward based on current state"""
        reward = 0
        
        # Reward for making progress towards finish line
        current_distance = self._get_distance_to_finish()
        reward += (self._last_distance - current_distance) * 0.1
        
        # Penalty for going off track
        if abs(self.game.carPos.x) > 25 or abs(self.game.carPos.y) > 2000:
            reward -= 100
        
        # Penalty for collisions
        if self.game.collisionHandler.getNumEntries() > 0:
            reward -= 100
        
        # Reward for maintaining good speed
        if self.game.carSpeed > 0:
            reward += 0.1
        
        # Large reward for reaching finish line
        if self.game.checkWin():
            reward += 1000
        
        # Small time penalty to encourage speed
        reward -= 0.01
        
        self._last_distance = current_distance
        return reward
    
    def _get_distance_to_finish(self):
        """Calculate distance to finish line"""
        return abs(2000 - self.game.carPos.y)
    
    def render(self):
        """Render the current state"""
        self.game.run()
    
    def close(self):
        """Clean up resources"""
        self.game.cleanup()

# Create environment
env = RacingEnvironment()

# Example episode
state = env.reset()
done = False
total_reward = 0

while not done:
    # Get action from your RL agent (random for now)
    action = (np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1]))
    
    # Take action
    state, reward, done, info = env.step(action)
    total_reward += reward
    
    # Render if needed
    env.render()

print(f"Episode finished with total reward: {total_reward}") 