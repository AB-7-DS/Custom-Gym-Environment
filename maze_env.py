import gymnasium as gym
from test_maze import MazeGameEnv # Assuming your environment class is in maze_env.py
import pygame
import numpy as np

# Maze configuration with a Start (S), Goal (G), Obstacles (#), Empty paths (.), and Death-pits (D)
maze_config = [
    ['S', '.', '.', 'D'],
    ['.', '#', '.', '#'],
    ['.', '.', '.', '.'], # Added another death-pit
    ['#', 'D', '#', 'G']  # Goal is now reachable, added a death-pit
]

# --- Option 1: Direct Instantiation (Recommended for simplicity here) ---
env = MazeGameEnv(maze=maze_config, render_mode='human')

# --- Option 2: Using gym.register and gym.make (useful for broader integration) ---
# from gymnasium.envs.registration import register
# if 'MazeGame-v0' not in gym.envs.registry: # Avoid re-registering if script is run multiple times
#     register(
#         id='MazeGame-v0',
#         entry_point='maze_env:MazeGameEnv', # Format: 'module_name:ClassName'
#     )
# # When using gym.make, pass kwargs for __init__ like this:
# env = gym.make('MazeGame-v0', maze=maze_config, render_mode='human')


# Test with Stable-Baselines3 environment checker (optional, but good practice)
try:
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True) # warn=True will show warnings, skip_render_check=False by default
    print("Environment passed SB3 check_env.")
except ImportError:
    print("Stable-Baselines3 not found, skipping check_env.")
except Exception as e:
    print(f"SB3 check_env failed or issued warnings: {e}")


# Main loop for testing the environment
obs, info = env.reset(seed=42) # Use a seed for consistent random actions if needed
env.render()

running = True
num_episodes = 5
max_steps_per_episode = 50 # To prevent infinitely running episodes with random agent

for episode in range(num_episodes):
    if not running:
        break
    
    print(f"\n--- Starting Episode {episode + 1} ---")
    current_episode_steps = 0
    total_episode_reward = 0.0
    
    terminated = False
    truncated = False

    while not (terminated or truncated) and current_episode_steps < max_steps_per_episode:
        for event in pygame.event.get(): # Essential for Pygame window to respond and catch close events
            if event.type == pygame.QUIT:
                running = False
                terminated = True # End current episode
                break
        if not running:
            break

        action = env.action_space.sample()  # Agent takes a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_episode_reward += reward
        
        env.render() # Render the environment after each step

        print(f"Step {current_episode_steps + 1}: Action: {action}, Obs: {obs}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")
        
        pygame.time.wait(150) # Milliseconds to wait, slows down for visualization

        current_episode_steps += 1
    
    if running: # Only print and reset if not exited due to pygame quit
        if truncated and not terminated:
            print(f"Episode {episode + 1} truncated after {current_episode_steps} steps.")
        else:
            print(f"Episode {episode + 1} finished after {current_episode_steps} steps.")
        print(f"Total reward for episode: {total_episode_reward:.2f}")

        if episode < num_episodes - 1: # Don't reset if it's the last episode
            obs, info = env.reset()
            env.render() # Render the reset state
            pygame.time.wait(500) # Pause to see the reset state

env.close()
print("\nDemo finished.")