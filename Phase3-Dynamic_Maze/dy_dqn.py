# train_dqn.py
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import DQN  # <-- Import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import pygame
from gymnasium import spaces
# Your environment and custom CNN are unchanged
from dy_maze_class import MazeGameEnv 

class CustomCnnExtractor(BaseFeaturesExtractor):
    """Custom CNN extractor for our 10x10 maze."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# --- Configuration for the Dynamic Maze (Unchanged) ---
base_maze_config = [
    ['S', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['D', '.', 'D', '.', '.', '.', '.', 'D', '.', '.'],
    ['.', '.', '.', '.', '.', 'D', '.', '.', '.', '.'], ['.', 'D', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', 'D', '.'], ['.', '.', 'D', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', 'D', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', 'D', '.'],
    ['.', 'D', '.', '.', '.', '.', 'D', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', 'G']
]
NUM_OBSTACLES = 15

def main():
    def env_creator(render_mode_val=None):
        return MazeGameEnv(
            base_maze=base_maze_config,
            num_obstacles=NUM_OBSTACLES,
            render_mode=render_mode_val
        )

    print("Checking the environment with `check_env`...")
    check_env(env_creator())
    print("Environment check passed!")

    # Vectorized environments help fill the replay buffer faster
    vec_env = make_vec_env(lambda: env_creator(), n_envs=4)

    policy_kwargs = dict(
        features_extractor_class=CustomCnnExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # --- Instantiate DQN Model ---
    model = DQN(
        "CnnPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./dqn_dynamic_maze_tensorboard/",
        
        # --- DQN-specific hyperparameters ---
        buffer_size=100_000,      # Size of the replay buffer (how many past experiences to store)
        learning_rate=1e-4,       # Learning rate for the optimizer
        learning_starts=5000,     # How many steps to take to fill the buffer before training starts
        batch_size=64,            # How many experiences to sample from the buffer for each training update
        gamma=0.99,               # Discount factor for future rewards
        train_freq=(4, "step"),   # Train the model every 4 steps
        gradient_steps=1,         # How many gradient steps to do after each rollout
        target_update_interval=1000, # How often to update the target network (in steps)
        
        # --- Epsilon-greedy exploration schedule ---
        exploration_fraction=0.2,       # Fraction of training to decay exploration rate over
        exploration_initial_eps=1.0,    # Starting exploration rate (100% random)
        exploration_final_eps=0.05,     # Final exploration rate (5% random)
    )

    TOTAL_TIMESTEPS = 500000
    print(f"Training DQN agent with Custom CNN for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    print("Training finished.")

    MODEL_PATH = "dqn_dynamic_custom_cnn_maze_model"
    print(f"Saving model to {MODEL_PATH}.zip")
    model.save(MODEL_PATH)

    # --- Testing Loop (Unchanged) ---
    print("\n--- Testing the trained DQN agent on new, unseen maze layouts ---")
    test_env = env_creator(render_mode_val='human')

    num_test_episodes = 10
    for episode in range(num_test_episodes):
        obs, info = test_env.reset()
        terminated, truncated = False, False
        total_episode_reward = 0
        current_steps = 0
        print(f"\n--- Starting Test Episode {episode + 1} ---")
        
        test_env.render()
        pygame.time.wait(1000)

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_episode_reward += reward
            test_env.render()

            running_test = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_test = False; break
            if not running_test: terminated = True; break
            pygame.time.wait(100)
            current_steps += 1
        
        # Adjusted the success check for the new reward shaping
        if terminated and total_episode_reward > 5: 
             print(f"SUCCESS: Episode {episode + 1} solved in {current_steps} steps! Total reward: {total_episode_reward:.2f}")
        else:
             print(f"FAILURE: Episode {episode + 1} finished in {current_steps} steps. Total reward: {total_episode_reward:.2f}")

        if not running_test: break

    print("Closing environments...")
    vec_env.close()
    test_env.close()

if __name__ == '__main__':
    main()