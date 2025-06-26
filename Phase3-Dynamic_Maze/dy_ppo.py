# train_ppo_local.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize #<-- IMPORT THIS
from stable_baselines3.common.env_checker import check_env
import pygame

from dy_maze_class import MazeGameEnv 

# Corrected Maze Configuration
base_maze_config = [
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', 'G'],
    ['.', '.', 'D', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', 'D', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', 'D', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['S', '.', '.', '.', '.', '.', '.', '.', '.', '.']
]
NUM_OBSTACLES =  5

def main():
    def env_creator(render_mode_val=None):
        return MazeGameEnv(
            base_maze=base_maze_config,
            num_obstacles=NUM_OBSTACLES,
            render_mode=render_mode_val
        )

    print("Checking the 'Local Perception' environment...")
    check_env(env_creator())
    print("Environment check passed!")

    # --- CHANGE 1: Create the raw vectorized environment FIRST ---
    raw_vec_env = make_vec_env(lambda: env_creator(), n_envs=4)

    # --- CHANGE 2: Wrap the environment with VecNormalize ---
    # This wrapper will normalize observations and, optionally, rewards.
    # We only normalize observations because our reward shaping is already well-scaled.
    print("Applying VecNormalize wrapper to observations.")
    vec_env = VecNormalize(raw_vec_env, norm_obs=True, norm_reward=False, gamma=0.99)


    # --- CHANGE 3: Tune hyperparameters for more stability ---
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_local_norm_maze_tensorboard/",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        # A slightly smaller learning rate can prevent the policy from collapsing
        learning_rate=1e-4, 
        # A slightly smaller entropy coefficient encourages the policy to converge
        ent_coef=0.005,
        clip_range=0.2 # PPO's clipping is already a great stabilizer
    )

    TOTAL_TIMESTEPS = 300000
    print(f"Training PPO agent with Local Perception and NORMALIZED observations for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    print("Training finished.")

    MODEL_PATH = "ppo_local_normalized_model"
    print(f"Saving model to {MODEL_PATH}.zip")
    model.save(MODEL_PATH)
    
    # It's important to save the normalization stats as well!
    vec_env.save(f"{MODEL_PATH}_vecnormalize.pkl")


    # --- Testing Loop ---
    print("\n--- Testing the trained agent ---")
    
    # Load the trained agent and the normalization stats
    loaded_model = PPO.load(MODEL_PATH)
    test_vec_env = VecNormalize.load(f"{MODEL_PATH}_vecnormalize.pkl", make_vec_env(lambda: env_creator(render_mode_val='human'), n_envs=1))
    
    # Set to evaluation mode
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    for episode in range(10):
        obs = test_vec_env.reset()
        terminated = False
        total_episode_reward = 0
        current_steps = 0
        print(f"\n--- Starting Test Episode {episode + 1} ---")
        test_vec_env.render()
        pygame.time.wait(1000)

        while not terminated:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, info = test_vec_env.step(action)
            total_episode_reward += reward[0] # Reward is a vector now
            test_vec_env.render()
            pygame.time.wait(100)
            current_steps += 1
        
        if total_episode_reward > 5:
             print(f"SUCCESS: Episode solved in {current_steps} steps! Reward: {total_episode_reward:.2f}")
        else:
             print(f"FAILURE: Episode finished. Reward: {total_episode_reward:.2f}")

    test_vec_env.close()

if __name__ == '__main__':
    main()