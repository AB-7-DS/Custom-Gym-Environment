# train_ppo.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env # For creating vectorized environments
from stable_baselines3.common.env_checker import check_env
import pygame # For pygame.QUIT event handling during testing

# Import your custom environment
from maze_class import MazeGameEnv # Make sure maze_class.py is in the same directory or accessible in PYTHONPATH

# Maze configuration (same as in your maze_env.py)
maze_config = [
    ['S', '.', '.', 'D'],
    ['.', '#', 'D', '#'],
    ['.', '.', '.', '.'],
    ['#', 'D', '#', 'G']
]

def main():
    # --- 1. Create and check the environment ---
    print("Creating the environment...")

    # It's good practice to wrap the environment creation in a function for make_vec_env
    def env_creator():
        # For training, render_mode=None is usually best for speed.
        # If you need to record videos during training, SB3 wrappers handle that.
        env = MazeGameEnv(maze=maze_config, render_mode=None)
        return env

    # First, check a single instance of the environment
    print("Checking the custom environment with Stable Baselines3 check_env...")
    try:
        # Create a temporary instance for checking
        # render_mode can be 'human' or None for the check.
        # If 'human', it also tests rendering aspects.
        check_env_instance = MazeGameEnv(maze=maze_config, render_mode='human')
        check_env(check_env_instance, warn=True)
        print("Environment passed SB3 check_env.")
        check_env_instance.close() # Important to close the instance used for checking
    except Exception as e:
        print(f"SB3 check_env failed or issued warnings: {e}")
        print("Exiting due to environment check failure. Please resolve issues.")
        return # Stop if the environment is not compatible

    # Create a vectorized environment for training (even if n_envs=1)
    # This is standard practice for SB3.
    print("Creating vectorized environment for training...")
    vec_env = make_vec_env(env_creator, n_envs=1)
    # Note: If you have a more complex observation space, you might wrap vec_env
    # with VecNormalize, e.g., from stable_baselines3.common.vec_env import VecNormalize
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    # For this simple coordinate-based observation, it might not be strictly necessary.

    # --- 2. Initialize the PPO agent ---
    print("Initializing PPO agent...")
    # "MlpPolicy" is suitable for this environment (Box observation, Discrete action).
    # tensorboard_log will save training logs for viewing in TensorBoard.
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_maze_tensorboard/")
    
    # You can customize PPO hyperparameters, e.g.:
    # model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=256, batch_size=64, n_epochs=10,
    #             gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
    #             tensorboard_log="./ppo_maze_tensorboard/")


    # --- 3. Train the agent ---
    TOTAL_TIMESTEPS = 50000  # Adjust as needed. More steps generally lead to better learning.
    print(f"Training PPO agent for {TOTAL_TIMESTEPS} timesteps...")
    # SB3 automatically handles callbacks, logging, etc.
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True )
    print("Training finished.")

    # --- 4. Save the trained model ---
    MODEL_PATH = "ppo_maze_model"
    print(f"Saving model to {MODEL_PATH}.zip")
    model.save(MODEL_PATH)
    
    # If you used VecNormalize, you should also save its running statistics:
    # if isinstance(vec_env, VecNormalize):
    #     vec_env.save("vecnormalize_maze.pkl")


    # --- 5. Test the trained agent ---
    print("\n--- Testing the trained agent ---")
    # Load the model (optional, as 'model' is already in memory, but good practice to show how)
    # loaded_model = PPO.load(MODEL_PATH)

    # Create a new, non-vectorized environment for testing with human rendering
    test_env = MazeGameEnv(maze=maze_config, render_mode='human')
    # If you used VecNormalize for training, you'd need to wrap test_env in VecNormalize
    # and load the saved statistics:
    # test_env_norm = VecNormalize.load("vecnormalize_maze.pkl", make_vec_env(lambda: MazeGameEnv(maze=maze_config, render_mode='human'), n_envs=1))
    # test_env_norm.training = False # Important for evaluation
    # test_env_norm.norm_reward = False

    num_test_episodes = 10
    max_steps_per_episode = 100 # Max steps to prevent infinite loops if agent is stuck

    for episode in range(num_test_episodes):
        obs, info = test_env.reset()
        terminated = False
        truncated = False
        total_episode_reward = 0
        current_steps = 0
        print(f"\n--- Starting Test Episode {episode + 1} ---")
        
        test_env.render() # Render initial state
        pygame.time.wait(500) # Pause to see initial state

        while not (terminated or truncated) and current_steps < max_steps_per_episode:
            # For testing, use deterministic actions for PPO
            action, _states = model.predict(obs, deterministic=True)
            # If using a loaded model: action, _states = loaded_model.predict(obs, deterministic=True)
            # If using VecNormalize: action, _states = model.predict(test_env_norm.normalize_obs(obs), deterministic=True)
            
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_episode_reward += reward
            
            test_env.render() # Render the environment after each step

            print(f"Step: {current_steps+1}, Action: {action}, Obs: {obs}, Reward: {reward:.2f}, Term: {terminated}, Trunc: {truncated}")
            
            # Handle Pygame events for window responsiveness and to catch QUIT event
            running_test = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Pygame window closed during testing.")
                    running_test = False
                    break
            if not running_test:
                terminated = True # Force end of episode and test loop
                break

            pygame.time.wait(200) # Milliseconds to wait, slows down for visualization
            current_steps += 1
        
        print(f"Episode {episode + 1} finished. Total reward: {total_episode_reward:.2f}")
        if not running_test: # If window was closed
            break

    # --- 6. Close environments ---
    print("Closing environments...")
    vec_env.close() # Close the training environment
    test_env.close()  # Close the testing environment
    # if 'test_env_norm' in locals(): # If VecNormalize wrapper was used for testing
    #    test_env_norm.close()

    print("\nTraining and testing script finished.")
    print(f"To view training logs, run: tensorboard --logdir={model.tensorboard_log}")

if __name__ == '__main__':
    # Before running, ensure you have the necessary libraries:
    # pip install stable-baselines3[extra] pygame gymnasium
    main()