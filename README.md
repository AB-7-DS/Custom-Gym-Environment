# Phase 1: Custom Gym Maze Environment with Death-Pit

This project implements a custom reinforcement learning environment using the Gymnasium (formerly OpenAI Gym) API. The environment is a simple 2D maze where an agent navigates from a starting point ('S') to a goal ('G'), avoiding obstacles ('#') and deadly pits ('D'). The environment is rendered using Pygame.

## Demo Video

You can view a demonstration of the environment in action by clicking the link below:

[Watch the Maze Demo Video](/demo/Phase1.mp4)

---

## Features

*   Custom Gymnasium environment (`MazeGameEnv`).
*   Configurable maze layout.
*   Player agent, start, goal, obstacles, and death-pits.
*   Visual rendering using Pygame.
*   Discrete action space (Up, Down, Left, Right).
*   Box observation space representing the agent's [row, col] position.
*   Compatibility with Stable-Baselines3 (verified with `check_env`).

## Files

*   `maze_env.py`: Contains the `MazeGameEnv` class definition.
*   `run_maze.py`: An example script to instantiate, run, and test the environment.
*   `README.md`: This file.



## Setup Instructions

1.  **Clone the Repository (or download the files):**
    If this were a full repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
    Otherwise, ensure `maze_env.py` and `maze_class.py` are in the same directory.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    You can install the required packages directly:

    ```bash
    pip install -r requirements.txt
    ```
    if it is not working then:
     ```bash
    pip install gymnasium pygame numpy stable-baselines3
    ```


## Usage

To run the example simulation with a random agent:

```bash
python maze_env.py
```
---
---

# Phase 2: Implementation of PPO Agent in Custom Gym Environment

This Phase of the project demonstrates how to train a Proximal Policy Optimization (PPO) agent from Stable Baselines3 to solve it.

## Updated Folder Structure

```bash
.
├── maze_class.py # Defines the MazeGameEnv class
├── train_ppo.py # Script to train and test the PPO agent
├── maze_env.py # Original script for testing the environment manually (optional)
├── ppo_maze_model.zip # Saved trained PPO model (after running train_ppo.py)
├── ppo_maze_tensorboard/ # Directory for TensorBoard logs (after running train_ppo.py)
├── docs/
| └── PPO_Output_Explanation.pdf 
├── demo/ 
│ └── Phase1.mp4
| └── PPO_demo.mp4
└── README.md 
```

---

## Demo Video

You can view a demonstration of the environment in action by clicking the link below:

[Watch the Maze Demo Video](/demo/PPO_demo.mp4)

---


## Files

*   (All Files are same as used in phase 1)
*   
*   `test_ppo.py`: New file created where PPO agent is trained.

---

## Setup Instructions

1.  **Clone the Repository (or download the files):**
    If this were a full repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
  

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    You can install the required packages directly:

    ```bash
    pip install -r requirements.txt
    ```
    if it is not working then:
     ```bash
    pip install gymnasium pygame numpy stable-baselines3
    ```


## Running instructions

To run the example simulation with a PPO agent:

```bash
python test_ppo.py
```
## PPO Agent Training

The script `train_ppo.py` uses the [Stable Baselines3](https://stable-baselines3.readthedocs.io/) library to train a Proximal Policy Optimization (PPO) agent on the `MazeGameEnv`.

### Key PPO Hyperparameters Used:

(You can list the main parameters you used if you customized them, otherwise mention defaults were used)
*   **Policy Network:** `MlpPolicy` (Multi-Layer Perceptron)
*   **Total Timesteps:** 50,000 (configurable in `train_ppo.py`)
*   **`n_steps`:** 2048 (default, or specify if changed)
*   **`batch_size`:** 64 (default, or specify if changed)
*   **`n_epochs`:** 10 (default, or specify if changed)
*   **`gamma` (Discount Factor):** 0.99
*   **`gae_lambda`:** 0.95
*   **`ent_coef` (Entropy Coefficient):** 0.01
*   **`learning_rate`:** 0.0003

### Training Process:

1.  **Environment Initialization:** A vectorized version of `MazeGameEnv` is created.
2.  **Agent Initialization:** A PPO agent is initialized with the specified policy and hyperparameters.
3.  **Learning:** The agent interacts with the environment, collecting experience (rollouts). After each rollout, the agent's policy and value functions are updated using the PPO algorithm.
4.  **Logging:** Training progress, including episode rewards, episode lengths, and various PPO-specific metrics, are logged to the console and to a TensorBoard directory (`ppo_maze_tensorboard/`).
5.  **Model Saving:** Upon completion of training, the trained agent's model is saved to `ppo_maze_model.zip`.

### Understanding PPO Training Output

The console output during training provides valuable insights into the learning process. Key metrics include:

*   **`rollout/ep_rew_mean`**: Average reward per episode. We aim for this to increase.
*   **`rollout/ep_len_mean`**: Average length of episodes. We aim for this to decrease as the agent finds more efficient paths.
*   **`train/explained_variance`**: How well the value function predicts returns. Should approach 1.
*   **`train/entropy_loss`**: Related to policy randomness. Tends to increase (become less negative) as the policy becomes more deterministic.
*   **`train/value_loss`**: Error of the value function. Should decrease.

**For a detailed explanation of all the metrics output by the PPO algorithm during training, please refer to the accompanying document: [PPO_Output_Explanation.pdf](/docs/PPO_Ouput_Explanation.pdf).**

You can also visualize the training progress using TensorBoard:
```bash
tensorboard --logdir=./ppo_maze_tensorboard/
```

## Prerequisites

*   Python 3.8+
*   pip (Python package installer)