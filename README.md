# Custom Gym Maze Environment with Death-Pit

This project implements a custom reinforcement learning environment using the Gymnasium (formerly OpenAI Gym) API. The environment is a simple 2D maze where an agent navigates from a starting point ('S') to a goal ('G'), avoiding obstacles ('#') and deadly pits ('D'). The environment is rendered using Pygame.

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

## Prerequisites

*   Python 3.8+
*   pip (Python package installer)

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