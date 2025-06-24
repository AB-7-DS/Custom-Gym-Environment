import gymnasium as gym # Use gymnasium for the new API
from gymnasium import spaces
import numpy as np
import pygame

class MazeGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, maze, render_mode=None):
        super(MazeGameEnv, self).__init__()

        if maze is None:
            raise ValueError("Maze layout cannot be None.")
        self.maze = np.array(maze, dtype=object)  # Maze represented as a 2D numpy array
        self.num_rows, self.num_cols = self.maze.shape

        # Find S (start), G (goal)
        start_coords = np.argwhere(self.maze == 'S')
        if start_coords.size == 0:
            raise ValueError("Starting position 'S' not found in maze.")
        self.start_pos = start_coords[0]  # e.g., np.array([row, col])

        goal_coords = np.argwhere(self.maze == 'G')
        if goal_coords.size == 0:
            raise ValueError("Goal position 'G' not found in maze.")
        self.goal_pos = goal_coords[0]    # e.g., np.array([row, col])

        self.current_pos = self.start_pos.copy() # Current position of the agent

        # 4 possible actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)

        # Observation space: agent's [row, col] position
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.num_rows - 1, self.num_cols - 1]),
            dtype=np.int32
        )

        # Pygame setup for rendering
        self.cell_size = 100  # Size of each cell in pixels
        self.screen_width = self.num_cols * self.cell_size
        self.screen_height = self.num_rows * self.cell_size
        
        self.screen = None # For 'human' mode rendering screen
        self.clock = None  # For 'human' mode rendering clock

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Game Environment")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility
        self.current_pos = self.start_pos.copy()
        
        observation = self.current_pos.astype(np.int32)
        info = {} # Auxiliary information

        # No automatic rendering on reset, user calls env.render()
        return observation, info

    def step(self, action):
        new_pos = self.current_pos.copy()

        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1
        else:
            raise ValueError(f"Received invalid action={action}. Not in action space.")

        # Check if the new position is valid (within bounds and not an obstacle)
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos

        # Determine cell type at the agent's current position
        cell_type = self.maze[self.current_pos[0], self.current_pos[1]]

        terminated = False  # True if episode ends (goal or death-pit)
        reward = 0.0

        if cell_type == 'G':  # Goal
            reward = 1.0
            terminated = True
        elif cell_type == 'D':  # Death-pit
            reward = -1.0
            terminated = True
        # else: (empty path or start) reward = 0.0, terminated = False (default)
        
        # Optional: small negative reward for each step to encourage efficiency
        # if not terminated:
        #     reward = -0.01 

        observation = self.current_pos.astype(np.int32)
        truncated = False  # True if episode exceeds a time limit (not used here)
        info = {}

        # No automatic rendering on step, user calls env.render()
        return observation, reward, terminated, truncated, info

    def _is_valid_position(self, pos):
        row, col = pos
        # Check bounds
        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            return False
        # Check obstacles
        if self.maze[row, col] == '#':
            return False
        return True # Position is valid (can be 'S', 'G', 'D', or '.')

    def render(self):
        if self.render_mode == "human":
            if self.screen is None: # Initialize pygame if not done (e.g. render_mode set after __init__)
                self._init_pygame()
            return self._render_frame_human()
        elif self.render_mode == "rgb_array":
            return self._render_frame_rgb_array()
        # If render_mode is None, do nothing or handle as an error.

    def _render_frame_human(self):
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        self._draw_maze_on_canvas(canvas)
        
        self.screen.blit(canvas, canvas.get_rect())
        pygame.event.pump() # Process event queue
        pygame.display.flip() # Update the full display
        self.clock.tick(self.metadata["render_fps"])
        return None

    def _render_frame_rgb_array(self):
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        self._draw_maze_on_canvas(canvas)
        # Convert Pygame surface to NumPy array
        rgb_array = np.array(pygame.surfarray.pixels3d(canvas))
        return np.transpose(rgb_array, axes=(1, 0, 2)) # Pygame has (width, height, channels)

    def _draw_maze_on_canvas(self, canvas):
        canvas.fill((255, 255, 255))  # White background

        # Draw maze elements
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                cell_left = c * self.cell_size
                cell_top = r * self.cell_size
                rect = pygame.Rect(cell_left, cell_top, self.cell_size, self.cell_size)
                
                cell_char = self.maze[r, c]
                
                # Default for empty path '.' or unknown
                color = (220, 220, 220) # Light gray for empty path

                if cell_char == '#':      # Obstacle
                    color = (50, 50, 50)    # Dark Gray/Black
                elif cell_char == 'S':    # Starting position
                    color = (0, 255, 0)     # Green
                elif cell_char == 'G':    # Goal position
                    color = (255, 0, 0)     # Red
                elif cell_char == 'D':    # Death-pit
                    color = (128, 0, 128)   # Purple

                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (100, 100, 100), rect, 1) # Cell border

        # Draw agent
        agent_center_x = self.current_pos[1] * self.cell_size + self.cell_size // 2
        agent_center_y = self.current_pos[0] * self.cell_size + self.cell_size // 2
        agent_radius = self.cell_size // 3
        pygame.draw.circle(canvas, (0, 0, 255), (agent_center_x, agent_center_y), agent_radius) # Blue circle for agent

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None