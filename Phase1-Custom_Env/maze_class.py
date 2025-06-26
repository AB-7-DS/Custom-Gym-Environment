# maze_class.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class MazeGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, maze, render_mode=None):
        super(MazeGameEnv, self).__init__()

        if maze is None:
            raise ValueError("Maze layout cannot be None.")
        self.maze = np.array(maze, dtype=object)
        self.num_rows, self.num_cols = self.maze.shape

        start_coords = np.argwhere(self.maze == 'S')
        if start_coords.size == 0:
            raise ValueError("Starting position 'S' not found in maze.")
        self.start_pos = start_coords[0]

        goal_coords = np.argwhere(self.maze == 'G')
        if goal_coords.size == 0:
            raise ValueError("Goal position 'G' not found in maze.")
        self.goal_pos = goal_coords[0]

        self.current_pos = self.start_pos.copy()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.num_rows - 1, self.num_cols - 1]),
            dtype=np.int32
        )

        self.cell_size = 100
        self.screen_width = self.num_cols * self.cell_size
        self.screen_height = self.num_rows * self.cell_size
        
        self.screen = None
        self.clock = None

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
        super().reset(seed=seed)
        self.current_pos = self.start_pos.copy()
        observation = self.current_pos.astype(np.int32)
        info = {}
        return observation, info

    def step(self, action):
        new_pos = self.current_pos.copy()
        if action == 0: new_pos[0] -= 1   # Up
        elif action == 1: new_pos[0] += 1 # Down
        elif action == 2: new_pos[1] -= 1 # Left
        elif action == 3: new_pos[1] += 1 # Right
        else:
            raise ValueError(f"Received invalid action={action}. Not in action space.")

        if self._is_valid_position(new_pos):
            self.current_pos = new_pos

        cell_type = self.maze[self.current_pos[0], self.current_pos[1]]
        terminated = False
        reward = 0.0

        if cell_type == 'G':
            reward = 1.0
            terminated = True
        elif cell_type == 'D':
            reward = -1.0
            terminated = True
        # Penalty for each step to encourage shorter paths (optional, but often helpful)
        else:
            reward = -0.01 # Small negative reward for each step

        observation = self.current_pos.astype(np.int32)
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def _is_valid_position(self, pos):
        row, col = pos
        if not (0 <= row < self.num_rows and 0 <= col < self.num_cols):
            return False
        if self.maze[row, col] == '#':
            return False
        return True

    def render(self):
        if self.render_mode == "human":
            if self.screen is None: # Safeguard if render_mode changed or init failed
                self._init_pygame()
            return self._render_frame_human()
        elif self.render_mode == "rgb_array":
            # For rgb_array, ensure pygame base is initialized for surface creation.
            if not pygame.get_init(): # Checks if any pygame module is initialized
                pygame.init() # Initializes all imported Pygame modules. Safe to call multiple times.
            return self._render_frame_rgb_array()
        # If render_mode is None or not supported, do nothing or return None

    def _render_frame_human(self):
        # This check should ideally be done before calling _render_frame_human
        # but as a safeguard:
        if self.screen is None or self.clock is None:
             self._init_pygame() # Ensure pygame is initialized for human mode

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        self._draw_maze_on_canvas(canvas)
        
        self.screen.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        return None

    def _render_frame_rgb_array(self):
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        self._draw_maze_on_canvas(canvas)
        rgb_array = np.array(pygame.surfarray.pixels3d(canvas))
        return np.transpose(rgb_array, axes=(1, 0, 2))

    def _draw_maze_on_canvas(self, canvas):
        canvas.fill((255, 255, 255))
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                cell_char = self.maze[r, c]
                color = (220, 220, 220)
                if cell_char == '#': color = (50, 50, 50)
                elif cell_char == 'S': color = (0, 255, 0)
                elif cell_char == 'G': color = (255, 0, 0)
                elif cell_char == 'D': color = (128, 0, 128)
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (100, 100, 100), rect, 1)

        agent_center_x = self.current_pos[1] * self.cell_size + self.cell_size // 2
        agent_center_y = self.current_pos[0] * self.cell_size + self.cell_size // 2
        agent_radius = self.cell_size // 3
        pygame.draw.circle(canvas, (0, 0, 255), (agent_center_x, agent_center_y), agent_radius)

    def close(self):
        if self.screen is not None: # Screen is only created in _init_pygame for human mode
            pygame.display.quit() # Quit the display module if it was initialized
        
        if pygame.get_init(): # If any part of Pygame was initialized (by _init_pygame or directly for rgb_array)
            pygame.quit()     # Uninitialize all Pygame modules
        
        self.screen = None # Ensure these are cleared
        self.clock = None