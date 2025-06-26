# local_perception_maze.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import collections

class MazeGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, base_maze, num_obstacles, render_mode=None, max_generation_retries=100):
        super(MazeGameEnv, self).__init__()

        self.base_maze = np.array(base_maze, dtype=object)
        self.num_rows, self.num_cols = self.base_maze.shape
        self.num_obstacles_to_place = num_obstacles
        self.max_generation_retries = max_generation_retries
        self.maze = self.base_maze.copy()

        self.start_pos = np.argwhere(self.base_maze == 'S')[0]
        self.goal_pos = np.argwhere(self.base_maze == 'G')[0]
        self.current_pos = self.start_pos.copy()
        self.previous_distance_to_goal = None

        # --- Define our local perception encoding ---
        # 0: Empty/Start, 1: Wall/Boundary, 2: Danger, 3: Goal
        self.perception_encoding = {'.': 0, 'S': 0, '#': 1, 'D': 2, 'G': 3}
        max_perception_val = max(self.perception_encoding.values())

        self.action_space = spaces.Discrete(4)

        ### CHANGE 1: Define the new "Local Perception" Observation Space ###
        # [agent_r, agent_c, goal_r, goal_c, val_up, val_down, val_left, val_right]
        obs_shape = 8
        low_bounds = np.zeros(obs_shape, dtype=np.float32)
        high_bounds = np.array([
            self.num_rows - 1, self.num_cols - 1,
            self.num_rows - 1, self.num_cols - 1,
            max_perception_val, max_perception_val,
            max_perception_val, max_perception_val
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, shape=(obs_shape,), dtype=np.float32
        )

        # Pygame setup (unchanged)
        self.cell_size = 50
        self.screen_width = self.num_cols * self.cell_size
        self.screen_height = self.num_rows * self.cell_size
        self.screen, self.clock, self.render_mode = None, None, render_mode
        if self.render_mode == "human": self._init_pygame()


    def _get_cell_value(self, r, c):
        """Helper to get the encoded value of a cell, handling boundaries."""
        if not (0 <= r < self.num_rows and 0 <= c < self.num_cols):
            return self.perception_encoding['#'] # Treat out-of-bounds as a wall
        return self.perception_encoding.get(self.maze[r, c], 0)

    def _get_obs(self):
        ### CHANGE 2: Construct the new observation vector ###
        r, c = self.current_pos
        
        # Get the encoded value for each of the 4 adjacent cells
        val_up = self._get_cell_value(r - 1, c)
        val_down = self._get_cell_value(r + 1, c)
        val_left = self._get_cell_value(r, c - 1)
        val_right = self._get_cell_value(r, c + 1)

        # Concatenate all features into a single vector
        obs = np.array([
            self.current_pos[0], self.current_pos[1],
            self.goal_pos[0], self.goal_pos[1],
            val_up, val_down, val_left, val_right
        ], dtype=np.float32)
        return obs
    
    # The rest of the code (reset, step, rendering, maze generation) is the same as before.
    # The reward shaping is still very useful.
    # I am pasting it below for completeness.
    
    def _get_manhattan_distance(self):
        return np.abs(self.current_pos[0] - self.goal_pos[0]) + np.abs(self.current_pos[1] - self.goal_pos[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_maze()
        self.current_pos = self.start_pos.copy()
        self.previous_distance_to_goal = self._get_manhattan_distance()
        return self._get_obs(), {}

    def step(self, action):
        new_pos = self.current_pos.copy()
        if action == 0: new_pos[0] -= 1
        elif action == 1: new_pos[0] += 1
        elif action == 2: new_pos[1] -= 1
        elif action == 3: new_pos[1] += 1

        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
        
        cell_type = self.maze[self.current_pos[0], self.current_pos[1]]
        terminated = False
        if cell_type == 'G':
            reward = 10.0
            terminated = True
        elif cell_type == 'D':
            reward = -5.0
            terminated = True
        else:
            current_distance = self._get_manhattan_distance()
            reward = (self.previous_distance_to_goal - current_distance) * 0.1
            self.previous_distance_to_goal = current_distance
            reward -= 0.02

        return self._get_obs(), reward, terminated, False, {}

    def _is_valid_position(self, pos):
        row, col = pos
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.maze[row, col] != '#'

    def _generate_maze(self):
        if not hasattr(self, 'np_random') or self.np_random is None: self.np_random, _ = gym.utils.seeding.np_random()
        possible_coords = [tuple(c) for c in np.argwhere(self.base_maze == '.')]
        for _ in range(self.max_generation_retries):
            self.maze = self.base_maze.copy()
            indices = self.np_random.choice(len(possible_coords), size=self.num_obstacles_to_place, replace=False)
            for i in indices: self.maze[possible_coords[i]] = '#'
            if self._is_path_solvable(): return
        raise RuntimeError("Failed to generate a solvable maze.")

    def _is_path_solvable(self):
        q = collections.deque([tuple(self.start_pos)])
        visited = {tuple(self.start_pos)}
        while q:
            r, c = q.popleft()
            if r == self.goal_pos[0] and c == self.goal_pos[1]: return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.num_rows and 0 <= nc < self.num_cols and self.maze[nr, nc] != '#' and (nr, nc) not in visited:
                    visited.add((nr, nc)); q.append((nr, nc))
        return False

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None: self._init_pygame()
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((255, 255, 255))
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = (220, 220, 220)
                if self.maze[r, c] == '#': color = (50, 50, 50)
                elif self.maze[r, c] == 'S': color = (0, 255, 0)
                elif self.maze[r, c] == 'G': color = (255, 0, 0)
                elif self.maze[r, c] == 'D': color = (128, 0, 128)
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (100, 100, 100), rect, 1)
        agent_center = (self.current_pos[1] * self.cell_size + self.cell_size // 2, self.current_pos[0] * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(canvas, (0, 0, 255), agent_center, self.cell_size // 3)
        self.screen.blit(canvas, (0, 0))
        pygame.event.pump(); pygame.display.flip(); self.clock.tick(self.metadata["render_fps"])

    def _init_pygame(self):
        pygame.init(); self.screen = pygame.display.set_mode((self.screen_width, self.screen_height)); pygame.display.set_caption("Local Perception Maze"); self.clock = pygame.time.Clock()

    def close(self):
        if self.screen: pygame.display.quit(); pygame.quit(); self.screen = None