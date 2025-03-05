# maze.py
import pygame
import config

class Maze:
    def __init__(self, grid=config.MAZE_GRID, cell_size=config.CELL_SIZE):
        self.grid = grid
        self.cell_size = cell_size
        self.rows = len(grid)
        self.cols = len(grid[0])
    
    def draw(self, screen, agent_pos):
        # Draw maze cells
        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                   self.cell_size, self.cell_size)
                if self.grid[r][c] == 1:
                    pygame.draw.rect(screen, (0, 0, 0), rect)  # Wall: black
                elif self.grid[r][c] == 0:
                    pygame.draw.rect(screen, (255, 255, 255), rect)  # Free space: white
                elif self.grid[r][c] == 3:
                    pygame.draw.rect(screen, (0, 255, 0), rect)  # Goal: green
        
        # Draw grid lines for clarity
        for r in range(self.rows):
            pygame.draw.line(screen, (200, 200, 200),
                             (0, r * self.cell_size),
                             (self.cols * self.cell_size, r * self.cell_size))
        for c in range(self.cols):
            pygame.draw.line(screen, (200, 200, 200),
                             (c * self.cell_size, 0),
                             (c * self.cell_size, self.rows * self.cell_size))
        
        # Draw the agent as a red square
        agent_rect = pygame.Rect(agent_pos[1] * self.cell_size,
                                 agent_pos[0] * self.cell_size,
                                 self.cell_size, self.cell_size)
        pygame.draw.rect(screen, (255, 0, 0), agent_rect)
    
    def step(self, pos, action):
        """
        Process a step in the maze.
        Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left.
        Returns: (new_position, reward, done)
        """
        r, c = pos
        new_r, new_c = r, c

        if action == 0:  # Up
            new_r = r - 1
        elif action == 1:  # Right
            new_c = c + 1
        elif action == 2:  # Down
            new_r = r + 1
        elif action == 3:  # Left
            new_c = c - 1

        # Check boundaries
        if new_r < 0 or new_r >= self.rows or new_c < 0 or new_c >= self.cols:
            return pos, -1, False

        # Check if hitting a wall
        if self.grid[new_r][new_c] == 1:
            return pos, -1, False

        new_pos = (new_r, new_c)
        if self.grid[new_r][new_c] == 3:
            return new_pos, 10, True  # Reached goal
        else:
            return new_pos, -0.1, False  # Regular step cost
