# tsp_gym_env.py

import gymnasium as gym
from gymnasium import spaces
import math
import pygame
import config

class TSPGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.cities = config.CITIES
        self.num_cities = config.NUM_CITIES

        # Action space: choose one of the cities
        self.action_space = spaces.Discrete(self.num_cities)
        
        # Observation space as a Dict: current city and visited mask
        self.observation_space = spaces.Dict({
            'current_city': spaces.Discrete(self.num_cities),
            'visited_mask': spaces.Discrete(1 << self.num_cities)
        })
        
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_city = 0
        self.visited_mask = 1  # only city 0 is visited (bitmask)
        self.route = [0]
        # Gymnasium's reset returns a tuple: (observation, info)
        return {'current_city': self.current_city, 'visited_mask': self.visited_mask}, {}

    def step(self, action):
        # Check for invalid move: city already visited
        if self.visited_mask & (1 << action):
            reward = -100  # Heavy penalty for invalid move
            done = False
            obs = {'current_city': self.current_city, 'visited_mask': self.visited_mask}
            # Gymnasium's step returns: obs, reward, terminated, truncated, info
            return obs, reward, done, False, {}
        
        # Calculate Euclidean distance as negative reward
        x1, y1 = self.cities[self.current_city]
        x2, y2 = self.cities[action]
        dist = math.hypot(x2 - x1, y2 - y1)
        reward = -dist

        # Update state: mark action city as visited and move the current city
        self.visited_mask |= (1 << action)
        self.route.append(action)
        self.current_city = action

        # Check if all cities have been visited
        if self.visited_mask == (1 << self.num_cities) - 1:
            # Add penalty for returning to start
            x_cur, y_cur = self.cities[self.current_city]
            x0, y0 = self.cities[0]
            return_dist = math.hypot(x_cur - x0, y_cur - y0)
            reward += -return_dist
            done = True
        else:
            done = False

        obs = {'current_city': self.current_city, 'visited_mask': self.visited_mask}
        # Return truncated as False (you can modify if needed)
        return obs, reward, done, False, {}

    def render(self, mode='human'):
        # Use pygame for rendering
        screen = pygame.display.get_surface()
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        screen.fill((255, 255, 255))
        
        # Draw cities as blue circles with labels
        for idx, (x, y) in enumerate(self.cities):
            pygame.draw.circle(screen, (0, 0, 255), (x, y), 8)
            font = pygame.font.SysFont(None, 24)
            text = font.render(str(idx), True, (0, 0, 0))
            screen.blit(text, (x - 10, y - 10))
        
        # Draw the route taken so far
        if len(self.route) > 1:
            points = [self.cities[i] for i in self.route]
            pygame.draw.lines(screen, (255, 0, 0), False, points, 2)
        
        pygame.display.flip()

    def close(self):
        pygame.quit()
