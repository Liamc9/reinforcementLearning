# tsp_env.py
import math
import pygame
import config

class TSPEnv:
    def __init__(self, cities=config.CITIES):
        self.cities = cities
        self.num_cities = config.NUM_CITIES
        self.reset()
    
    def reset(self):
        # Start at city 0; visited_mask with bit 0 set indicates city 0 is visited.
        self.current_city = 0
        self.visited_mask = 1  # (binary 000...001)
        self.route = [0]       # Record the route taken
        return (self.current_city, self.visited_mask)
    
    def step(self, state, action):
        """
        Executes an action (choosing the next city).
        - If the chosen city is already visited, a heavy penalty is applied.
        - Otherwise, the reward is the negative Euclidean distance between cities.
        - When all cities have been visited, an extra cost is added for returning to the start.
        """
        current_city, visited_mask = state
        
        # Check if the city has already been visited
        if visited_mask & (1 << action):
            return state, -100, False  # Invalid move
        
        # Calculate distance from current city to the selected city
        dist = self._distance(self.cities[current_city], self.cities[action])
        reward = -dist
        
        # Update the visited mask and route
        new_visited_mask = visited_mask | (1 << action)
        new_state = (action, new_visited_mask)
        self.route.append(action)
        
        # Check if all cities have been visited
        if new_visited_mask == (1 << self.num_cities) - 1:
            # Add cost for returning to the starting city (city 0)
            return_dist = self._distance(self.cities[action], self.cities[0])
            reward += -return_dist
            done = True
            new_state = None  # Terminal state
            self.route.append(0)  # Complete the tour by returning to start
        else:
            done = False
        
        return new_state, reward, done
    
    def _distance(self, city1, city2):
        x1, y1 = city1
        x2, y2 = city2
        return math.hypot(x2 - x1, y2 - y1)
    
    def render(self, screen):
        # Clear screen
        screen.fill((255, 255, 255))
        
        # Draw each city as a blue circle with its index
        for idx, (x, y) in enumerate(self.cities):
            pygame.draw.circle(screen, (0, 0, 255), (x, y), 8)
            font = pygame.font.SysFont(None, 24)
            text = font.render(str(idx), True, (0, 0, 0))
            screen.blit(text, (x - 10, y - 10))
        
        # Draw the route taken so far (if any)
        if len(self.route) > 1:
            points = [self.cities[i] for i in self.route]
            pygame.draw.lines(screen, (255, 0, 0), False, points, 2)
        
        pygame.display.flip()
