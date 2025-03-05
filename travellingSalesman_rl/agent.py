# agent.py
import numpy as np
import random
import config

class Agent:
    def __init__(self, env, epsilon=config.EPSILON, alpha=config.ALPHA, gamma=config.GAMMA):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_cities = config.NUM_CITIES
        # There are 2^(NUM_CITIES) possible visited_mask states.
        self.num_states = 1 << self.num_cities
        # Q-table dimensions: (current_city, visited_mask, action)
        self.q_table = np.zeros((self.num_cities, self.num_states, self.num_cities))
    
    def choose_action(self, state):
        current_city, visited_mask = state
        # Compute list of valid actions (cities not yet visited)
        valid_actions = [a for a in range(self.num_cities) if not (visited_mask & (1 << a))]
        if not valid_actions:
            return None
        
        # Epsilon-greedy action selection among valid actions
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table[current_city, visited_mask, :].copy()
            # Mask invalid actions by setting their Q-values very low
            for a in range(self.num_cities):
                if a not in valid_actions:
                    q_values[a] = -1e9
            return int(np.argmax(q_values))
    
    def update_q(self, state, action, reward, next_state):
        current_city, visited_mask = state
        current_q = self.q_table[current_city, visited_mask, action]
        
        if next_state is None:
            target = reward  # Terminal state
        else:
            next_city, next_visited = next_state
            target = reward + self.gamma * np.max(self.q_table[next_city, next_visited, :])
        
        self.q_table[current_city, visited_mask, action] = current_q + self.alpha * (target - current_q)
    