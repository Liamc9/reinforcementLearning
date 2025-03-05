# agent.py
import numpy as np
import random
import config

class Agent:
    def __init__(self, maze, epsilon=config.EPSILON, alpha=config.ALPHA, gamma=config.GAMMA):
        self.maze = maze
        self.epsilon = epsilon      # Exploration rate
        self.alpha = alpha          # Learning rate
        self.gamma = gamma          # Discount factor
        self.start_pos = (1, 1)     # Ensure this is a free space in your maze
        self.position = self.start_pos
        # Q-table: dimensions = (rows, cols, number of actions)
        self.q_table = np.zeros((maze.rows, maze.cols, 4))
    
    def choose_action(self, state):
        r, c = state
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return int(np.argmax(self.q_table[r, c]))
    
    def update_q(self, state, action, reward, next_state):
        r, c = state
        next_r, next_c = next_state
        best_next = np.max(self.q_table[next_r, next_c])
        current_q = self.q_table[r, c, action]
        # Q-learning update rule
        self.q_table[r, c, action] = current_q + self.alpha * (reward + self.gamma * best_next - current_q)
    
    def reset(self):
        self.position = self.start_pos
