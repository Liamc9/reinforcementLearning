import random

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        # For demonstration, select a random action.
        return self.action_space.sample()
