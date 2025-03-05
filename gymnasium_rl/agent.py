# agent.py
from stable_baselines3 import DQN

def create_agent(env):
    """
    Create and return a DQN agent using the MlpPolicy for the given environment.
    DQN works well for discrete action spaces like those in Blackjack.
    """
    model = DQN("MlpPolicy", env, verbose=1)
    return model
