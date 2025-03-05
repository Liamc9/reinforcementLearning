# env.py
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

def create_env(render_mode="human"):
    """
    Create and return the Blackjack-v1 environment with observations flattened.
    This wrapper converts the tuple observation space into a flat Box space.
    """
    # Create the Blackjack environment
    env = gym.make("Blackjack-v1", render_mode=render_mode)
    # Flatten the tuple observation into a 1D array
    env = FlattenObservation(env)
    return env
