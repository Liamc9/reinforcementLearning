# env.py
import gymnasium as gym

def create_env(render_mode="human"):
    """
    Create and return the CartPole-v1 environment with the specified render mode.
    """
    env = gym.make("CartPole-v1", render_mode=render_mode)
    return env
