from stable_baselines3 import PPO

def create_agent(env):
    """
    Create and return a PPO agent using the MlpPolicy with the given environment.
    """
    model = PPO("MlpPolicy", env, verbose=1)
    return model
