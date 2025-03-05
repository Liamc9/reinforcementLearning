from trainer import train_agent
from trackmania_pygame_env import TrackmaniaPygameEnv
from agent import DQNAgent
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    agent_config = load_config("config/agent_config.yaml")
    env_config = load_config("config/env_config.yaml")

    env = TrackmaniaPygameEnv(env_config)
    agent = DQNAgent(env.observation_space.shape[0] if hasattr(env, "observation_space") else 4, 4, agent_config)
    train_agent(env, agent, env_config)
