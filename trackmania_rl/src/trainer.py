import os
import numpy as np
import math
from trackmania_pygame_env import TrackmaniaPygameEnv
from agent import DQNAgent
import yaml

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_agent(env, agent, config):
    episodes = config.get('episodes', 100)
    max_steps = config.get('max_steps', 200)
    batch_size = config.get('batch_size', 32)
    checkpoint_path = config.get('checkpoint_path', "models/checkpoint_latest.pth")

    # Try to resume training from a saved checkpoint.
    if os.path.exists(checkpoint_path):
        try:
            agent.load_checkpoint(checkpoint_path)
            print("Checkpoint loaded. Resuming training...")
        except Exception as e:
            print("Failed to load checkpoint, starting fresh.", e)
    else:
        print("No checkpoint found. Starting training from scratch.")

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            env.render()  # Visualize the current state
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

            # Perform replay training
            agent.replay(batch_size)

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

        # Example of curriculum learning: gradually make the environment more challenging.
        # (In this example, we reduce the off_track_threshold after 50 episodes to force better precision.)
        if e == 50:
            print("Increasing environment difficulty via curriculum learning.")
            # This change should be reflected in your environment's logic if it uses off_track_threshold.
            env.off_track_threshold = max(20, env.off_track_threshold - 10)

        # Save a checkpoint every 10 episodes.
        if (e + 1) % 10 == 0:
            agent.save_checkpoint(checkpoint_path)

def evaluate_agent(env, agent, episodes=5):
    """
    Run several evaluation episodes where the agent uses its greedy policy.
    """
    # Save the current exploration rate and set epsilon to 0 to use the optimal policy.
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()  # Visualize the optimal route.
            action = agent.act(state)  # With epsilon=0, agent acts greedily.
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
        print(f"Evaluation Episode {e+1}: Total Reward = {total_reward}")
    # Restore the original exploration rate.
    agent.epsilon = original_epsilon

    env.close()

if __name__ == "__main__":
    agent_config = load_config("config/agent_config.yaml")
    env_config = load_config("config/env_config.yaml")
    env = TrackmaniaPygameEnv(env_config)
    
    obs_size = env.observation_space.shape[0] if hasattr(env, "observation_space") else 4
    action_size = env.action_space.n if hasattr(env, "action_space") else 4
    agent = DQNAgent(obs_size, action_size, agent_config)
    
    train_agent(env, agent, env_config)
    
    # After training, run evaluation episodes to see the optimal route.
    evaluate_agent(env, agent, episodes=5)

