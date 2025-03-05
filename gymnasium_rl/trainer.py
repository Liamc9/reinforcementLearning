# trainer.py
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent(model, timesteps=10000):
    """
    Train the agent for a specified number of timesteps and save the model.
    """
    model.learn(total_timesteps=timesteps)
    model.save("dqn_blackjack_model")
    return model

def evaluate_agent(model, env, n_eval_episodes=10):
    """
    Evaluate the trained agent over a given number of episodes.
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Evaluation: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

def run_agent(model, env, num_episodes=10):
    """
    Run the trained agent in the environment for a specified number of episodes,
    printing the cumulative reward for each episode.
    """
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {episode}: Reward = {total_reward}")
