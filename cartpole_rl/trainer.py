# trainer.py
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent(model, timesteps=10000):
    """
    Train the agent for the specified number of timesteps and save the model.
    """
    model.learn(total_timesteps=timesteps)
    model.save("ppo_cartpole_model")
    return model

def evaluate_agent(model, env, n_eval_episodes=10):
    """
    Evaluate the agent over a given number of episodes and print the mean reward.
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Evaluation: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

def run_agent(model, env, num_steps=1000):
    """
    Run the trained agent in the environment for a specified number of steps.
    """
    obs, _ = env.reset()
    for _ in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
