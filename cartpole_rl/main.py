import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Create the environment with the desired render mode.
    # 'human' mode will open a window to visualize the environment.
    env = gym.make("CartPole-v1", render_mode="human")

    # Create the PPO model using a multi-layer perceptron policy.
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the model for 10,000 timesteps.
    model.learn(total_timesteps=10000)

    # Save the model for later use
    model.save("ppo_cartpole_model")

    # Evaluate the trained agent over 10 episodes.
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")

    # Optionally, run the trained agent in the environment.
    obs, _ = env.reset()
    for _ in range(1000):
        # Get the action from the trained model (deterministic)
        action, _states = model.predict(obs, deterministic=True)
        # Step the environment using the action
        obs, reward, terminated, truncated, info = env.step(action)
        # If the episode ends, reset the environment
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

if __name__ == '__main__':
    main()
