# main.py

import gym
import pygame
import sys
import config
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from tsp_gym_env import TSPGymEnv

def main():
    # Create the Gym environment for TSP
    env = TSPGymEnv()
    
    # (Optional) Check if the environment follows Gym API
    check_env(env, warn=True)
    
    # Create the PPO model using an MLP policy
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Train the model; you can adjust total_timesteps as needed.
    model.learn(total_timesteps=10000)
    
    # Demonstration phase
    obs = env.reset()
    done = False

    # Initialize pygame display for rendering
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("TSP Stable Baselines3 Demo")
    clock = pygame.time.Clock()
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Predict the next action using the trained model
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        clock.tick(config.CLOCK_TICK_DEMO)
    
    print("Route complete! Press the close button to exit.")
    # Keep the window open until closed by the user
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == '__main__':
    main()
