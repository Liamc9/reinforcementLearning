# trainer.py
import pygame
import sys
import config

class Trainer:
    def __init__(self, agent, env, num_episodes=config.NUM_EPISODES):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
    
    def train(self, screen, clock):
        print("Training TSP RL Agent...")
        for episode in range(self.num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < config.MAX_STEPS + 1:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                action = self.agent.choose_action(state)
                if action is None:
                    break
                
                next_state, reward, done = self.env.step(state, action)
                self.agent.update_q(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1
            
            print(f"Episode {episode+1}/{self.num_episodes}, Total Reward: {total_reward:.2f}")
            # Optionally render the environment every 100 episodes to see progress
            if (episode + 1) % 100 == 0:
                self.env.render(screen)
                pygame.time.wait(500)
