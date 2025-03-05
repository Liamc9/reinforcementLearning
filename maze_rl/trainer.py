# trainer.py
import pygame
import sys
import config

class Trainer:
    def __init__(self, agent, maze,
                 num_episodes=config.NUM_EPISODES,
                 max_steps=config.MAX_STEPS,
                 clock_tick=config.CLOCK_TICK_TRAINING):
        self.agent = agent
        self.maze = maze
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.clock_tick = clock_tick
    
    def train(self, screen, clock):
        print("Training...")
        for episode in range(self.num_episodes):
            self.agent.reset()
            state = self.agent.position
            done = False
            step = 0
            while not done and step < self.max_steps:
                # Process Pygame events to allow window closure
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                action = self.agent.choose_action(state)
                next_state, reward, done = self.maze.step(state, action)
                self.agent.update_q(state, action, reward, next_state)
                state = next_state
                self.agent.position = state

                # Optional visualization during training
                screen.fill((0, 0, 0))
                self.maze.draw(screen, self.agent.position)
                pygame.display.flip()
                clock.tick(self.clock_tick)
                step += 1

            print(f"Episode {episode+1}/{self.num_episodes}, steps: {step}")
        print("Training completed.")
