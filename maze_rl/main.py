# main.py
import pygame
import sys
import config
from maze import Maze
from agent import Agent
from trainer import Trainer

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("Complex RL Maze Navigation")
    clock = pygame.time.Clock()

    # Initialize the maze, agent, and trainer
    maze = Maze()
    agent = Agent(maze)
    trainer = Trainer(agent, maze)

    # Run the training loop
    trainer.train(screen, clock)

    # Demonstration: disable exploration and show the learned policy
    print("Now demonstrating the learned policy...")
    agent.epsilon = 0
    agent.reset()
    state = agent.position
    path = [state]
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action = agent.choose_action(state)
        next_state, reward, done = maze.step(state, action)
        state = next_state
        path.append(state)

        screen.fill((0, 0, 0))
        maze.draw(screen, state)
        # Draw the path taken so far (blue outline)
        for pos in path:
            r, c = pos
            rect = pygame.Rect(c * config.CELL_SIZE, r * config.CELL_SIZE,
                               config.CELL_SIZE, config.CELL_SIZE)
            pygame.draw.rect(screen, (0, 0, 255), rect, 3)
        pygame.display.flip()
        clock.tick(config.CLOCK_TICK_DEMO)

    print("Goal reached! Press the close button to exit.")
    # Keep the window open until the user closes it.
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == '__main__':
    main()
