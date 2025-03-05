# trainer.py
class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def train_episode(self):
        # Note: Gymnasium's reset returns (observation, info)
        obs, _ = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            # Render the environment (this will update the "human" window)
            self.env.render()
            action = self.agent.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Episode ends if terminated or truncated
            if terminated or truncated:
                done = True

        return total_reward
