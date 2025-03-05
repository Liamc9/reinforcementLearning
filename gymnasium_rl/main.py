# main.py
from env import create_env
from agent import create_agent
from trainer import train_agent, evaluate_agent, run_agent

def main():
    # Create the Blackjack environment (no visualization needed, so render_mode=None)
    env = create_env(render_mode="None")
    
    # Create the agent using the environment
    model = create_agent(env)
    
    # Train the agent for 10,000 timesteps
    model = train_agent(model, timesteps=10000)
    
    # Evaluate the trained agent over 10 episodes
    evaluate_agent(model, env, n_eval_episodes=10)
    
    # Run the trained agent for 10 episodes and print rewards
    run_agent(model, env, num_episodes=10)
    
    env.close()

if __name__ == '__main__':
    main()
