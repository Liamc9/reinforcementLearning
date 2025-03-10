# main.py
from env import create_env
from agent import create_agent
from trainer import train_agent, evaluate_agent, run_agent

def main():
    # Create the environment with real-time visualization
    env = create_env(render_mode="human")
    
    # Create the agent using the environment
    model = create_agent(env)
    
    # Train the agent and save the model
    model = train_agent(model, timesteps=10000)
    
    # Evaluate the trained agent
    evaluate_agent(model, env, n_eval_episodes=10)
    
    # Run the trained agent in the environment
    run_agent(model, env, num_steps=1000)
    
    env.close()

if __name__ == '__main__':
    main()