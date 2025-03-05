# config.py
import random

# Set random seed for reproducibility
random.seed(42)

# TSP configuration
NUM_CITIES = 20
WIDTH = 800
HEIGHT = 800

# Generate city coordinates randomly within given bounds (with margins)
CITIES = [(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)) for _ in range(NUM_CITIES)]

# Training configuration
NUM_EPISODES = 10000
# In TSP, the maximum number of moves (steps) is NUM_CITIES - 1 (since the start is fixed)
MAX_STEPS = NUM_CITIES - 1

# RL hyperparameters
EPSILON = 0.1  # Exploration rate
ALPHA = 0.5    # Learning rate
GAMMA = 0.9    # Discount factor

# Visualization parameters
CLOCK_TICK_TRAINING = 60  # Speed during training visualization (if used)
CLOCK_TICK_DEMO = 2       # Slower speed during demonstration
