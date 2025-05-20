import numpy as np
import pandas as pd
import gym
from gym import spaces
import talib
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Import the custom environment from the local file
try:
    from fx_option_env import FxOptionEnv
except ImportError:
    print("Error: fx_option_env.py not found. Please make sure it's in the same directory.")
    # Exit or handle the error appropriately
    exit()


# --- RL Algorithm and Training ---

# Create the environment
# We use make_vec_env to create a vectorized environment, which can speed up training
# n_envs=1 for a single environment instance
env = make_vec_env(lambda: FxOptionEnv(), n_envs=1)

# Define the PPO model
# We'll use a MultiLayer Perceptron (MLP) policy, which is suitable for state vectors
# verbose=1 provides training information
model = PPO("MlpPolicy", env, verbose=1)

# Define hyperparameters for training
# These are example values and will likely need tuning for optimal performance
learning_rate = 0.0003  # Learning rate for the optimizer
n_steps = 2048        # Number of steps to run for each environment per update (batch size for collecting data)
batch_size = 64         # Minibatch size for the PPO optimization
n_epochs = 10           # Number of epochs when optimizing the surrogate loss function
gamma = 0.99          # Discount factor for future rewards
gae_lambda = 0.95     # Factor for trade-off between bias and variance for Generalized Advantage Estimator
clip_range = 0.2      # Clipping parameter, it is a function of the current and the old policy

# Update the model with the chosen hyperparameters
# This is done automatically if you pass them during model initialization,
# but explicitly setting them here can be useful for clarity or later modification.
# model.learning_rate = learning_rate # Hyperparameters can be passed directly to the constructor
# model.n_steps = n_steps
# model.batch_size = batch_size
# model.n_epochs = n_epochs
# model.gamma = gamma
# model.gae_lambda = gae_lambda
# model.clip_range = clip_range

# Training loop
# total_timesteps is the total number of environmental steps to train for
timesteps_to_train = 100000

print(f"Training the PPO model for {timesteps_to_train} timesteps...")
# The learn method handles the entire training process:
# collecting data, calculating advantages, updating the policy and value function
model.learn(total_timesteps=timesteps_to_train)
print("Training finished.")

# Save the trained model (optional)
# This allows you to load and use the model later without retraining
# model.save("fx_option_trading_model")

# To load a trained model:
# model = PPO.load("fx_option_trading_model", env=env)