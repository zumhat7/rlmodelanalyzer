# evaluate_model.py

import numpy as np
import pandas as pd
import gym
from gym import spaces
import talib
from stable_baselines3 import PPO

# Import your FxOptionEnv class from fx_option_env.py
from fx_option_env import FxOptionEnv

# --- Create a separate test environment ---
# It's important to evaluate the model on unseen data.
# We'll create a test environment with a different number of days for variety.
test_n_days = 150  # Shorter or different period for testing
test_env = FxOptionEnv(n_days=test_n_days)

# --- Load the trained model ---
# Specify the path where you saved the model in the training script.
model_path = "ppo_fx_option_trading_model"

try:
    # Load the trained model. The environment needs to be passed to the load method.
    model = PPO.load(model_path, env=test_env)
    print(f"Trained model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Trained model not found at {model_path}")
    print("Please ensure you have run the training script (train_model.py) first to save the model.")
    exit() # Exit if the model file is not found

# --- Run simulation using the loaded model ---
print("Running simulation on the test environment...")

obs = test_env.reset()  # Reset the test environment
done = False
episode_rewards = []
portfolio_values = [test_env.portfolio_value] # Track portfolio value

while not done:
    # Predict the action using the trained model.
    # deterministic=True selects the action with the highest probability.
    action, _states = model.predict(obs, deterministic=True)

    # Execute the action in the test environment
    obs, reward, done, info = test_env.step(action)

    # Record the reward and portfolio value for analysis
    episode_rewards.append(reward)
    portfolio_values.append(test_env.portfolio_value)

print("Simulation finished.")

# --- Calculate Evaluation Metrics ---

# Convert portfolio values to a numpy array for easier calculations
portfolio_values = np.array(portfolio_values)

# Calculate daily returns
# Avoid division by zero if initial portfolio value is 0
returns = np.diff(portfolio_values) / portfolio_values[:-1]
# Handle potential division by zero if a previous value was zero
returns[np.isnan(returns)] = 0
returns[np.isinf(returns)] = 0

# Sharpe Ratio
# Assuming risk-free rate is 0 for simplicity.
# Annualize if needed (multiply mean_return by sqrt(annualization_factor) and std_return by sqrt(annualization_factor))
# For daily data, annualization_factor = 252 (trading days in a year)
mean_return = np.mean(returns)
std_return = np.std(returns)

# Avoid division by zero in Sharpe Ratio calculation
sharpe_ratio = mean_return / std_return if std_return != 0 else 0

# Maximum Drawdown
# Calculate the peak value seen so far
peak_value = np.maximum.accumulate(portfolio_values)
# Calculate the drawdown at each step
drawdowns = (peak_value - portfolio_values) / peak_value
# The maximum drawdown is the largest value in the drawdowns array
max_drawdown = np.max(drawdowns)

# Total PnL
total_pnl = portfolio_values[-1] - portfolio_values[0]

# --- Report Metrics ---

print("\n--- Simulation Results ---")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")

# --- Optional: Plot Portfolio Value ---
# You can uncomment this section to visualize the portfolio performance.
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(portfolio_values)
# plt.title("Portfolio Value Over Time (Simulation)")
# plt.xlabel("Time Steps")
# plt.ylabel("Portfolio Value")
# plt.grid(True)
# plt.show()