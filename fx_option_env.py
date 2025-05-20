import gymnasium as gym
import numpy as np
import pandas as pd

class FxOptionEnv(gym.Env):
    def __init__(self, price_data, contract_properties, initial_cash):
        super(FxOptionEnv, self).__init__()

        self.price_data = price_data
        self.contract_properties = contract_properties
        self.initial_cash = initial_cash

        self.current_step = 0
        self.cash_balance = initial_cash
        self.portfolio_value = initial_cash
        self.position = 0  # Number of contracts held (positive for long, negative for short)
        self.average_buy_price = 0  # For PnL calculation

        # Define your observation and action spaces here
        # Example:
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10 + 2 + 2,), dtype=np.float32) # Example state size
        self.action_space = gym.spaces.Discrete(3) # BUY, SELL, HOLD

        # Assume transaction costs are a fixed percentage for simplicity
        self.transaction_cost_rate = 0.001

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash_balance = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.position = 0
        self.portfolio_history = [self.portfolio_value] # Track portfolio value for drawdown calculation
        self.average_buy_price = 0
        # Generate initial observation
        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        # This is a placeholder; replace with your actual feature calculation
        # based on self.price_data and self.current_step
        recent_log_returns = np.zeros(10) # Dummy
        technical_indicators = np.zeros(2) # Dummy RSI and MACD
        leverage = (self.portfolio_value - self.cash_balance) / max(self.cash_balance, 1) if self.cash_balance > 0 else 0 # Simple leverage calculation
        observation = np.concatenate([recent_log_returns, technical_indicators, [leverage, self.cash_balance]])
        return observation

    def step(self, action):
        reward = 0

        current_price = self.price_data['price'].iloc[self.current_step]
        next_price = self.price_data['price'].iloc[self.current_step + 1] if self.current_step + 1 < len(self.price_data) else current_price

        # Calculate average margin rate from contract_properties
        average_margin_rate = self.contract_properties['margin_rate'].mean()
        margin_per_contract = next_price * average_margin_rate # Assuming margin is based on next day's price
        transaction_cost = 0

        if action == 0:  # BUY
            # Assume buying 1 contract for simplicity in margin calculation
            contracts_to_buy = 1 # Example: Can be extended to handle different sizes based on action space
            total_margin_required = contracts_to_buy * margin_per_contract
            transaction_cost = contracts_to_buy * current_price * self.transaction_cost_rate

            if self.cash_balance >= total_margin_required + transaction_cost:
                self.cash_balance -= total_margin_required + transaction_cost
                # Update average buy price for PnL tracking
                if self.position > 0:
                    self.average_buy_price = (self.average_buy_price * self.position + current_price * contracts_to_buy) / (self.position + contracts_to_buy)
                else:
                    self.average_buy_price = current_price
                self.position += contracts_to_buy
            else:
                pass # Insufficient funds, no action taken

        elif action == 1:  # SELL
            # Assume selling 1 contract if holding any
            if self.position > 0:
                contracts_to_sell = 1 # Example
                transaction_cost = contracts_to_sell * current_price * self.transaction_cost_rate
                self.cash_balance += contracts_to_sell * margin_per_contract # Release margin
                self.cash_balance -= transaction_cost

                # Calculate PnL for sold contracts and update cash
                pnl = (current_price - self.average_buy_price) * contracts_to_sell
                # PnL is reflected in portfolio value update

                self.position -= contracts_to_sell
                # Re-calculate average buy price if still holding positions
                if self.position > 0:
                     self.average_buy_price = (self.average_buy_price * (self.position + contracts_to_sell) - current_price * contracts_to_sell) / self.position
                else:
                     self.average_buy_price = 0 # Reset if no position

            elif self.position < 0: # Selling to close a short position (not fully implemented here)
                 pass # Add logic for covering short positions

            else:
                # No position to sell
                pass # No action, no penalty

        elif action == 2:  # HOLD
            pass # No action, no transaction cost

        # Update portfolio value: cash + value of position
        self.portfolio_value = self.cash_balance + self.position * next_price

        self.portfolio_history.append(self.portfolio_value)

        # --- Reward Calculation ---
        # Reward = PnL - Transaction Costs - Drawdown Penalty

        # PnL for the step (change in portfolio value considering potential trades)
        # This needs to accurately reflect the profit/loss from held positions + any PnL realized from selling
        # A simpler approach for portfolio-based RL is to use the change in portfolio value
        # adjusted for transaction costs.

        # For this simplified environment, let's calculate PnL from price change on existing position
        daily_pnl = self.position * (next_price - current_price) if self.current_step < len(self.price_data) - 1 else 0

        # Total PnL for the step is daily PnL minus transaction cost
        step_reward = daily_pnl - transaction_cost

        # Add drawdown penalty
        # Find the peak value in the portfolio history
        max_portfolio_value = max(self.portfolio_history)

        # Calculate the current drawdown
        current_drawdown = (max_portfolio_value - self.portfolio_value) / max_portfolio_value if max_portfolio_value > 0 else 0

        # Apply a penalty for the drawdown
        drawdown_penalty_weight = 100  # Hyperparameter to tune
        drawdown_penalty = -drawdown_penalty_weight * current_drawdown

        self.current_step += 1
        if self.current_step >= len(self.price_data) - 1:
            done = True

        observation = self._get_observation()
        info = {} # Add additional info like current PnL, drawdown etc.
        reward = step_reward + drawdown_penalty
        done = self.current_step >= len(self.price_data) - 1 # Ensure done is set correctly
        return observation, reward, done, False, info

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

    def close(self):
        # Clean up resources if needed
        pass

# Example usage (requires sample price_data and contract_properties DataFrames)
# price_data_dummy = pd.DataFrame({'price': np.linspace(100, 110, 252)})
# contract_properties_dummy = pd.DataFrame({
#     'type': ['call', 'put', 'futures'],
#     'expiration': [pd.Timestamp('2025-01-01')] * 3,
#     'strike': [105, 95, None],
#     'margin_rate': [0.1, 0.1, 0.1]
# })
# initial_cash_dummy = 100000
# env = FxOptionEnv(price_data_dummy, contract_properties_dummy, initial_cash_dummy)

# observation, info = env.reset()
# print("Initial Observation:", observation)

# # Example step (BUY action)
# next_observation, reward, done, truncated, info = env.step(0)
# print("Next Observation:", next_observation)
# print("Reward:", reward)
# print("Done:", done)
# print("Cash Balance:", env.cash_balance)
# print("Portfolio Value:", env.portfolio_value)
# print("Position:", env.position)