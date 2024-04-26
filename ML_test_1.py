import pandas as pd
import datetime
import MetaTrader5 as mt5
from datetime import datetime
import gymnasium as gym
import pytz
import numpy as np
import torch.nn as nn
from gym import spaces
import pickle
import torch
import pandas_ta as ta
import torch.optim as optim
import os
from dateutil.relativedelta import relativedelta
from datetime import datetime
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import shutil
import re

# Define the ForexTradingEnv class, inheriting from gym.Env to create a custom trading environment
class ForexTradingEnv(gym.Env):
    # Metadata for the environment, specifying available render modes
    metadata = {'render.modes': ['human']}

    # Constructor with parameters for initializing the environment
    def __init__(self, data, num_features=0, window_size=24, initial_balance=10000, leverage=100, transaction_cost=0.0002, stop_loss_percent=0.5, take_profit_percent=1, lot_size=1000, history_size=10):
        # Call the constructor of the superclass
        super(ForexTradingEnv, self).__init__()

        # Initialize environment variables with provided arguments or defaults
        self.data = data  # Data for trading, typically historical price data
        self.initial_balance = initial_balance  # Starting balance for the trading account
        self.current_balance = initial_balance  # Current balance, initialized to the starting balance
        self.max_daily_loss_limit = initial_balance * 0.05  # Set the maximum daily loss limit to 5% of the initial balance
        self.max_loss_limit = initial_balance * 0.1  # Set the overall loss limit to 10% of the initial balance
        self.profit_target = initial_balance * 0.1  # Set a profit target of 10% of the initial balance
        self.balance = initial_balance  # Current balance, repeated initialization for clarity
        self.peak_balance = initial_balance  # Peak balance achieved, initialized to the starting balance
        self.equity = initial_balance  # Equity, initially equal to the starting balance
        self.leverage = leverage  # Leverage available for trading
        self.transaction_cost = transaction_cost  # Cost per transaction
        self.stop_loss_percent = stop_loss_percent  # Stop loss threshold in percentage
        self.take_profit_percent = take_profit_percent  # Take profit threshold in percentage
        self.lot_size = lot_size  # Size of a single trade lot
        self.position = 'neutral'  # Current trading position, initialized as neutral (no open positions)
        self.is_open_position = False  # Flag to indicate if a trading position is open
        self.accumulated_lot_size = 0  # Total size of the open position
        self.entry_price = None  # Price at which the current position was opened
        self.position_open_step = None  # Step number when the current position was opened
        self.sl_price = 0  # Stop loss price
        self.tp_price = 0  # Take profit price
        self.current_step = 0  # Current step in the environment
        self.pnl_history = [0]  # History of profit and loss, starting with 0
        self.trade_profit = 0  # Profit from the current trade, initialized to 0
        self.position_closed_recently = False  # Flag to indicate if a position was closed recently
        self.window_size = window_size  # Size of the window for observation data
        self.num_features = num_features # Number of features in the observation data (7 for no indicators, 25 with indicators)
        self.profit_streak = 0
        self.loss_streak = 0
        self.final_profit = 0
        self.worst_case_pnl = 0
        self.best_case_pnl = 0
        self.daily_pnl = 0  # Initialize daily profit and loss
        self.last_processed_date = None  # Keep track of the last processed date
        self.daily_loss_limit_reached = False
        self.trade_history = []  # History of trades made
        self.history_size = history_size
        self.step_live_bool = False
        # Assume 7 features based on selection: Profit, Position Type, Lot Size, Price at Close, Worst Case PnL, Best Case PnL, Closed Balance
        self.history = np.zeros((self.history_size, 10))

        # Define the action space of the environment
        self.action_space = spaces.Discrete(4)  # 0 - Buy, 1 - Sell, 2 - Close, 3 - Hold

        # Update observation space size: data window + trade history features
        total_features = self.window_size * num_features + self.history_size * 10

        # Define the observation space of the environment
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)  # Observation space defined as a box with values ranging from 0 to infinity, shaped according to the window size and number of features

    def save_state(self, base_filename='ForexTradingEnv_state'):
        # Combine base filename, timestamp, and file extension
        filename = f"{base_filename}.pkl"
        
        # Use the constructed filename to save the file
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_state(self, filename, new_data=None):
        with open(filename, 'rb') as f:
            state_dict = pickle.load(f)
            self.__dict__.update(state_dict)

        # Update the data attribute if new data is provided
        if new_data is not None:
            self.data = new_data
            print("Data has been updated.")

        print("State loaded from file.")

    def update_current_pnl(self, high_price, low_price):
        if self.position == 'long':
            # Worst-case loss for a long position at the lowest price
            self.worst_case_pnl = (low_price - self.entry_price) * self.accumulated_lot_size - self.transaction_cost
            # Best-case profit for a long position at the highest price
            self.best_case_pnl = (high_price - self.entry_price) * self.accumulated_lot_size - self.transaction_cost
        elif self.position == 'short':
            # Worst-case loss for a short position at the highest price
            self.worst_case_pnl = (self.entry_price - high_price) * self.accumulated_lot_size - self.transaction_cost
            # Best-case profit for a short position at the lowest price
            self.best_case_pnl = (self.entry_price - low_price) * self.accumulated_lot_size - self.transaction_cost
        else:
            # If there is no open position, set both PnLs to 0
            self.worst_case_pnl = 0
            self.best_case_pnl = 0

    # Define a method to execute a trade based on the action, current market price
    def execute_trade(self, action, current_price, high_price, low_price):

        # Close the position if the price crosses the stop loss or take profit thresholds
        if self.position == 'long':
            # For a long position, close if the low price drops below the stop loss or the high price exceeds the take profit.
            if low_price <= self.sl_price or high_price >= self.tp_price:
                self.close_position_any(current_price)
        elif self.position == 'short':
            # For a short position, close if the high price exceeds the stop loss or the low price drops below the take profit.
            if high_price >= self.sl_price or low_price <= self.tp_price:
                self.close_position_any(current_price)

        trade_executed = False  # Initialize a flag to track if a trade was executed

        min_lot_size = 1000  # Minimum lot size for a trade
        max_lot_size = 10000  # Maximum lot size for a trade

        self.lot_size = max_lot_size  # or a predetermined static value within the range

        # If the current balance is greater than the peak balance, update the peak balance to the current balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Calculate the accumulated lot size by summing the lot sizes of all open trades from the trade history
        self.accumulated_lot_size = sum([trade.get('lot_size') for trade in self.trade_history if trade.get('is_open', True)])

        # Determine if there is an open position by checking if any trade in the history is marked as open
        self.is_open_position = any(trade.get('is_open', False) for trade in self.trade_history)

        # If the action is 2 (close), call the method to close any open position at the current market price
        if action == 2:
            self.close_position_any(current_price)

        if not self.step_live_bool:  # Simplified condition
            date = self.data.index[self.current_step].strftime('%Y-%m-%d')
        else:
            date = self.data.index[-1].strftime('%Y-%m-%d')

        # If the action is 3 (hold), log the decision to hold without making a trade by appending a record to the trade history
        if action == 3:
            self.trade_history.append({
                'Step': self.current_step,  # Record the current step (time) of the hold action
                'Action': 'Hold',  # Indicate the action taken was to hold
                'Position': self.position,  # Record the current trading position
                'Price': current_price,  # Record the current market price at the time of holding
                'Balance': self.balance,  # Record the current balance of the account
                'Date': date,
                'lot_size': 0,  # Indicate that no lot size is applicable for hold actions
                'worst_case_pnl': self.worst_case_pnl,
                'best_case_pnl': self.best_case_pnl
            })

        self.step_live_bool = False

        # Check if the accumulated lot size exceeds the maximum allowed limit
        if self.accumulated_lot_size >= 100000:
            self.close_position_any(current_price)  # Close any position if the accumulated lot size is too large
            # Allow closing and opening of new positions based on the action and current position
            if self.position != 'neutral':  # Ensure there's an active position
                if self.position == 'long' and action == 1:  # If the action is to sell from a long position
                    trade_executed = True
                    self.close_position_any(current_price)  # Close the long position
                    self.open_position(current_price, 'short', self.lot_size)  # Open a new short position
                elif self.position == 'short' and action == 0:  # If the action is to buy from a short position
                    trade_executed = True
                    self.close_position_any(current_price)  # Close the short position
                    self.open_position(current_price, 'long', self.lot_size)  # Open a new long position

        # Assuming no early return, we proceed to check if new positions can be opened
        # Given no open positions, we can open a new one based on the action
        if not self.is_open_position or self.position == 'neutral':
            if action == 0:  # If the action is to buy
                trade_executed = True  # Mark that a trade was executed
                self.open_position(current_price, 'long', self.lot_size)  # Open a new long position
            elif action == 1:  # If the action is to sell
                trade_executed = True  # Mark that a trade was executed
                self.open_position(current_price, 'short', self.lot_size)  # Open a new short position

        return trade_executed  # Return whether a trade was executed or not

    # Define a method to open a new trading position
    def open_position(self, current_price, position_type, lot_size):
        """Opens a new trading position."""

        # Calculate the cost of opening the position based on transaction cost and lot size
        cost = self.transaction_cost * lot_size

        # Deduct the opening cost from the current balance
        self.current_balance -= cost

        # Calculate and set the margin required for the position
        self.margin = (lot_size / current_price) / self.leverage

        # Set the type of the position (long or short)
        self.position = position_type

        # Record the price at which the position is opened
        self.entry_price = current_price

        # Update the total accumulated lot size with the lot size of the new position
        self.accumulated_lot_size += lot_size

        # Indicate that there is now an open position
        self.is_open_position = True

        # Record the current step as the step at which the position was opened
        self.position_open_step = self.current_step

        # If the position is long, calculate and set the stop loss and take profit prices accordingly
        if position_type == 'long':
            self.sl_price = current_price * (1 - self.stop_loss_percent / 100)  # Calculate stop loss price for a long position
            self.tp_price = current_price * (1 + self.take_profit_percent / 100)  # Calculate take profit price for a long position

        # If the position is short, calculate and set the stop loss and take profit prices accordingly
        elif position_type == 'short':
            self.sl_price = current_price * (1 + self.stop_loss_percent / 100)  # Calculate stop loss price for a short position
            self.tp_price = current_price * (1 - self.take_profit_percent / 100)  # Calculate take profit price for a short position

        # Deduct the cost again from the balance for some reason (seems like an error or redundant step)
        self.balance -= cost

        # Append a new entry to the trade history with details of the open position
        self.trade_history.append({
            'Step': self.current_step,  # Record the current step (time) of the hold action
            'Action': 'Open ' + position_type.capitalize(),  # Record the action taken (Open Long/Open Short)
            'Position': self.position,  # Record the type of the position opened
            'Price': current_price,  # Record the opening price of the position
            'Balance': self.balance,  # Record the balance after opening the position
            'lot_size': lot_size,  # Record the lot size of the position
            'Date': datetime.now().strftime('%Y-%m-%d'),  # Record the current date of the position opening
            'SL Price': self.sl_price,  # Record the calculated stop loss price
            'TP Price': self.tp_price,  # Record the calculated take profit price
            'is_open': self.is_open_position,  # Indicate that this trade entry is currently open
            'worst_case_pnl': self.worst_case_pnl,
            'best_case_pnl': self.best_case_pnl
        })

    # Define a method to close any open trading position at a given price
    def close_position_any(self, current_price):
        # Check if there is an open position to close
        if self.position in ['long', 'short']:
            # Check if the position is a long position
            if self.position == 'long':
                self.position_closed_recently = True  # Mark that a position was recently closed
                # Calculate the profit or loss for the long position
                self.trade_profit = (current_price - self.entry_price) * self.lot_size - self.transaction_cost
            # Check if the position is a short position
            elif self.position == 'short':
                self.position_closed_recently = True  # Mark that a position was recently closed
                # Calculate the profit or loss for the short position
                self.trade_profit = (self.entry_price - current_price) * self.lot_size - self.transaction_cost
            else:
                # This else block seems redundant due to the initial if condition and can be considered unreachable
                self.position_closed_recently = True  # Ensure the flag is set even if neither long nor short (redundant)
                self.trade_profit = 0  # Set profit to 0 as a fallback

            # Iterate through the trade history to find and close the open position
            for trade in self.trade_history:
                # Check if the trade is marked as open
                if trade.get('is_open', False):  # Default to False if 'is_open' key is not found
                    trade['is_open'] = False  # Mark the trade as closed

            self.accumulated_lot_size = 0  # Reset the accumulated lot size to 0 as the position is closed

            # Update the account balance with the profit or loss from closing the position
            self.balance += self.trade_profit

            if not self.step_live_bool:  # Simplified condition
                date = self.data.index[self.current_step].strftime('%Y-%m-%d')
            else:
                date = self.data.index[-1].strftime('%Y-%m-%d')

            # Record the closing of the position in the trade history
            self.trade_history.append({
                'Step': self.current_step,  # Record the step at which the position was closed
                'Action': 'Closed Position',  # Indicate the action taken was closing a position
                'Position': self.position,  # Record the type of the position that was closed
                'Price': current_price,  # Record the price at which the position was closed
                'Profit': self.trade_profit,  # Record the profit or loss from the trade
                'Balance': self.balance,  # Record the updated balance after closing the position
                'Date': date,  # Record the date of the position closing
                'lot_size': self.accumulated_lot_size,  # Record the lot size of the position that was closed
                'Closed Balance': self.balance,  # Record the balance after the position was closed
            })

            self.step_live_bool = False

            # Reset attributes related to the position
            self.position = 'neutral'  # Reset the position status to neutral
            self.entry_price = None  # Clear the entry price
            self.sl_price = 0  # Reset the stop loss price
            self.tp_price = None  # Clear the take profit price
            self.margin = 0  # Reset the margin used
            self.position_open_step = None  # Clear the step at which the position was opened
        else:
            # If there's no open position to close, record this attempt in the trade history
            self.trade_history.append({
                'Step': self.current_step,  # Record the current step (time) of the hold action
                'Action': 'No open position to close',  # Indicate that there was an attempt to close a non-existent position
                'Position': self.position,  # Record the current position status, which should be neutral
                'Price': current_price,  # Record the current price at which the close attempt was made
                'Balance': self.balance,  # Record the current balance
                'lot_size': 0,  # Indicate that no lot size is involved in this action
                'Date': datetime.now().strftime('%Y-%m-%d'),  # Record the date of the attempt
                'SL Price': self.sl_price,  # Record the current stop loss price
                'TP Price': self.tp_price,  # Record the current take profit price
                'is_open': self.is_open_position,  # Indicates that the trade is open
            })

    # Define a method to reset attributes related to any open trading position
    def reset_position_attributes(self):
        # Reset the position to 'neutral', indicating no open positions
        self.position = 'neutral'
        # Clear the entry price as there's no open position
        self.entry_price = None
        # Reset the stop loss and take profit prices to their default values
        self.sl_price = 0
        self.tp_price = 0
        # Reset the accumulated lot size to zero
        self.accumulated_lot_size = 0
        # Reset the margin used for the position to zero
        self.margin = 0
        # Clear the step at which the position was opened
        self.position_open_step = None

    # Define a method to reset the environment to its initial state
    def reset(self):
        # Reset the account balance to the initial balance
        self.balance = self.initial_balance
        # Reset the equity to the initial balance
        self.equity = self.initial_balance
        # Reset the current step to zero
        self.current_step = 0
        # Reset the position status to 'neutral'
        self.position = 'neutral'
        # Clear the entry price
        self.entry_price = None
        # Reset the trade history to an empty list
        self.trade_history = []
        # Reset the profit and loss history, starting with an initial value of 0
        self.pnl_history = [0]
        # Generate the next observation based on the reset state
        observation = self._next_observation()

        assert observation.shape == (self.observation_space.shape[0],), f"Observation shape mismatch: {observation.shape} != {self.observation_space.shape}"
        # Return the observation to the caller, typically for starting a new episode
        return observation
    
    def extract_state(self):
        observation = self._next_observation()
        return observation

    def update_history_from_trade_history(self):
        # Define the action mapping
        action_mapping = {'Open Long': 0, 'Open Short': 1, 'Closed Position': 2, 'Hold': 3, 'No open position to close': 3}
        
        # Initialize an empty list to hold structured trade data
        structured_trade_data = []
        
        # Extract features from each trade dictionary, using the last self.history_size trades
        for trade in self.trade_history[-self.history_size:]:
            action_number = action_mapping.get(trade.get('Action', 'Hold'), 3)  # Default to 'Hold' if action is unknown
            position_type = 1 if trade.get('Position', 'neutral') == 'long' else -1 if trade.get('Position', 'neutral') == 'short' else 0
            is_open_number = 1 if trade.get('is_open', False) else 0

            # Using 'or 0' to ensure None is treated as 0
            sl_price = trade.get('SL Price') or 0
            tp_price = trade.get('TP Price') or 0
            
            structured_trade_data.append([
                float(trade.get('Profit', 0)),
                float(trade.get('Balance', self.initial_balance)),
                float(position_type),
                float(is_open_number),
                float(sl_price),  # Ensure None values are converted to 0 before applying float
                float(tp_price),  # Ensure None values are converted to 0 before applying float
                float(trade.get('lot_size', 0)),
                float(trade.get('Price', 0)),
                float(trade.get('Closed Balance', self.initial_balance)),
                float(action_number)  # Adding the mapped action number   
            ])

        # Convert the list to a NumPy array
        self.history = np.array(structured_trade_data, dtype=np.float32)
        
    def _next_observation(self):
        # Initialize the count of missing rows to 0
        missing_rows = 0

        # Calculate the end index for the observation window, ensuring it doesn't exceed the length of the data
        end_idx = min(self.current_step + self.window_size, len(self.data))

        # Slice the DataFrame from the current step up to the calculated end index
        obs_df = self.data.iloc[self.current_step:end_idx]

        # Check if the obtained slice is smaller than the window size
        if len(obs_df) < self.window_size:
            # Calculate how many rows are missing to reach the window size
            missing_rows = self.window_size - len(obs_df)
            # Create a DataFrame of zeros to pad the observation to the correct size
            padding = pd.DataFrame(np.zeros((missing_rows, self.data.shape[1])), columns=self.data.columns)
            # Concatenate the padding DataFrame to the end of the observation DataFrame
            obs_df = pd.concat([obs_df, padding], ignore_index=True)

        # Flatten the DataFrame into a 1D array for use in the environment
        flattened_data_obs = obs_df.values.flatten()

        # Update and format trade history part
        self.update_history_from_trade_history()
        flattened_history = self.history.flatten()

        # Ensure the flattened history matches the expected number of features
        if len(flattened_history) != self.history_size * 10:
            additional_zeros = np.zeros(self.history_size * 10 - len(flattened_history))
            flattened_history = np.concatenate([flattened_history, additional_zeros])
        
        combined_observation = np.concatenate([flattened_data_obs, flattened_history])

        # Increment the current step
        self.current_step += 1

        # Return the combined observation array
        return combined_observation

    # Define a method to take a step in the environment based on an action
    def step(self, action, trade_history):
        final_reward = 0
        # Reset the flag indicating whether a position was closed recently
        self.position_closed_recently = False

        # Get the current price from the data using the 'close' column
        current_price = self.data['close'].iloc[self.current_step]

        # Get the high price from the data using the 'high' column
        high_price = self.data['high'].iloc[self.current_step]

        low_price = self.data['low'].iloc[self.current_step]

        current_datetime = self.data.index[self.current_step]

        current_date = current_datetime.date()  # Assuming datetime index

        # Check for date change
        if self.last_processed_date is None or self.last_processed_date != current_date:
            # Reset daily P&L if it's a new day
            if self.last_processed_date is not None:  # Ensure it's not the first step
                self.daily_pnl = 0

        # Update last processed date
        self.last_processed_date = current_date

        # Update the current profit and loss (P&L) based on the current market price
        self.update_current_pnl(high_price, low_price)

        worst_balance = self.balance + self.worst_case_pnl

        best_balance = self.balance + self.best_case_pnl

        self.daily_pnl += self.worst_case_pnl

        # Check if the best case scenario achieves the profit target
        if best_balance >= self.profit_target + self.initial_balance:
            done = True
            self.balance = best_balance
            print("Episode terminated: Profit target achieved in best case scenario.")
        # Check if the worst case scenario drops below the allowed loss limit
        elif worst_balance <= self.initial_balance - self.max_loss_limit:
            done = True
            self.balance = worst_balance
            print("Episode terminated: Loss limit breached in worst case scenario.")
        elif self.daily_pnl < -self.max_daily_loss_limit:
            done = True
            self.daily_loss_limit_reached = True
            print("Episode terminated: Daily Loss limit breached in worst case scenario.")
        else:
            # Continue the episode if neither condition is met
            done = self.current_step >= len(self.data) - 5  # Check if there are enough steps left

        # Format the current date as a string
        current_date = self.data.index[self.current_step].strftime('%Y-%m-%d')
        # Execute the trade based on the provided action
        self.execute_trade(action, current_price, high_price, low_price)

        #if done:
            #self.evaluate_performance()
            #final_reward = self.calculate_final_reward() 
            
        # Calculate the reward for the current step
        step_reward = self.calculate_step_reward(action, trade_history, self.current_step)

        reward = step_reward + final_reward

        # Get the next observation after taking the step
        next_observation = self._next_observation()

        # Ensure the shape of the next observation matches the expected observation space shape
        assert next_observation.shape == (self.observation_space.shape[0],), f"Next observation shape mismatch: {next_observation.shape} != {self.observation_space.shape}"
        # Return the next observation, the step reward, the done flag, and an empty info dict
        return next_observation, reward, done, {}

    def step_live(self, action):
        # Access the last row directly with .iloc[-1]
        current_row = self.data.iloc[-1]
        current_price = current_row['close']
        high_price = current_row['high']
        low_price = current_row['low']
        current_datetime = self.data.index[-1]

        self.step_live_bool = True

        print(f'current price: {current_price}, high_price: {high_price},  low_price: {low_price}, current_datetime: {current_datetime}')

        # Reset flags and variables as needed
        self.position_closed_recently = False
        current_date = current_datetime.date()

        # Check for a date change and update daily P&L
        if self.last_processed_date is None or self.last_processed_date != current_date:
            self.daily_pnl = 0 if self.last_processed_date is not None else self.daily_pnl

        # Update last processed date
        self.last_processed_date = current_date

        # Update P&L based on new price data
        self.update_current_pnl(high_price, low_price)
        worst_balance = self.balance + self.worst_case_pnl
        best_balance = self.balance + self.best_case_pnl
        self.daily_pnl += self.worst_case_pnl

        # Evaluate end conditions
        done = False
        if best_balance >= self.profit_target + self.initial_balance:
            done = True
            self.balance = best_balance
            print("Profit target achieved.")
        elif worst_balance <= self.initial_balance - self.max_loss_limit:
            done = True
            self.balance = worst_balance
            print("Loss limit breached.")
        elif self.daily_pnl < -self.max_daily_loss_limit:
            done = True
            print("Daily loss limit breached.")

        # Execute the trade based on the action
        self.execute_trade(action, current_price, high_price, low_price)

        # Format the current date and return state information
        current_date_str = current_datetime.strftime('%Y-%m-%d')

        return self.extract_state(), done

    # Define a method to evaluate and summarize the trading performance
    def evaluate_performance(self):
        # Initialize peak balance, drawdown, and highest daily loss variables
        peak = self.initial_balance  # Set the initial peak to the initial balance
        drawdown = 0  # Initial drawdown value
        highest_daily_loss = 0  # Initial highest daily loss value

        # Create a list to hold daily balances, starting with the initial balance
        daily_balances = [self.initial_balance]

        # Iterate through each trade in the trade history
        for trade in self.trade_history:
            current_balance = trade['Balance']  # Extract the balance after the trade
            daily_balances.append(current_balance)  # Append the balance to the daily balances list

            # Update the peak balance if the current balance exceeds the peak
            if current_balance > peak:
                peak = current_balance

            # Calculate the drawdown as a percentage
            drawdown = max(drawdown, (peak - current_balance) / peak * 100)

            # Calculate daily loss if there are at least two daily balances to compare
            if len(daily_balances) > 1:
                daily_loss = (daily_balances[-2] - daily_balances[-1]) / daily_balances[-2] * 100
                highest_daily_loss = min(highest_daily_loss, daily_loss)  # Update highest daily loss if this loss is greater

        # Convert the highest daily loss into a positive value for reporting
        magnitude_of_highest_daily_loss = abs(highest_daily_loss)
        # Round the magnitude of the highest daily loss to two decimal places
        magnitude_of_highest_daily_loss = round(magnitude_of_highest_daily_loss, 2)

        # Initialize variables to calculate additional metrics
        total_profit = 0  # Sum of all profitable trades
        total_loss = 0  # Sum of all losing trades
        win_count = 0  # Count of profitable trades
        loss_count = 0  # Count of losing trades

        # Iterate through each trade to calculate profits, losses, and counts
        for trade in self.trade_history:
            if 'Profit' in trade:
                if trade['Profit'] > 0:
                    total_profit += trade['Profit']  # Add to total profit
                    win_count += 1  # Increment win count
                else:
                    total_loss += abs(trade['Profit'])  # Add to total loss
                    loss_count += 1  # Increment loss count

        # Calculate the profit factor, win rate, and other metrics
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')  # Profit factor calculation
        number_of_closed_trades = win_count + loss_count  # Total number of trades that were closed
        percent_profitable = (win_count / number_of_closed_trades * 100) if number_of_closed_trades > 0 else 0  # Calculate the percentage of profitable trades

        # Calculate final profit by subtracting the initial balance from the final balance
        self.final_profit = self.balance - self.initial_balance

        # Return a dictionary summarizing the performance metrics
        return {
                "Final Balance": self.balance,  # The final account balance
                "Final Profit": self.final_profit,  # The net profit or loss
                "Maximum Drawdown": f"{drawdown:.2f}%",  # The maximum drawdown in percentage
                "Highest Daily Loss": f"{highest_daily_loss:.2f}%",  # The highest daily loss in percentage
                "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "Inf",  # The profit factor
                "Number of Closed Trades": number_of_closed_trades,  # The total number of trades closed
                "Percent Profitable": f"{percent_profitable:.2f}%",  # The percentage of trades that were profitable
                "Magnitude of loss": magnitude_of_highest_daily_loss  # The magnitude of the highest daily loss
            }

    def calculate_step_reward(self, action, trade_history, step_number):
        # Initialize the reward variable
        reward = 0

        if self.daily_loss_limit_reached:
            reward += -self.max_daily_loss_limit

        if self.balance > self.initial_balance:
            net_profit = self.balance - self.initial_balance
            reward += net_profit * 0.01
        elif self.balance < self.initial_balance:
            net_profit = self.balance - self.initial_balance
            reward += net_profit * 0.01

        # Directly link the reward to the profit or loss
        if self.trade_profit > 0:
            reward += self.trade_profit * 0.01
        elif self.trade_profit < 0:
            reward += self.trade_profit * 0.01

        # Check for hitting profit or loss targets
        if self.balance >= self.profit_target + self.initial_balance:
            net_profit = self.balance - self.initial_balance
            reward += net_profit
        elif self.balance <= self.initial_balance - self.max_loss_limit:
            net_loss = self.initial_balance - self.balance
            reward += net_loss

        return reward
              
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_dim, action_dim, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64, epsilon=0.1, decay_rate=0.99, min_epsilon=0.01):
        self.model = DQN(input_dim, action_dim)
        self.target_model = DQN(input_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    def update(self):
        # Check if enough samples are available in the buffer
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough samples to perform an update

        # Sample a batch of transitions from the replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        # Convert lists of numpy arrays to tensors
        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32)
        batch_action = torch.tensor(np.array(batch_action), dtype=torch.long)
        batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32)
        batch_done = torch.tensor(np.array(batch_done), dtype=torch.float32)

        # Compute current Q-values using the policy network
        current_q_values = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        # Compute next Q-values using the target network, and detach from the graph
        next_q_values = self.target_model(batch_next_state).max(1)[0].detach()
        
        # Compute the expected Q-values
        expected_q_values = batch_reward + self.gamma * next_q_values * (1 - batch_done)

        # Compute loss using Mean Squared Error
        loss = nn.MSELoss()(current_q_values, expected_q_values)

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Return the loss value as a Python float for monitoring

    def sync_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
            q_values = np.zeros(self.action_dim)  # This ensures q_values is always an array
        return action, q_values
        
    def decay_epsilon(self):
        """Decays the epsilon value by the decay rate until it reaches a minimum value."""
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon)

    def save_state(self, filepath):
        """Saves the agent state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load_state(self, filepath, reset_epsilon=False, new_epsilon=0):
        """Loads the agent state, with an option to reset epsilon for evaluation."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if reset_epsilon:
            self.epsilon = new_epsilon  # Set epsilon to a specific value for evaluation if needed
        else:
            self.epsilon = checkpoint['epsilon']  # Use the saved epsilon value
        self.model.eval()  # Set the model to evaluation mode after loading
    # Define a function to evaluate the agent on out-of-sample data
    
def validate_agent(agent_save_filename, testing_set):
        num_columns = len(testing_set.columns)
        # Initialize the trading environment with out-of-sample data
        env_out_of_sample = ForexTradingEnv(testing_set, num_columns)

        observation_space_shape = env_out_of_sample.observation_space.shape[0]  # Or any other method to find the total features per timestep.

        input_dim = observation_space_shape

        # Determine the action space size from the environment
        action_size = env_out_of_sample.action_space.n

        # Create a new PPOAgent instance for evaluation purposes
        agent_evaluation = DQNAgent(input_dim, action_size)

        # Load the saved agent state for evaluation
        agent_evaluation.load_state(agent_save_filename, reset_epsilon=True, new_epsilon=0)

        # Initialize the done flag to False
        done = False

        # Initialize the total reward for out-of-sample testing
        total_test_reward = 0

        # Initialize a list to store all observations during out-of-sample testing
        all_observations = []

        # Reset the environment to get the initial state for out-of-sample testing
        state = env_out_of_sample.reset()

        # Continue the loop until the episode ends (the 'done' flag is True)
        while not done:
            # Select an action based on the current state using the agent's policy
            action, _ = agent_evaluation.act(state)
            # Temporary storage for out-of-sample trade history
            out_of_sample_trade_history = env_out_of_sample.trade_history
            # Execute the selected action in the environment and observe the next state and reward
            next_state, reward, done, _ = env_out_of_sample.step(action, out_of_sample_trade_history)
            # Accumulate the total reward for this episode
            total_test_reward += reward
            # Record the next state for later analysis
            all_observations.append(next_state)

            # Update the current state to the next state to continue the loop
            state = next_state

        # Evaluate performance
        performance_metrics_out_of_sample = env_out_of_sample.evaluate_performance()

        final_profit_evaluation = {performance_metrics_out_of_sample['Final Profit']}

        # Return the final profit from the out-of-sample evaluation as a float, the environment instance, and the total test reward for further analysis or use
        return total_test_reward, final_profit_evaluation

def train_agent_in_sample(episodes, training_set, testing_set, Pair, timeframe_str):
        num_columns = len(training_set.columns)
        # Instantiate the trading environment with in-sample data
        env_in_sample = ForexTradingEnv(training_set, num_columns)
        best_validation_score = -float('inf')

        # Initialize a list to store observations collected during training
        all_observations = []

        observation_space_shape = env_in_sample.observation_space.shape[0]  # Or any other method to find the total features per timestep.

        input_dim = observation_space_shape

        # Initialize the PPO agent with the specified input dimension and action space size
        agent = DQNAgent(input_dim, 4)

        writer = SummaryWriter()  # Initialize TensorBoard

        # Initialize a variable to keep track of the total number of steps taken during training
        total_steps = 0

        # Loop through each episode for training the agent
        for e in range(episodes):
            print(f'On episode {e}')

            # Reset the environment to get the initial state and store it
            state = env_in_sample.reset()
            all_observations.append(state)

            # Initialize total reward for the episode
            total_reward = 0
            # Flag to indicate if the episode is done
            done = False

            # Main loop to interact with the environment until the episode ends
            while not done:
                # Agent selects an action based on the current state
                action, _ = agent.act(state)

                # Fetch the current trade history from the environment
                in_sample_trade_history = env_in_sample.trade_history

                # Apply the selected action to the environment and receive feedback
                next_state, reward, done, _ = env_in_sample.step(action, in_sample_trade_history)
                # Add experience to the replay buffer
                agent.replay_buffer.push(state, action, reward, next_state, done)
                # Store the next observation
                all_observations.append(next_state)
                # Accumulate the total reward
                total_reward += reward

                loss = agent.update()

                if loss is not None:
                    writer.add_scalar('Loss/step', loss, total_steps)

                # Update the state for the next iteration
                state = next_state

                # Increment the total steps counter
                total_steps += 1

            agent.decay_epsilon()

            current_epsilon = agent.epsilon

            if current_epsilon is not None:
                writer.add_scalar('Epsilon/step', current_epsilon, total_steps)

            writer.add_scalar('Reward/Episode', total_reward, e)

            agent_save_filename = os.path.join(f"agent_state_{Pair}_{timeframe_str}.pkl")
                
            agent.save_state(agent_save_filename)  # Save the current best model state to the specified path.

            # Evaluate the agent's performance based on in-sample data
            performance_metrics_in_sample = env_in_sample.evaluate_performance()

            # In sample profit
            total_in_sample_profit = performance_metrics_in_sample['Final Profit']

            total_test_reward, total_out_of_sample_profit = validate_agent(agent_save_filename, testing_set)

            total_out_of_sample_profit_number = total_out_of_sample_profit.pop()

            print(f'epoch {e} total training reward is {total_reward} and total profit is {total_in_sample_profit} and total validation reward is {total_test_reward} and total profit is {total_out_of_sample_profit_number}')

            if (total_in_sample_profit > 0 and total_out_of_sample_profit_number > 0):
                # Define the directory to save the model based on some parameters
                save_dir = os.path.join("saved_models", f"{Pair}_{timeframe_str}")

                # Check if this directory exists, if not, create it
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # Construct the filename with the best validation score included
                agent_save_filename_best = os.path.join(save_dir, f"agent_state_best_{Pair}_{timeframe_str}_{total_in_sample_profit:.2f}_ {total_out_of_sample_profit_number:.2f}.pkl")
                
                # Save the state of the agent
                agent.save_state(agent_save_filename_best)
                print(f"Saved improved model to {agent_save_filename_best}")

        writer.close()  # Close the TensorBoard writer
            
def evaluate_model(agent_save_filename, testing_set):
        num_columns = len(testing_set.columns)
        # Initialize the trading environment with out-of-sample data
        env_out_of_sample = ForexTradingEnv(testing_set, num_columns)

        observation_space_shape = env_out_of_sample.observation_space.shape[0]  # Or any other method to find the total features per timestep.

        input_dim = observation_space_shape

        # Determine the action space size from the environment
        action_size = env_out_of_sample.action_space.n

        # Create a new PPOAgent instance for evaluation purposes
        agent_evaluation = DQNAgent(input_dim, action_size)

        # Load the saved agent state for evaluation
        agent_evaluation.load_state(agent_save_filename, reset_epsilon=True, new_epsilon=0)

        # Initialize the total reward for out-of-sample testing
        total_test_reward = 0

        # Initialize a list to store all observations during out-of-sample testing
        all_observations = []
        all_actions = []
        all_rewards = []
        all_q_values = []

        # Reset the environment to get the initial state for out-of-sample testing
        state = env_out_of_sample.reset()
        # Append the initial state to the observations list
        all_observations.append(state)

        done = False

        # Continue the loop until the episode ends (the 'done' flag is True)
        while not done:
            # Select an action based on the current state using the agent's policy
            action, q_values = agent_evaluation.act(state)  # No exploration
            # Temporary storage for out-of-sample trade history
            out_of_sample_trade_history = env_out_of_sample.trade_history

            # Execute the selected action in the environment and observe the next state and reward
            next_state, reward, done, _ = env_out_of_sample.step(action, out_of_sample_trade_history)
            # Accumulate the total reward for this episode
            total_test_reward += reward
            # Record the next state for later analysis
            all_observations.append(next_state)

            all_observations.append(next_state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_q_values.append(q_values)

            # Update the current state to the next state to continue the loop
            state = next_state

        # Evaluate performance
        performance_metrics_out_of_sample = env_out_of_sample.evaluate_performance()
                
        # Assuming equity curve plotting is desired at the end of each episode
        plt.figure(figsize=(15, 10))  # Create a figure with specified size
        ax = plt.gca()  # Get the current axes for plotting

        # Extract and plot the equity curve based on closed balances and corresponding dates
        balances = []
        dates = []
        for trade in env_out_of_sample.trade_history:
            if 'Closed Balance' in trade:
                balances.append(trade['Closed Balance'])
                dates.append(datetime.strptime(trade['Date'], '%Y-%m-%d'))

        ax.plot(dates, balances, '-o', label='Equity Curve')  # Plot the equity curve with markers

        # Customize the plot with titles, labels, and formatting
        ax.set_title(f'Equity Curve for {total_test_reward}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Balance')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the x-axis dates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Auto-format date labels for readability

        # Prepare data for a table summarizing key performance metrics
        metrics_data = [
            ['Final Balance', f"{performance_metrics_out_of_sample['Final Balance']:.2f}"],
            ['Final Profit', f"{performance_metrics_out_of_sample['Final Profit']:.2f}"],
            ['Maximum Drawdown', performance_metrics_out_of_sample['Maximum Drawdown']],
            ['Highest Daily Loss', performance_metrics_out_of_sample['Highest Daily Loss']],
            ['Profit Factor', performance_metrics_out_of_sample['Profit Factor']],
            ['Number of Closed Trades', str(performance_metrics_out_of_sample['Number of Closed Trades'])],
            ['Percent Profitable', performance_metrics_out_of_sample['Percent Profitable']],
            ['Total Reward', total_test_reward]
        ]

        # Add the table to the plot, specifying its location and appearance
        table = plt.table(cellText=metrics_data, colLabels=['Metric', 'Value'], loc='bottom', cellLoc='center', bbox=[0.0, -0.5, 1.0, 0.3])
        table.auto_set_font_size(False)  # Disable automatic font size setting
        table.set_fontsize(9)  # Set font size for the table
        table.scale(1, 1.5)  # Scale the table to fit better in the plot

        # Adjust the layout to make room for the table
        plt.subplots_adjust(left=0.2, bottom=0.2, top=0.583, right=0.567)

        # Use the current working directory as the base path
        base_path = os.getcwd()

        # Generate the folder name with a timestamp and reward
        specific_folder_name_validation = f"Evaluation_{total_test_reward}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Combine the base path with the new folder name to form the full path
        full_folder_path = os.path.join(base_path, specific_folder_name_validation)

        # Check if the directory exists, create it if not
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)

        # Construct the filename using the total reward and the unique identifier
        plot_filename_out_of_sample = os.path.join(specific_folder_name_validation, f"Evaluation_EquityCurve.png")

        # Save the plot to the specified file
        plt.savefig(plot_filename_out_of_sample)
        plt.close()  # Close the plot to free up system resources

        # Clear the current figure to avoid any overlap with future plots
        plt.clf()

        # Construct the filename for saving performance metrics, including the unique identifier and the current datetime
        metrics_filename = os.path.join(specific_folder_name_validation, f"Evaluation_PerformanceMetrics.txt")

        # Open the file for writing and record each performance metric
        with open(metrics_filename, 'w') as file:
            for metric in metrics_data:
                key, value = metric
                file.write(f"{key}: {value}\n")  

        # Construct the filename for saving the out-of-sample trade history, including the currency pair, timeframe, evaluation dates, and episode number
        out_of_sample_filename = os.path.join(specific_folder_name_validation, f"Evaluation_trade_history_episode.csv")

        # Convert the list of dictionaries (trade history) into a pandas DataFrame
        df_out_of_sample = pd.DataFrame(env_out_of_sample.trade_history)

        # Save the DataFrame to a CSV file at the specified path
        df_out_of_sample.to_csv(out_of_sample_filename, index=False)

        # Plotting the actions and rewards
        plt.figure(figsize=(14, 7))
        plt.subplot(2, 1, 1)
        plt.plot(all_rewards, label='Rewards')
        plt.title('Rewards per Step')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(all_actions, label='Actions', marker='o')
        plt.title('Actions per Step')
        plt.xlabel('Step')
        plt.ylabel('Action')
        plt.legend()

        plt.tight_layout()
        # Construct the filename using the total reward and the unique identifier
        plot_filename = os.path.join(specific_folder_name_validation, f"Evaluation_Rewards_and_Actions_per_Step.png")

        # Save the plot to the specified file
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free up system resources

        # Optionally print detailed decision data
        decision_data = pd.DataFrame({
            'Actions': all_actions,
            'Rewards': all_rewards,
            'Q-Values': [list(qv) for qv in all_q_values]  # Convert each tensor of Q-values to list
        })
        print(decision_data)
        return full_folder_path

def fetch_fx_data_mt5(symbol, timeframe_str, start_date, end_date):

    # Define your MetaTrader 5 account number
    account_number = 530062481
    # Define your MetaTrader 5 password
    password = 'N?G9rPt@'
    # Define the server name associated with your MT5 account
    server_name ='FTMO-Server3'

    # Initialize MT5 connection; if it fails, print error message and exit
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # Attempt to log in with the given account number, password, and server
    authorized = mt5.login(account_number, password=password, server=server_name)
    # If login fails, print error message, shut down MT5 connection, and exit
    if not authorized:
        print("login failed, error code =", mt5.last_error())
        mt5.shutdown()
        quit()
    # On successful login, print a confirmation message
    else:
        print("Connected to MetaTrader 5")

    # Set the timezone to Berlin, as MT5 times are in UTC
    timezone = pytz.timezone("Europe/Berlin")

    # Convert start and end dates to datetime objects, considering the timezone
    start_date = start_date.replace(tzinfo=timezone)
    end_date = end_date.replace(hour=23, minute=59, second=59, tzinfo=timezone)

    # Define a mapping from string representations of timeframes to MT5's timeframe constants
    timeframes = {
        '1H': mt5.TIMEFRAME_H1,
        'DAILY': mt5.TIMEFRAME_D1,
        '12H': mt5.TIMEFRAME_H12,
        '2H': mt5.TIMEFRAME_H2,
        '3H': mt5.TIMEFRAME_H3,
        '4H': mt5.TIMEFRAME_H4,
        '6H': mt5.TIMEFRAME_H6,
        '8H': mt5.TIMEFRAME_H8,
        '1M': mt5.TIMEFRAME_M1,
        '10M': mt5.TIMEFRAME_M10,
        '12M': mt5.TIMEFRAME_M12,
        '15M': mt5.TIMEFRAME_M15,
        '2M': mt5.TIMEFRAME_M2,
        '20M': mt5.TIMEFRAME_M20,
        '3M': mt5.TIMEFRAME_M3,
        '30M': mt5.TIMEFRAME_M30,
        '4M': mt5.TIMEFRAME_M4,
        '5M': mt5.TIMEFRAME_M5,
        '6M': mt5.TIMEFRAME_M6,
        '1MN': mt5.TIMEFRAME_MN1,
        '1W': mt5.TIMEFRAME_W1
    }

    # Retrieve the MT5 constant for the requested timeframe
    timeframe = timeframes.get(timeframe_str)
    # If the requested timeframe is invalid, print error message, shut down MT5, and return None
    if timeframe is None:
        print(f"Invalid timeframe: {timeframe_str}")
        mt5.shutdown()
        return None

    # Fetch the rates for the given symbol and timeframe within the start and end dates
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    # If no rates were fetched, print error message, shut down MT5, and return None
    if rates is None:
        print("No rates retrieved, error code =", mt5.last_error())
        mt5.shutdown()
        return None
    
    # Convert the fetched rates into a Pandas DataFrame
    rates_frame = pd.DataFrame(rates)
    # Convert the 'time' column from UNIX timestamps to human-readable dates
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

    # Set the 'time' column as the DataFrame index and ensure its format is proper for datetime
    rates_frame.set_index('time', inplace=True)
    # %Y-%m-%d %H:%M:%S
    rates_frame.index = pd.to_datetime(rates_frame.index, format="%Y-%m-%d %H:%M:%S")

    # Check if 'tick_volume' column is present in the fetched data
    if 'tick_volume' not in rates_frame.columns:
        print("tick_volume is not in the fetched data. Ensure it's included in the API call.")
    else:
        print("tick_volume is included in the data.")
    
    # Shut down the MT5 connection before returning the data
    mt5.shutdown()
    
    # Return the prepared DataFrame containing the rates
    return rates_frame

def get_user_date_input(prompt):
    # Specify the expected date format
    date_format = '%Y-%m-%d'
    # Prompt the user for a date
    date_str = input(prompt)
    # Loop until a valid date format is entered
    while True:
        try:
            # Attempt to parse the date; if successful, it's valid, so return it
            pd.to_datetime(date_str, format=date_format)
            return date_str
        except ValueError:
            # If parsing fails, notify the user and prompt again
            print("The date format is incorrect. Please enter the date in 'YYYY-MM-DD' format.")
            date_str = input(prompt)

def calculate_indicators(data, bollinger_length=12, bollinger_std_dev=1.5, sma_trend_length=50, window=9):
    # Calculate the 50-period simple moving average of the 'close' price
    data['SMA_50'] = ta.sma(data['close'], length=50)
    # Calculate the 200-period simple moving average of the 'close' price
    data['SMA_200'] = ta.sma(data['close'], length=200)
    
    # Calculate the 50-period exponential moving average of the 'close' price
    data['EMA_50'] = ta.ema(data['close'], length=50)
    # Calculate the 200-period exponential moving average of the 'close' price
    data['EMA_200'] = ta.ema(data['close'], length=200)

    # Calculate the 9-period exponential moving average for scalping strategies
    data['EMA_9'] = ta.ema(data['close'], length=9)
    # Calculate the 21-period exponential moving average for scalping strategies
    data['EMA_21'] = ta.ema(data['close'], length=21)
    
    # Generate original Bollinger Bands with a 20-period SMA and 2 standard deviations
    original_bollinger = ta.bbands(data['close'], length=20, std=2)
    # The 20-period simple moving average for the middle band
    data['SMA_20'] = ta.sma(data['close'], length=20)
    # Upper and lower bands from the original Bollinger Bands calculation
    data['Upper Band'] = original_bollinger['BBU_20_2.0']
    data['Lower Band'] = original_bollinger['BBL_20_2.0']

    # Generate updated Bollinger Bands for scalping with custom length and standard deviation
    updated_bollinger = ta.bbands(data['close'], length=bollinger_length, std=bollinger_std_dev)
    # Assign lower, middle, and upper bands for scalping
    data['Lower Band Scalping'], data['Middle Band Scalping'], data['Upper Band Scalping'] = updated_bollinger['BBL_'+str(bollinger_length)+'_'+str(bollinger_std_dev)], ta.sma(data['close'], length=bollinger_length), updated_bollinger['BBU_'+str(bollinger_length)+'_'+str(bollinger_std_dev)]
    
    # Calculate the MACD indicator and its signal line
    macd = ta.macd(data['close'])
    data['MACD'] = macd['MACD_12_26_9']
    data['Signal_Line'] = macd['MACDs_12_26_9']
    
    # Calculate the Relative Strength Index (RSI) with the specified window length
    data[f'RSI_{window}'] = ta.rsi(data['close'], length=window).round(2)

    # Calculate a 5-period RSI for scalping strategies
    data[f'RSI_5 Scalping'] = ta.rsi(data['close'], length=5).round(2)

    # Calculate a simple moving average for trend analysis in scalping strategies
    data[f'SMA_{sma_trend_length}'] = ta.sma(data['close'], length=sma_trend_length)

    # Calculate the Stochastic Oscillator
    stoch = ta.stoch(data['high'], data['low'], data['close'])
    data['Stoch_%K'] = stoch['STOCHk_14_3_3']
    data['Stoch_%D'] = stoch['STOCHd_14_3_3']

    # Return the data with added indicators
    return data

def split_and_save_dataset(dataset, timeframe, pair):
    # Calculate the split index for training and testing
    split_index_train_test = int(len(dataset) * 0.8)
    
    # Split the dataset into training and testing
    training_set = dataset.iloc[:split_index_train_test]
    testing_set = dataset.iloc[split_index_train_test:]
    
    # Further split the training set to create a validation set (80% training, 20% validation of the training set)
    split_index_train_val = int(len(training_set) * 0.8)
    final_training_set = training_set.iloc[:split_index_train_val]
    validation_set = training_set.iloc[split_index_train_val:]

    # Search for CSV files starting with "validation"
    validation_files = glob.glob('validation*.csv')

    # Search for CSV files starting with "training"
    training_files = glob.glob('training*.csv')

    # Search for CSV files starting with "testing"
    testing_files = glob.glob('testing*.csv')

    for file in validation_files:
        os.remove(file)

    for file in training_files:
        os.remove(file)

    for file in testing_files:
        os.remove(file)

    # Save the sets into CSV files
    dataset.to_csv(f'Full_data.csv', index=True)
    final_training_set.to_csv(f'training_{pair}_{timeframe}_data.csv', index=True)
    validation_set.to_csv(f'validation_{pair}_{timeframe}_data.csv', index=True)
    testing_set.to_csv(f'testing_{pair}_{timeframe}_data.csv', index=True)
    
    # Return the split datasets
    return final_training_set, validation_set

def read_csv_to_dataframe(file_path):
    # Read the CSV file into a DataFrame and set the first column as the index
    df = pd.read_csv(file_path)
    # Set the 'time' column as the DataFrame index
    df.set_index('time', inplace=True)
    
    try:
        # Try to convert the index using the first datetime format
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    except ValueError:
        # If there is a ValueError, try the second datetime format
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    
    return df

def training_DQN_model(choice):
    if choice == '1':
        training_set, testing_set, Pair, timeframe_str = get_data()
        episodes = 1000

        train_agent_in_sample(episodes, training_set, testing_set, Pair, timeframe_str)

        target_directory = evaluate_DQN_model('6')

        clean_evaluate_folder(target_directory)
    elif choice == '3':
        # Search for CSV files starting with "validation"
        validation_files = glob.glob('validation*.csv')

        # Search for CSV files starting with "training"
        training_files = glob.glob('training*.csv')

        for file in validation_files:
            parts = file.split('_')

            # Extract the pair and timeframe
            Pair = parts[1]
            timeframe_str = parts[2]

            testing_set = read_csv_to_dataframe(file)

        for file in training_files:
            training_set = read_csv_to_dataframe(file)

        episodes = 1000

        train_agent_in_sample(episodes, training_set, testing_set, Pair, timeframe_str)

        evaluate_DQN_model('6')

        clean_evaluate_folder()
    elif choice == '6':
        forex_pairs = [
            'GBPUSD', 'USDCHF', 'USDCAD', 'AUDUSD', 'AUDNZD', 'AUDCAD',
            'AUDCHF', 'GBPCAD', 'NZDUSD', 'EURGBP', 'EURAUD',
            'EURCHF', 'EURNZD', 'EURCAD', 'GBPCAD', 'GBPCHF',
            'CADCHF', 'GBPAUD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'EURUSD'
        ]

        timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()

        for pair in forex_pairs:
            print(f"Processing {pair} on {timeframe_str}")
            try:
                training_set, testing_set = get_data_multiple(pair, timeframe_str)
                episodes = 1000
                train_agent_in_sample(episodes, training_set, testing_set, pair, timeframe_str)
                target_directory = evaluate_DQN_model('6')
                clean_evaluate_folder(target_directory)
            except Exception as e:
                print(f"Failed to process {pair}: {str(e)}")

def clean_evaluate_folder(target_directory):
    # Assume the base directory and file pattern for testing
    testing_files = glob.glob('testing*.csv')
    
    if not testing_files:
        print("No testing files found.")
        return

    # Assume the first (and only) testing file contains the pair and timeframe in its name
    test_file_path = testing_files[0]
    parts = os.path.basename(test_file_path).split('_')

    # Extract the pair and timeframe from the file name
    Pair = parts[1]
    timeframe_str = parts[2].split('.')[0]  # Assuming the file extension follows immediately after the timeframe

    # Read the testing data to determine the period covered
    df = pd.read_csv(test_file_path, parse_dates=['time'], index_col='time')
    first_date = df.index.min().strftime('%Y-%m-%d')
    last_date = df.index.max().strftime('%Y-%m-%d')

    # Define the base folder name based on the currency pair, timeframe, and date range
    base_folder_name = f"evaluate_{Pair}_{timeframe_str}_validate_period_{first_date}_to_{last_date}"

    # Determine the full path of the base folder
    base_folder_path = os.path.join(os.getcwd(), base_folder_name)

    # Ensure the base directory exists before proceeding
    if not os.path.exists(base_folder_path):
        print(f"Directory does not exist: {base_folder_path}")
        return

    # List all subdirectories within the base folder
    subdirectories = [os.path.join(base_folder_path, d) for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d))]

    # Define the regex pattern to extract the numerical value (including negative numbers) from the directory names
    pattern = r"_(-?\d+\.\d+)_"

    # Loop through each subdirectory
    for subdir in subdirectories:
        # Use regular expression to find matches
        match = re.search(pattern, os.path.basename(subdir))
        if match:
            # Convert the extracted value to a float
            value = float(match.group(1))
            # Check if the value is positive and less than 100
            if value < 100:
                shutil.rmtree(subdir)  # Uncomment this line to enable actual deletion

    testing_file = glob.glob('testing*.csv')

    training_file = glob.glob('training*.csv')

    validation_file = glob.glob('validation*.csv')

    full_data_file = glob.glob('Full_data*.csv')

    # Corrected code to move the file to the target_directory
    test_file_path = testing_file[0]  # Use the first (and only) testing file
    destination_test_file_path = os.path.join(target_directory, os.path.basename(test_file_path))
    print(destination_test_file_path)
    # Corrected parameters order for shutil.move
    shutil.move(test_file_path, destination_test_file_path)

    # Corrected code to move the file to the target_directory
    train_file_path = training_file[0]  # Use the first (and only) testing file
    destination_train_file_path = os.path.join(target_directory, os.path.basename(train_file_path))
    # Corrected parameters order for shutil.move
    shutil.move(train_file_path, destination_train_file_path)

    # Corrected code to move the file to the target_directory
    validation_file_path = validation_file[0]  # Use the first (and only) testing file
    destination_validation_file_path = os.path.join(target_directory, os.path.basename(validation_file_path))
    # Corrected parameters order for shutil.move
    shutil.move(validation_file_path, destination_validation_file_path)

    # Corrected code to move the file to the target_directory
    full_data_file_path = full_data_file[0]  # Use the first (and only) testing file
    destination_full_data_file_path = os.path.join(target_directory, os.path.basename(full_data_file_path))
    # Corrected parameters order for shutil.move
    shutil.move(full_data_file_path, destination_full_data_file_path)

def extract_weight_from_folder_name(folder_name):
    # Use regular expression to extract the numerical value from the folder name
    match = re.search(r"_(\d+\.\d+)_", folder_name)
    if match:
        return float(match.group(1))

def extract_info_from_folder_name_2(folder_name):
    # Split the folder name by underscores
    parts = folder_name.split('_')
    
    # Extract currency pair and timeframe assuming the format includes them
    if len(parts) >= 4:
        pair = parts[2]
        timeframe_str = parts[3]
        return pair, timeframe_str
    else:
        raise ValueError("Folder name format is incorrect or missing essential information.")

def extract_info_from_folder_name(folder_name):
    # Extracts information assuming folder name format "evaluate_{Pair}_{timeframe}_otherinfo"
    parts = folder_name.split('_')
    pair = parts[1]
    timeframe = parts[2]
    return pair, timeframe

def get_evaluation_folders(Pair, timeframe_str):
    # Get the current working directory
    base_directory = os.getcwd()

    # Construct the prefix to match directories for evaluations
    directory_prefix = f"evaluate_{Pair}_{timeframe_str}"

    # List all entries in the base directory
    full_paths = [os.path.join(base_directory, d) for d in os.listdir(base_directory)
                  if os.path.isdir(os.path.join(base_directory, d)) and d.startswith(directory_prefix)]

    # Dictionaries to hold all pairs and their evaluations, and paths to agent states
    evaluations_by_pair_and_timeframe = {}
    agent_state_paths = {}
    folder_paths = {}  # Dictionary to store full paths of evaluation folders

    # Process each folder that matches the evaluation pattern
    for folder in full_paths:
        key = (Pair, timeframe_str)

        # Directory for individual evaluations within the main folder
        evaluation_prefix = "Evaluation_"
        evaluation_paths = [os.path.join(folder, d) for d in os.listdir(folder)
                            if os.path.isdir(os.path.join(folder, d)) and d.startswith(evaluation_prefix)]

        # Extract last sections of the evaluation paths and collect agent state files
        evaluations = []
        for eval_path in evaluation_paths:
            evaluation_name = os.path.basename(eval_path)
            evaluations.append(evaluation_name)
            
            # Store full path for each evaluation folder
            folder_paths[evaluation_name] = eval_path
            
            # Search for agent_state_best_ files
            agent_files = [f for f in os.listdir(eval_path) if f.startswith('agent_state_best_')]
            if agent_files:
                agent_state_paths[evaluation_name] = os.path.join(eval_path, agent_files[0])

        # Store the evaluations under the corresponding pair and timeframe
        evaluations_by_pair_and_timeframe[key] = evaluations

    return evaluations_by_pair_and_timeframe, agent_state_paths, folder_paths

def ensemble_predict(state, input_dim, action_size, Pair, timeframe_str):
    evaluations_by_pair_and_timeframe, agent_state_paths, _ = get_evaluation_folders(Pair, timeframe_str)
    
    actions = []
    weights = []

    # Create a new PPOAgent instance for evaluation purposes
    agent = DQNAgent(input_dim, action_size)

    # Collect all actions predicted by each agent and corresponding weights from folder names
    for evaluation in evaluations_by_pair_and_timeframe.values():
        for folder_name in evaluation:
            agent_path = agent_state_paths.get(folder_name)
            if agent_path:
                agent.load_state(agent_path, reset_epsilon=True, new_epsilon=0)
                result = agent.act(state)
                
                # Check if the result is a tuple and extract the action
                if isinstance(result, tuple):
                    action = result[0]  # Assuming the action is the first element of the tuple
                    if isinstance(action, torch.Tensor):
                        action = action.detach().item()  # Converts tensor to a Python int
                else:
                    action = result  # Direct use if not a tuple
                
                weight = extract_weight_from_folder_name(folder_name)

                actions.append(action)
                weights.append(weight)

    # Create a weighted action count array
    action_counts = np.zeros(5)  # Assuming there are 5 possible actions (0 to 4)
    for action, weight in zip(actions, weights):
        action_counts[action] += weight

    # Determine the action with the highest weighted frequency
    consensus_action = np.argmax(action_counts)
    return consensus_action

def evaluate_ensemble():
    training_set, testing_set, Pair, timeframe_str = get_data()
    num_columns = len(testing_set.columns)
    # Initialize the trading environment with out-of-sample data
    env_out_of_sample = ForexTradingEnv(testing_set, num_columns)

    observation_space_shape = env_out_of_sample.observation_space.shape[0]  # Or any other method to find the total features per timestep.

    input_dim = observation_space_shape

    # Determine the action space size from the environment
    action_size = env_out_of_sample.action_space.n

    # Initialize the total reward for out-of-sample testing
    total_test_reward = 0

    # Initialize a list to store all observations during out-of-sample testing
    all_observations = []
    all_actions = []
    all_rewards = []

    # Reset the environment to get the initial state for out-of-sample testing
    state = env_out_of_sample.reset()
    # Append the initial state to the observations list
    all_observations.append(state)

    done = False

    # Continue the loop until the episode ends (the 'done' flag is True)
    while not done:
        # Select an action based on the current state using the agent's policy
        action = ensemble_predict(state, input_dim, action_size, Pair, timeframe_str)
        # Temporary storage for out-of-sample trade history
        out_of_sample_trade_history = env_out_of_sample.trade_history

        # Execute the selected action in the environment and observe the next state and reward
        next_state, reward, done, _ = env_out_of_sample.step(action, out_of_sample_trade_history)
        # Accumulate the total reward for this episode
        total_test_reward += reward
        # Record the next state for later analysis
        all_observations.append(next_state)

        all_observations.append(next_state)
        all_actions.append(action)
        all_rewards.append(reward)

        # Update the current state to the next state to continue the loop
        state = next_state

    # Evaluate performance
    performance_metrics_out_of_sample = env_out_of_sample.evaluate_performance()
            
    # Assuming equity curve plotting is desired at the end of each episode
    plt.figure(figsize=(15, 10))  # Create a figure with specified size
    ax = plt.gca()  # Get the current axes for plotting

    # Extract and plot the equity curve based on closed balances and corresponding dates
    balances = []
    dates = []
    for trade in env_out_of_sample.trade_history:
        if 'Closed Balance' in trade:
            balances.append(trade['Closed Balance'])
            dates.append(datetime.strptime(trade['Date'], '%Y-%m-%d'))

    ax.plot(dates, balances, '-o', label='Equity Curve')  # Plot the equity curve with markers

    # Customize the plot with titles, labels, and formatting
    ax.set_title(f'Equity Curve for {total_test_reward}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Balance')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format the x-axis dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()  # Auto-format date labels for readability

    # Prepare data for a table summarizing key performance metrics
    metrics_data = [
        ['Final Balance', f"{performance_metrics_out_of_sample['Final Balance']:.2f}"],
        ['Final Profit', f"{performance_metrics_out_of_sample['Final Profit']:.2f}"],
        ['Maximum Drawdown', performance_metrics_out_of_sample['Maximum Drawdown']],
        ['Highest Daily Loss', performance_metrics_out_of_sample['Highest Daily Loss']],
        ['Profit Factor', performance_metrics_out_of_sample['Profit Factor']],
        ['Number of Closed Trades', str(performance_metrics_out_of_sample['Number of Closed Trades'])],
        ['Percent Profitable', performance_metrics_out_of_sample['Percent Profitable']],
        ['Total Reward', total_test_reward]
    ]

    # Add the table to the plot, specifying its location and appearance
    table = plt.table(cellText=metrics_data, colLabels=['Metric', 'Value'], loc='bottom', cellLoc='center', bbox=[0.0, -0.5, 1.0, 0.3])
    table.auto_set_font_size(False)  # Disable automatic font size setting
    table.set_fontsize(9)  # Set font size for the table
    table.scale(1, 1.5)  # Scale the table to fit better in the plot

    # Adjust the layout to make room for the table
    plt.subplots_adjust(left=0.2, bottom=0.2, top=0.583, right=0.567)

    # Use the current working directory as the base path
    base_path = os.getcwd()

    # Generate the folder name with a timestamp and reward
    specific_folder_name_validation = f"Evaluation_{total_test_reward}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Combine the base path with the new folder name to form the full path
    full_folder_path = os.path.join(base_path, specific_folder_name_validation)

    # Check if the directory exists, create it if not
    if not os.path.exists(full_folder_path):
        os.makedirs(full_folder_path)

    # Construct the filename using the total reward and the unique identifier
    plot_filename_out_of_sample = os.path.join(specific_folder_name_validation, f"Evaluation_EquityCurve.png")

    # Save the plot to the specified file
    plt.savefig(plot_filename_out_of_sample)
    plt.close()  # Close the plot to free up system resources

    # Clear the current figure to avoid any overlap with future plots
    plt.clf()

    # Construct the filename for saving performance metrics, including the unique identifier and the current datetime
    metrics_filename = os.path.join(specific_folder_name_validation, f"Evaluation_PerformanceMetrics.txt")

    # Open the file for writing and record each performance metric
    with open(metrics_filename, 'w') as file:
        for metric in metrics_data:
            key, value = metric
            file.write(f"{key}: {value}\n")  

    # Construct the filename for saving the out-of-sample trade history, including the currency pair, timeframe, evaluation dates, and episode number
    out_of_sample_filename = os.path.join(specific_folder_name_validation, f"Evaluation_trade_history_episode.csv")

    # Convert the list of dictionaries (trade history) into a pandas DataFrame
    df_out_of_sample = pd.DataFrame(env_out_of_sample.trade_history)

    # Save the DataFrame to a CSV file at the specified path
    df_out_of_sample.to_csv(out_of_sample_filename, index=False)

    # Plotting the actions and rewards
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(all_rewards, label='Rewards')
    plt.title('Rewards per Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(all_actions, label='Actions', marker='o')
    plt.title('Actions per Step')
    plt.xlabel('Step')
    plt.ylabel('Action')
    plt.legend()

    plt.tight_layout()
    # Construct the filename using the total reward and the unique identifier
    plot_filename = os.path.join(specific_folder_name_validation, f"Evaluation_Rewards_and_Actions_per_Step.png")

    # Save the plot to the specified file
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up system resources

    # Optionally print detailed decision data
    decision_data = pd.DataFrame({
        'Actions': all_actions,
        'Rewards': all_rewards
    })
    print(decision_data)

def evaluate_DQN_model(choice):
    if choice == '2':
        # Search for CSV files starting with "testing"
        testing_files = glob.glob('testing*.csv')

        for file in testing_files:
            parts = file.split('_')

            # Extract the pair and timeframe
            Pair = parts[1]
            timeframe_str = parts[2]

            evaluation_dataset = read_csv_to_dataframe(file)
        evaluate_model(f"agent_state_{Pair}_{timeframe_str}.pkl", evaluation_dataset)
    elif choice == '4':
        # Search for CSV files starting with "testing"
        testing_files = glob.glob('testing*.csv')

        for file in testing_files:
            parts = file.split('_')

            # Extract the pair and timeframe
            Pair = parts[1]
            timeframe_str = parts[2]

        file_data = 'Full_data.csv'
        evaluation_dataset = read_csv_to_dataframe(file_data)
        evaluate_model(f"agent_state_{Pair}_{timeframe_str}.pkl", evaluation_dataset)
    elif choice == '6':
        # Search for CSV files starting with "testing"
        testing_files = glob.glob('testing*.csv')

        for file in testing_files:
            parts = file.split('_')

            # Extract the pair and timeframe
            Pair = parts[1]
            timeframe_str = parts[2]

            test_file_path = testing_files[0]  # Use the first (and only) testing file

            df = pd.read_csv(test_file_path, parse_dates=['time'], index_col='time')
    
            # Get the first and last index values, which are the first and last dates
            first_date = df.index.min()
            last_date = df.index.max()

        evaluation_dataset = read_csv_to_dataframe(file)
        model_dir = f'saved_models/{Pair}_{timeframe_str}'

        # Current file name
        current_file_name = f'agent_state_{Pair}_{timeframe_str}.pkl'
        
        # Current directory path of the file
        current_file_path = os.path.join(os.getcwd(), current_file_name)

        # Construct the target file path
        target_file_path = os.path.join(model_dir, current_file_name)
        
        # Move the file
        shutil.move(current_file_path, target_file_path)

        model_paths = glob.glob(os.path.join(model_dir, '*.pkl'))

        for model_path in model_paths:
            full_folder_path =  evaluate_model(f"{model_path}", evaluation_dataset)

            # Copy the testing file to the evaluated model's results folder
            destination_test_file_path = os.path.join(full_folder_path, os.path.basename(test_file_path))
            shutil.copy(test_file_path, destination_test_file_path)

            # Extract the filename from the model path
            model_filename = os.path.basename(model_path)
            
            # Create the new model path using the folder path and model filename
            new_model_path = os.path.join(full_folder_path, model_filename)
            
            # Move the model file to the evaluation directory
            shutil.move(model_path, new_model_path)

            # Determine the target directory for organizing evaluations
            target_directory = os.path.join(os.getcwd(), f"evaluate_{Pair}_{timeframe_str}_validate_period_{first_date.strftime('%Y-%m-%d')}_to_{last_date.strftime('%Y-%m-%d')}")
            os.makedirs(target_directory, exist_ok=True)  # Ensure the directory exists

            # Construct the new path for the evaluated folder
            new_folder_path = os.path.join(target_directory, os.path.basename(full_folder_path))

            # Move the entire folder
            shutil.move(full_folder_path, new_folder_path)

        return target_directory

def get_data():
    # Retrieve and store the current date
    current_date = str(datetime.now().date())
    current_date_datetime = datetime.now().date()

    # Calculate the date for one month before the current date
    one_month_before = current_date_datetime - relativedelta(months=1)
    # Convert to string if needed
    one_month_before_str = str(one_month_before)

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    # Prompt the user for the desired timeframe for analysis and standardize the input
    timeframe_str = input("Enter the currency pair (e.g., Daily, 1H): ").strip().upper()
    # Prompt the user for the currency pair they're interested in and standardize the input
    Pair = input("Enter the currency pair (e.g., GBPUSD, EURUSD): ").strip().upper()

    training_start_date = "2023-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    eur_usd_data = calculate_indicators(eur_usd_data) 

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    dataset = dataset.fillna(0)

    training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)

    return training_set, testing_set, Pair, timeframe_str

def get_data_multiple(Pair, timeframe_str):
    # Retrieve and store the current date
    current_date = str(datetime.now().date())
    current_date_datetime = datetime.now().date()

    # Calculate the date for one month before the current date
    one_month_before = current_date_datetime - relativedelta(months=1)
    # Convert to string if needed
    one_month_before_str = str(one_month_before)

    # Hardcoded start date for strategy evaluation
    strategy_start_date_all = "1971-01-04"
    # Use the current date as the end date for strategy evaluation
    strategy_end_date_all = current_date

    # Convert string representation of dates to datetime objects for further processing
    start_date_all = datetime.strptime(strategy_start_date_all, "%Y-%m-%d")
    end_date_all = datetime.strptime(strategy_end_date_all, "%Y-%m-%d")

    training_start_date = "2023-01-01"
    training_end_date = current_date

    # Fetch and prepare the FX data for the specified currency pair and timeframe
    eur_usd_data = fetch_fx_data_mt5(Pair, timeframe_str, start_date_all, end_date_all)

    eur_usd_data = calculate_indicators(eur_usd_data) 

    # Filter the EUR/USD data for the in-sample training period
    dataset = eur_usd_data[(eur_usd_data.index >= training_start_date) & (eur_usd_data.index <= training_end_date)]

    dataset = dataset.fillna(0)

    training_set, testing_set = split_and_save_dataset(dataset, timeframe_str, Pair)

    return training_set, testing_set

def extract_weights_from_filename(filename):
    # Enhanced regex pattern to extract numbers (weights) from the filename
    # This pattern also handles spaces and assumes weights may be surrounded by varied characters
    pattern = re.compile(r'_\s*([\d\.]+)\s*_\s*([\d\.]+)\s*\.pkl$')
    match = pattern.search(filename)
    if match:
        weight1 = float(match.group(1))
        weight2 = float(match.group(2))
        return weight1, weight2
    else:
        return None, None

def get_best_folder():
    prefix = "evaluate_"
    current_directory = os.getcwd()
    entries = os.listdir(current_directory)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(current_directory, entry)) and entry.startswith(prefix)]
    
    # Dictionary to track the maximum weight and corresponding folder path for each pair and timeframe
    max_folders_by_pair_and_timeframe = {}

    for folder in folders:
        Pair, timeframe_str = extract_info_from_folder_name(folder)
        if Pair and timeframe_str:
            key = (Pair, timeframe_str)
            evaluations_by_pair_and_timeframe, agent_state_paths, folder_paths = get_evaluation_folders(Pair, timeframe_str)
            
            # Initialize max weight and folder path if the key is not already present
            if key not in max_folders_by_pair_and_timeframe:
                max_folders_by_pair_and_timeframe[key] = {'max_weight': 0, 'max_folder_path': None}

            for evaluation in evaluations_by_pair_and_timeframe.values():
                for folder_name in evaluation:
                    weight = extract_weight_from_folder_name(folder_name)
                    if weight > max_folders_by_pair_and_timeframe[key]['max_weight']:
                        max_folders_by_pair_and_timeframe[key]['max_weight'] = weight
                        max_folders_by_pair_and_timeframe[key]['max_folder_path'] = folder_paths[folder_name]
    
    return max_folders_by_pair_and_timeframe

def get_best_agent():
    max_folders_by_pair_and_timeframe = get_best_folder()
    max_total_profit = -float('inf')
    best_details = None

    for key, value in max_folders_by_pair_and_timeframe.items():
        pair = key[0]
        timeframe = key[1]
        folder_path = value['max_folder_path']

        # Ensure folder_path is not None
        if folder_path:
            folder_name = os.path.basename(folder_path)
            pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
            
            if pkl_files:  # Check if there are any .pkl files
                pkl_file_path = os.path.join(folder_path, pkl_files[0])  # Use the first .pkl file found
                pkl_file_name = os.path.basename(pkl_file_path)  # Extract the filename from the full path

                # Extract weights which are used as profit metrics
                profit_evaluate = extract_weight_from_folder_name(folder_name)
                profit_training, profit_validation = extract_weights_from_filename(pkl_file_name)

                # Calculate the total profit for comparison
                total_profit = profit_evaluate + profit_training + profit_validation

                print(f"{pair}, {timeframe}, Training: {profit_training}, Validation: {profit_validation}, Evaluation: {profit_evaluate}, Total: {total_profit}")

                # Check if this total profit is the highest encountered so far
                if total_profit > max_total_profit:
                    max_total_profit = total_profit
                    best_details = {
                        "pair": pair,
                        "timeframe": timeframe,
                        "pkl_file_path": pkl_file_path,
                        "folder_path": folder_path
                    }
        else:
            print(f"No valid folder path provided for {pair} {timeframe}")

    # After the loop, print details of the best result
    if best_details:
        print(f"Highest total profit found: {max_total_profit}")
        print(f"Pair: {best_details['pair']}, Timeframe: {best_details['timeframe']}")
        print(f"PKL File Path: {best_details['pkl_file_path']}")
        print(f"Folder Path: {best_details['folder_path']}")

    return best_details

def get_best_trade_history():
    max_folders_by_pair_and_timeframe = get_best_folder()
    all_dfs = []

    for key, value in max_folders_by_pair_and_timeframe.items():
        pair = key[0]
        timeframe = key[1]
        folder_path = value['max_folder_path']
        csv_file_path = os.path.join(folder_path, 'Evaluation_trade_history_episode.csv')

        try:
            df = pd.read_csv(csv_file_path)
            # Include pair and timeframe in the DataFrame for clarity
            df['Pair'] = pair
            df['Timeframe'] = timeframe
            all_dfs.append(df)
            print(f"DataFrame for {pair}-{timeframe} loaded with shape: {df.shape}")
        except FileNotFoundError:
            print(f"CSV file not found in {csv_file_path}")
        except Exception as e:
            print(f"Failed to load CSV from {csv_file_path}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Sort the DataFrame by 'Step' column if it exists
        if 'Step' in combined_df.columns:
            combined_df = combined_df.sort_values('Step').reset_index(drop=True)
        # Save the sorted DataFrame
        combined_df.to_csv('sorted_combined_trade_history_by_step.csv', index=False)
        print("Sorted and combined DataFrame saved to 'sorted_combined_trade_history_by_step.csv'.")
    else:
        print("No DataFrames were loaded, no CSV file created.")

def main_menu():
    while True:
        print("\nMain Menu:")
        print("1 - Train a DQN model with latest data")
        print("2 - Evaluate a DQN model - Validate data")
        print("3 - Train a DQN model with csv file data")
        print("4 - Evaluate a DQN model - Full data")
        print("5 - Get Data")
        print("6 - Train Multiple Different Forex Pairs")
        print("7 - Ensemble Evaluate")
        print("8 - Find Best Agent")

        choice = input("Enter your choice (1/2/3/4/5): ")

        if choice == '1':
            training_DQN_model(choice)
            break
        elif choice == '2':
            evaluate_DQN_model(choice)
            break
        elif choice == '3':
            training_DQN_model(choice)
            break
        elif choice == '4':
            evaluate_DQN_model(choice)
            break
        elif choice == '5':
            get_data()
            break
        elif choice == '6':
            training_DQN_model(choice)
            break
        elif choice == '7':
            evaluate_ensemble()
            break
        elif choice == '8':
            get_best_agent()
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4 or 5.")

if __name__ == "__main__":
    main_menu()