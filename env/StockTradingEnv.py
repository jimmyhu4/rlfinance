import random
import gym
import numpy as np 
import pandas as pd 

MAX_ACCOUNT_BALANCE = 2147483647 # Use maximum integer
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000
N_PREV_DAYS = 5

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, df):
        """Initialize parameters"""
        super(StockTradingEnv, self).__init__()

        # df contains historical price of the stock
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)    

        # Actions of the format Buy x%, Sell x%, Hold
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16
        )

        # Open, High, Low, Close prices for the last N days
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(N_PREV_DAYS+1, N_PREV_DAYS+1), dtype=np.float16
        )

        def _next_observation(self):
            # Get the stock data points for the last N days and scale to between 0-1
            frame = np.array([
                self.df.loc[self.current_step: self.current_step +
                            N_PREV_DAYS, "Open"].values / MAX_SHARE_PRICE,
                self.df.loc[self.current_step: self.current_step +
                            N_PREV_DAYS, "High"].values / MAX_SHARE_PRICE,
                self.df.loc[self.current_step: self.current_step +
                            N_PREV_DAYS, "Low"].values / MAX_SHARE_PRICE,
                self.df.loc[self.current_step: self.current_step +
                            N_PREV_DAYS, "Close"].values / MAX_SHARE_PRICE,
                self.df.loc[self.current_step: self.current_step +
                            N_PREV_DAYS, "Volume"].values / MAX_NUM_SHARES,
            ])

            # Append additional data and scale each value to between 0-1
            obs = np.append(frame, [[
                self.balance / MAX_ACCOUNT_BALANCE,
                self.max_net_worth / MAX_ACCOUNT_BALANCE,
                self.shares_held / MAX_NUM_SHARES,
                self.cost_basis / MAX_SHARE_PRICE,
                self.total_shares_sold / MAX_NUM_SHARES,
                self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
            ]], axis=0)

            return obs