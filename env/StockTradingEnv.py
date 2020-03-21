import random
import gym
import numpy as np 
import pandas as pd 

MAX_ACCOUNT_BALANCE = 2147483647 # Use maximum integer

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

        # Open, High, Low, Close prices for the last certain number of days
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(N_PREV_DAYS+1, N_PREV_DAYS+1), dtype=np.float16
        )
