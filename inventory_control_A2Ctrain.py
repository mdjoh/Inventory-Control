# Train A2C algorithm to find optimal policy for inventory control problem

import numpy
import torch
import gym
from gym_inventory.envs.inventory_env import InventoryEnv
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import pandas

# ----------------------------------SETUP------------------------------------------------
numpy.random.seed(42)

# Make inventory control environment
env = gym.make('Inventory-v0')

# Define algorithm and number of timesteps
algo = 'A2C-default-85e-5-adam'
steps = 800000

# Pass in created environment to monitor results
# A csv file with the monitor results will be generated in the current working directory
env = Monitor(env, './'+algo+'-time_steps'+str(steps))

# ----------------------------------TRAINING------------------------------------------------
env.reset()

# Define model to run in inventory control environment
model = A2C('MlpPolicy', env, learning_rate= 85e-5, use_rms_prop= False,  verbose=1, seed = 42)
model.learn(total_timesteps=steps)

# Get all episode rewards in a list
ep = env.get_episode_rewards()
