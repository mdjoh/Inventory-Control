# Train PPO algorithm to find optimal policy for inventory control problem

import gym
import torch
from gym_inventory.envs.inventory_env import InventoryEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import time
import numpy as np

# ----------------------------------SETUP------------------------------------------------

# Make inventory control environment
env = gym.make('Inventory-v0')

# Set Constants
# N.B. the length of each episode is 90 timesteps
n_timesteps = 500000 # number of timesteps for training
n_episodes = 500 # number of business quarters to make predictions for
name = f"PPO_{n_timesteps}_{n_episodes}_{int(time.time())}" # set monitor csv file and tensorboard log name; append time to name to ensure the name is unique

# Pass in created environment to monitor results
env = Monitor(env, f"./{name}") # a csv file with the monitor results will be generated in the current working directory

# Define neural network architecture to use for model training
# pi: actor, vf: value function
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
					 net_arch=[dict(pi=[128, 128], vf=[128, 128])])


# ----------------------------------TRAINING------------------------------------------------

# Train PPO algorithm to learn optimal inventory control policy
model = PPO(policy='MlpPolicy', env=env, batch_size=128, n_epochs=10, learning_rate=10e-4, gamma=0.8, verbose=1, policy_kwargs=policy_kwargs,
			tensorboard_log="./inventory_ppo_tensorboard/")

model.learn(total_timesteps=n_timesteps, tb_log_name=name, reset_num_timesteps=False) # tensorboard log will display training metrics


# ----------------------------------PREDICTION------------------------------------------------

# Initialize empty lists to append prediction results
predicted_actions = [] # list of all predicted actions (i.e., quantity ordered) from action space
episode_reward = [] # list of reward total for each episode

# Make predictions for each episode
for ep in range(n_episodes):
	state = env.reset()

	predicted_reward = [] # initialize empty list to store all rewards of each episode
	done = False

	while not done:
		# pass state to model to get predicted action
		action, _states = model.predict(state)

		# append predicted action to list
		predicted_actions.append(action)

		# pass action to env to return new state and reward
		state, reward, done, info = env.step(action)

		# append reward at each timestep
		predicted_reward.append(reward)

	# calculate and append total reward of each episode
	episode_reward.append(sum(predicted_reward))


# Plot predicted episode rewards
plt.plot(episode_reward, '.')
plt.ylim([0, 9000])
plt.xlabel('Quarter (Episode)')
plt.ylabel('Predicted Profit ($) Per Quarter')
plt.show()

# Plot predicted actions
plt.plot(predicted_actions, '.')
plt.ylim([0, 100])
plt.yticks(np.arange(0, 101, step=10))
plt.xlabel('Timestep')
plt.ylabel('Quantity Ordered')
plt.show()
