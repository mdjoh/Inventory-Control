import numpy
import torch
import gym
from gym_inventory.envs.inventory_env import InventoryEnv
from stable_baselines3 import PPO

numpy.random.seed(42)

env = gym.make('Inventory-v0')

env.reset()

# define model to run in inventory control environment
model = PPO('MlpPolicy', env, verbose=1, seed = 42, policy_kwargs= dict(net_arch = [32,32]))
model.learn(total_timesteps=150000)

env.reset()

episodes = 10



for ep in range(episodes):
	
	reward_list = []

	obs = env.reset()

	done = False
	
	while not done:
        # pass observation to model to get predicted action
		action, _states = model.predict(obs)
		
		#print('current stock : ', obs)

        # pass action to env and get info back
		obs, rewards, done, info = env.step(action)
		
		#print('tomorrow stock: ', obs,' - reward : ', rewards,  ' - action : ', action)

		reward_list.append(rewards)

	print('cumulative reward for episode ',ep,' : ', sum(reward_list))



#print(len(obs_list))

