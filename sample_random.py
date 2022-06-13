import gym
import numpy

from gym_inventory.envs.inventory_env import InventoryEnv

numpy.random.seed(42)

env = gym.make('Inventory-v0')
env.reset()

for ep in range(10):
	
	reward_list = []

	obs = env.reset()

	done = False
	
	while not done:
        # pass observation to model to get predicted action
		action = 8
		
		#print('current stock : ', obs)

        # pass action to env and get info back
		obs, rewards, done, info = env.step(action)
		
		#print('tomorrow stock: ', obs,' - reward : ', rewards,  ' - action : ', action)

		reward_list.append(rewards)

	print('cumulative reward for episode ',ep,' : ', sum(reward_list))