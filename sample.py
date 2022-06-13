import gym

from gym_inventory.envs.inventory_env import InventoryEnv

env = gym.make('Inventory-v0')
env.reset()

for step in range(20):
    obs, reward, done, info = env.step(env.action_space.sample())
    print('Stock @ EOD-yesterday : ',obs, ' - Reward : ', reward)

env.close()