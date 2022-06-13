import gym

from gym_inventory.envs.inventory_env import InventoryEnv

env = InventoryEnv()

#env.reset()

print('sample action : ', env.action_space.sample())

# observation space shape:
print("observation space shape:", env.observation_space.shape)

# sample observation:
print("sample observation:", env.observation_space.sample())

env.close()