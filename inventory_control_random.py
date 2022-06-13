# Take random actions and see if it an optimal policy to maximize profit is found

from gym_inventory.envs.inventory_env import InventoryEnv
import matplotlib.pyplot as plt
import pandas as pd

# Make inventory control environment
env = InventoryEnv()

# Make action predictions and compute the corresponding rewards for 500 episodes (i.e., business quarters)
n_episodes = 500
episode_timesteps = 90 # each episode is 90 timesteps

# Initialize empty list to append episode rewards
episode_reward = []

for ep in range(n_episodes):
    
    predicted_reward = [] # initialize empty list to store all rewards of each episode

    for step in range(episode_timesteps):
        obs, obs2, reward, action, demand, done, info = env.random_step(env.action_space.sample())
        
        predicted_reward.append(reward)

        # Print daily status
        print('Stock @ EOD-yesterday : ', obs,' - Replenishment : ' , action,' - Demand : ', demand,  ' - Reward : ', reward, ' - Stock @ EOD today : ', obs2)

    episode_reward.append(sum(predicted_reward))

env.close()

# Write predicted episode rewards to csv
df_rewards = pd.DataFrame(episode_reward, columns=['Reward'])
df_rewards.to_csv('episode_rewards_random.csv', index=False)

# Plot predicted episode rewards
plt.plot(episode_reward, '.')
plt.xlabel('Quarter (Episode)')
plt.ylabel('Predicted Profit ($) Per Quarter')
plt.show()