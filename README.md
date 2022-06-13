# MMAI 845 Final Project - Inventory Control

## Problem
The final project environment is the inventory environment. This is an environment for inventory control problems, which can be encountered in commercial settings.
Inventory control needs to be optimized in order to maximize profit by having sufficient product available to sell—thereby also reducing lost sales—and minimizing unsold product.

## Environment Details
In the inventory environment, there is one agent present with discrete state and action spaces.

Within the inventory environment, the following reinforcement learning terms are defined as such:

**Agent** – inventory manager that makes decisions regarding inventory

**State space** – set of discrete states that indicate the inventory level; the size of the state space is n+1 (where n is the maximum inventory level)

**Action space** – set of discrete actions the agent can take regarding the inventory; the possible actions are increase inventory by ordering more product and maintain current inventory quantity; the size of the action space is n+1

**Reward** – profit achieved

**Episode** - duration used to calculate the profit achieved; we defined it as one fiscal quarter (90 days)

**Terminal state** - the end of one fiscal quarter

## Reward Function
The reward function is comprised of the following components:
- Fixed order cost
- Cost per ordered unit
- Holding cost per unit held in inventory
- Sales revenue

The sale price, cost per unit, and holding cost are greater than 0.

The reward function is:

`r = -k * (a > 0) - c * max(min(x + a, n) - x, 0) - h * x + p * max(min(x + a, n) - y, 0)`

where:
- `r`: reward
- `k`: fixed order cost when the inventory quantity ordered is greater than 0; we set it as $5
- `a`: inventory quantity ordered
- `c`: cost per unit; we set it as $7
- `x`: current inventory quantity
- `n`: maximum inventory level; we set it as 100 units
- `h`: holding cost per unit; we set it as $2
- `p`: sale price per unit; we set it as $20
- `y`: updated inventory quantity

## Results
Our best model was a two-layered neural network with 128 nodes in each layer that trained with the PPO algorithm for 500,000 timesteps. It converged to a reward of approximately $8,000 per episode.

![Reward vs Timesteps when training timesteps vary](/Figures/timestep_presentation.png)
***Figure 1.*** Reward progression for models trained with various number of timesteps

Training a model with two layers of 128 nodes for 100,000 timesteps was not enough as it did not lead to reward convergence. The reward converged when training occurred for 200,000 timesteps but at a slightly lesser reward than the model that was trained for 500,000 timesteps.

![Reward vs Timesteps when network architecture varies](/Figures/nn_presentation.png)
***Figure 2.*** Reward progression for models trained with various network architectures

Among the neural network architectures we tried, the two-layered network with 128 nodes in each layer converged the earliest.

## Code in this Repository
Core gym-inventory package files are found in the `gym_inventory` folder.

Executable files and their description are as follows:\
`inventory_control_random.py`: prints the starting and resulting states, action taken, demand, and reward when random actions are taken

`inventory_control_A2Ctrain.py`: trains an A2C algorithm to find an optimal inventory control policy and outputs the reward of each training episode to a csv file via the Stable Baselines3 Monitor

`inventory_control_PPOtrain.py`: trains an PPO algorithm to find an optimal inventory control policy and generates a Tensorboard log for training and outputs the reward of each training and prediction episode to a csv file via the Stable Baselines3 Monitor

To run our code, gym-inventory must be installed.
Installation and run instructions are in our [RUNME](RUNME.md).
