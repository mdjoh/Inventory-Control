# MMAI 845 Reinforcement Learning and Its Applications
# Final Project - Inventory Control

## Problem
The final project environment is the inventory environment. This is an environment for inventory control problems, which can be encountered in commercial settings.
Inventory control needs to be optimized in order to maximize profit by having sufficient product available to sell—thereby also reducing lost sales—and minimizing unsold product.

## Environment Details
In the inventory environment, there is one agent present with discrete state and action spaces.

Within the inventory environment, the following reinforcement learning terms are defined as such:

**Agent** – inventory manager that makes decisions regarding inventory

**State space** – set of discrete states that indicate the inventory level

**Action space** – set of discrete actions the agent can take regarding the inventory; the actions are increase inventory by ordering more product and maintain current inventory quantity

**Reward** – profit achieved

## Reward Function
The reward function is comprised of the following components:
- Fixed order cost
- Cost per ordered unit
- Holding cost per unit held in inventory
- Sales revenue

The sale price, cost per unit, and holding cost are greater than 0.

The reward function is written as:

`r = -k * (a > 0) - c * max(min(x + a, m) - x, 0) - h * x + p * max(min(x + a, m) - y, 0)`

where:

- `r`:reward
- `k`: fixed order cost when the inventory quantity ordered is greater than 0
- `a`: inventory quantity ordered
- `c`: cost per unit
- `x`: current inventory quantity
- `m`: maximum inventory level
- `h`: holding cost per unit
- `p`: sale price per unit
- `y`: updated inventory quantity 
