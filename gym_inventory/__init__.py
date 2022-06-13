import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Inventory-v0',
    entry_point='gym_inventory.envs:InventoryEnv',
    max_episode_steps=90,
    reward_threshold=1.0,
    nondeterministic = True,
)
