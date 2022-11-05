from gym.envs.registration import register
import numpy as np

register(
    id="Safe-Air3d-v0",
    entry_point="atu3.envs:Air3dEnv",
    max_episode_steps=400,
    kwargs={'fixed_goal' : False, 'walls': True, 'version': 1}
)

register(
    id="Safe-Air3d-NoWalls-v0",
    entry_point="atu3.envs:Air3dEnv",
    max_episode_steps=1_000,
    kwargs={'fixed_goal' : False, 'walls': False, 'version': 1}
)

register(
    id="Safe-Air3d-NoWalls-v0",
    entry_point="atu3.envs:Air3dEnv",
    max_episode_steps=1_000,
    kwargs={'fixed_goal' : False, 'walls': False, 'version': 2}
)

register(
    id="Safe-Air3d-Fixed-v0",
    entry_point="atu3.envs:Air3dEnv",
    max_episode_steps=400,
    kwargs={'fixed_goal' :True, 'walls': True}
)
