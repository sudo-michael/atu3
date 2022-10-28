from gym.envs.registration import register
import numpy as np

register(
    id="Safe-Air3d-v0",
    entry_point="atu3.envs:Air3dEnv",
    max_episode_steps=400,
    kwargs={'fixed_goal' : False}
)

register(
    id="Safe-Air3d-Fixed-v0",
    entry_point="atu3.envs:Air3dEnv",
    max_episode_steps=400,
    kwargs={'fixed_goal' :True}
)