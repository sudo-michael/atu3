from gymnasium.envs.registration import register
import numpy as np

# register(
#     id="Safe-Air3d-Fixed-Goal-v0",
#     entry_point="atu3.envs:Air3dEnv",
#     max_episode_steps=400,
#     kwargs={"fixed_goal": True, "version": 0},
# )

# register(
#     id="Safe-StaticAir3d-v0",
#     entry_point="atu3.envs:StaticAir3dEnv",
#     max_episode_steps=450,
# )

# register(
#     id="Safe-StaticAir6d-v0",
#     entry_point="atu3.envs:StaticAirNdEnv",
#     kwargs={"n": 2},
#     max_episode_steps=450,
# )

# register(
#     id="Safe-StaticAir9D-v0",
#     entry_point="atu3.envs:StaticAirNdEnv",
#     kwargs={"n": 2},
#     max_episode_steps=450,
# )

register(
    id="Safe-Air3D-v0",
    entry_point="atu3.envs:Air3DNpEnv",
    kwargs={"n": 1, "use_hj": True},
    max_episode_steps=450,
)

# register(
#     id="Safe-Air9D-v0",
#     entry_point="atu3.envs:Air3DNpEnv",
#     kwargs={"n": 2},
#     max_episode_steps=450,
# )

# register(
#     id="Safe-GoalAir3d-NoWalls-v0",
#     entry_point="atu3.envs:GoalAir3dEnv",
#     max_episode_steps=1_000,
#     kwargs={'fixed_goal' : False, 'walls': False, 'version': 1}
# )

# register(
#     id="Safe-GoalAir3d-NoWalls-v1",
#     entry_point="atu3.envs:GoalAir3dEnv",
#     max_episode_steps=1_000,
#     kwargs={'fixed_goal' : False, 'walls': False, 'version': 2}
# )


# register(
#     id="Safe-StaticAir3d-NoWalls-v0",
#     entry_point="atu3.envs:StaticAir3dEnv",
#     max_episode_steps=1_000,
#     kwargs={'fixed_goal' : False, 'walls': False, 'version': 1}
# )

# register(
#     id="Safe-Air3d-2p-NoWalls-Fixed-v1",
#     entry_point="atu3.envs:Air3dNpEnv",
#     max_episode_steps=1_000,
#     kwargs={'fixed_goal' : True, 'walls': False, 'version': 2, 'n_persuers': 2}
# )

# register(
#     id="Safe-Air6d-NoWalls-Fixed-v0",
#     entry_point="atu3.envs:Air6dEnv",
#     max_episode_steps=1_000,
#     kwargs={'fixed_goal' : True, 'walls': False, 'version': 1}
# )
