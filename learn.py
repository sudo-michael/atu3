# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mambaforge/lib
import omnisafe
import safety_gymnasium
import gymnasium as gym
import atu3

from atu3.envs.omnisafe_air_3d_Np import OmniSafeAir3DEnv
env_id = "Safe-Air6D-v0"

custom_cfgs = {
    'train_cfgs': {
        'total_steps': 204_800,
        'vector_env_nums': 1,
        'parallel': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 2048,
        'update_iters': 1,
    },
    'logger_cfgs': {
        'use_wandb': True,
        'wandb_project': 'atu3',
    },
    'lagrange_cfgs': {
        'cost_limit': 1.0
    },
}
agent = omnisafe.Agent('SACLag', env_id, custom_cfgs=custom_cfgs)
agent.learn()
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mambaforge/lib