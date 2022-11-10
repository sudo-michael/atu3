# %%
import atu3
import gym
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def make_env():
    def funct():
        env = gym.make('Safe-GoalAir3d-NoWalls-v0')
        return env
    return funct
# envs = gym.vector.SyncVectorEnv(
#     [make_env() for _ in range(1)]
# )
envs = DummyVecEnv([make_env()])
# %%
envs.reset()

# env = gym.make('Safe-Air3d-NoWalls-v1')
# env = gym.vector.make('Safe-Air3d-NoWalls-v1', num_envs=3)
# print(env.reset())
obs = envs.reset()
# use 'future;' goal selctino strategy since it worked the best in HER
# n_sample_goal
rb = DictReplayBuffer(
    1_5000,
    envs.observation_space,
    envs.action_space,
    handle_timeout_termination=True,
)

hrb = HerReplayBuffer(
    envs, replay_buffer=rb, buffer_size=1500, online_sampling=False
)
for _ in range(2_000):
    action = np.array([1.0]).reshape((1,1))
    next_obs, rewards, done, info = envs.step(action)

    hrb.add(obs, next_obs, action.flatten(), rewards, done, info)

    obs = next_obs

batch  = hrb.sample(4)

# obs, infos = envs.reset()
# %%
