# %%
import atu3
import gymnasium as gym
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

global_step=978171, episodic_return=2.7846999168395996 collision=goal
global_step=979171, episodic_return=0.5993618369102478 collision=timeout
global_step=979377, episodic_return=-8.796401023864746 collision=timeout
global_step=979503, episodic_return=5.420310974121094 collision=goal
global_step=979733, episodic_return=2.9543190002441406 collision=goal
global_step=980061, episodic_return=1.8298001289367676 collision=goal
global_step=980339, episodic_return=7.481004238128662 collision=goal
global_step=980813, episodic_return=5.915885925292969 collision=goal
global_step=980965, episodic_return=1.4654431343078613 collision=goal
global_step=981115, episodic_return=2.945199489593506 collision=goal
global_step=981462, episodic_return=2.3186841011047363 collision=goal
global_step=981724, episodic_return=6.704771041870117 collision=goal
global_step=981889, episodic_return=-5.630126476287842 collision=timeout
global_step=982463, episodic_return=1.8185900449752808 collision=goal
global_step=982646, episodic_return=3.3617024421691895 collision=goal
global_step=983256, episodic_return=4.965806484222412 collision=goal
global_step=983513, episodic_return=3.7146072387695312 collision=goal
global_step=984513, episodic_return=-6.626838684082031 collision=timeout
global_step=984674, episodic_return=8.453025817871094 collision=goal
global_step=984827, episodic_return=5.4756245613098145 collision=goal
global_step=985127, episodic_return=7.166014671325684 collision=goal
global_step=985857, episodic_return=4.956247329711914 collision=goal
global_step=986754, episodic_return=-10.722494125366211 collision=timeout
global_step=987164, episodic_return=4.998676300048828 collision=goal
global_step=987341, episodic_return=-6.125448703765869 collision=timeout
global_step=987803, episodic_return=3.30322527885437 collision=goal
global_step=988063, episodic_return=6.176459312438965 collision=goal
global_step=988589, episodic_return=6.830019950866699 collision=goal
global_step=989014, episodic_return=6.698606967926025 collision=goal
global_step=989145, episodic_return=-4.8132500648498535 collision=timeout
global_step=989440, episodic_return=4.113499641418457 collision=goal
global_step=989800, episodic_return=2.976330518722534 collision=goal
global_step=990148, episodic_return=-6.738757133483887 collision=timeout
global_step=990280, episodic_return=1.7946442365646362 collision=goal
global_step=990659, episodic_return=3.300304889678955 collision=goal
global_step=990726, episodic_return=4.102163314819336 collision=goal
global_step=990921, episodic_return=4.880030632019043 collision=goal
global_step=991129, episodic_return=5.810039520263672 collision=goal
global_step=991393, episodic_return=6.780701160430908 collision=goal
global_step=991424, episodic_return=2.4057703018188477 collision=goal
global_step=991719, episodic_return=7.170440196990967 collision=goal
global_step=991907, episodic_return=4.9603776931762695 collision=goal
global_step=992211, episodic_return=3.7497329711914062 collision=goal
global_step=992611, episodic_return=3.327152729034424 collision=goal
global_step=993196, episodic_return=5.565140247344971 collision=goal
global_step=993344, episodic_return=4.191956996917725 collision=goal
global_step=993857, episodic_return=4.417708396911621 collision=goal
global_step=994062, episodic_return=-9.204710960388184 collision=timeout
global_step=994265, episodic_return=-4.632760047912598 collision=timeout
global_step=994329, episodic_return=4.0864033699035645 collision=goal
global_step=994477, episodic_return=4.465582847595215 collision=goal
global_step=994946, episodic_return=3.4303133487701416 collision=goal
global_step=995436, episodic_return=3.0934033393859863 collision=goal
global_step=995557, episodic_return=1.6070036888122559 collision=goal
global_step=995775, episodic_return=-7.923125743865967 collision=timeout
global_step=996163, episodic_return=7.208039283752441 collision=goal
global_step=996552, episodic_return=-9.468781471252441 collision=timeout
global_step=996729, episodic_return=-7.170980930328369 collision=timeout
global_step=996810, episodic_return=2.0489466190338135 collision=goal
global_step=997010, episodic_return=2.9666264057159424 collision=goal
global_step=997185, episodic_return=5.543075084686279 collision=goal
global_step=997644, episodic_return=3.2903616428375244 collision=goal
global_step=997839, episodic_return=6.061570644378662 collision=goal
global_step=997979, episodic_return=6.898174285888672 collision=goal
global_step=998130, episodic_return=4.938259124755859 collision=goal
global_step=998564, episodic_return=4.194055557250977 collision=goal
global_step=999564, episodic_return=1.4880218505859375 collision=timeout
global_step=999947, episodic_return=4.970484733581543 collision=goal