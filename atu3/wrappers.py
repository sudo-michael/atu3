"""Wrapper that tracks the cumulative rewards and episode lengths."""
import time
from collections import deque
from typing import Optional

import numpy as np

import gymnasium as gym

class RecordCollisions(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
        """
        gym.Wrapper.__init__(self, env)

        self.num_envs = getattr(env, "num_envs", 1)
        self.goal_counter = 0
        self.persuer_counter = 0
        self.hj_counter = 0
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)
        return obs, info

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones:
            if infos.get("collision", False) == 'goal':
                self.goal_counter += 1
            elif infos.get("collision", False) == 'persuer':
                self.persuer_counter += 1
            
            if infos.get("used_hj", False):
                self.hj_counter += 1

            if "episode" in infos or "_episode" in infos:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                infos["counter"] = {
                    "goal": self.goal_counter,
                    "persuer": self.persuer_counter,
                    "hj": self.hj_counter,
                }

                if self.is_vector_env:
                    infos["_episode"] = np.where(dones, True, False)
        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )