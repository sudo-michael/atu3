from __future__ import annotations

import random
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import Box
import datetime
from safety_gymnasium.wrappers import Gymnasium2SafetyGymnasium


@env_register
class OmniSafeAir3DEnv(CMDP):
    """Gymnasium Mujoco environment.

    Attributes:
        _support_envs (list[str]): List of supported environments.
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    _support_envs = [
        'Safe-Air3D-v0',
    ]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    need_action_repeat_wrapper = False

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = 'cpu',
        **kwargs: Any,
    ) -> None:
        """Initialize the environment.

        Args:
            env_id (str): Environment id.
            num_envs (int, optional): Number of environments. Defaults to 1.
            device (torch.device, optional): Device to store the data. Defaults to 'cpu'.
            **kwargs: Other arguments.
        """
        super().__init__(env_id)
        self._env_id = env_id
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        if num_envs == 1:
            # set healthy_reward=0.0 for removing the safety constraint in reward
            env = gymnasium.make(id=env_id, render_mode="rgb_array", **kwargs)
            env = gymnasium.wrappers.RecordVideo(env, f"videos/{now}")
            self._env = Gymnasium2SafetyGymnasium(env)
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        else:
            raise NotImplementedError('Only support num_envs=1 now.')
        self._device = torch.device(device)

        self._num_envs = num_envs
        self._metadata = self._env.metadata

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment.

        .. note::

            OmniSafe use auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode.
            And the true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )

        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            seed (int, optional): Seed to reset the environment. Defaults to None.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = self._env.reset(seed=seed)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def sample_action(self) -> torch.Tensor:
        """Sample a random action.

        Returns:
            torch.Tensor: A random action.
        """
        return torch.as_tensor(
            self._env.action_space.sample(),
            dtype=torch.float32,
            device=self._device,
        )

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Any: Rendered environment.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()