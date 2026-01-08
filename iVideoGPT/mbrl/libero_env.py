import logging
from typing import Any, NamedTuple

import cv2
import numpy as np
import torch
import dm_env
from dm_env import StepType, specs
from omegaconf import OmegaConf

from metaworld_env import (
    ActionDTypeWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
)
from rlinf.envs.libero.libero_env import LiberoEnv as RLinfLiberoEnv

logger = logging.getLogger(__name__)


class LiberoTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    success: Any
    state: Any = None

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class Libero(dm_env.Environment):
    """Single-task LIBERO environment with a dm_env interface."""

    def __init__(self, cfg, seed, action_repeat=1, size=(64, 64)):
        env_cfg = OmegaConf.create(cfg)
        env_cfg.seed = seed
        env_cfg.num_envs = 1
        env_cfg.num_group = 1
        env_cfg.group_size = 1

        self._env = RLinfLiberoEnv(env_cfg, seed_offset=0, total_num_processes=1)
        self._env.is_start = False
        self._size = size
        self._action_repeat = action_repeat
        self._done = True
        self._obs_spec = None
        self._last_obs = None
        self._use_wrist = bool(env_cfg.get("use_wrist_image", False))

        action_dim = int(env_cfg.get("action_dim", 7))
        self._action_spec = specs.BoundedArray(
            shape=(action_dim,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )

    def observation_spec(self):
        if self._obs_spec is None:
            time_step = self.reset()
            self._obs_spec = specs.BoundedArray(
                shape=time_step.observation.shape,
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name="observation",
            )
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def _resize_image(self, image):
        image = image.astype(np.uint8)
        return cv2.resize(image, (self._size[1], self._size[0]))

    def _extract_policy_obs(self, obs):
        images_and_states = obs["images_and_states"]
        full_image = _to_numpy(images_and_states["full_image"][0])
        state = _to_numpy(images_and_states["state"][0]).astype(np.float32)
        task_desc = obs["task_descriptions"][0]

        policy_obs = {
            "images": torch.as_tensor(full_image).permute(2, 0, 1).unsqueeze(0),
            "states": torch.as_tensor(state, dtype=torch.float32).unsqueeze(0),
            "task_descriptions": [task_desc],
        }
        if self._use_wrist and "wrist_image" in images_and_states:
            wrist_image = _to_numpy(images_and_states["wrist_image"][0])
            policy_obs["wrist_images"] = (
                torch.as_tensor(wrist_image).permute(2, 0, 1).unsqueeze(0)
            )
        return policy_obs, full_image

    def reset(self, seed=None):
        raw_obs, _infos = self._env.reset()
        policy_obs, full_image = self._extract_policy_obs(raw_obs)
        image = self._resize_image(full_image)
        self._last_obs = image
        self._done = False
        return LiberoTimeStep(
            observation=image,
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            success=0.0,
            state=policy_obs,
        )

    def step(self, action):
        assert not self._done, "Must reset environment."
        total_reward = 0.0
        discount = 1.0
        success = 0.0
        done = False
        info_obs = None

        for _ in range(self._action_repeat):
            env_action = np.asarray(action, dtype=np.float32)
            obs, reward, terminations, truncations, infos = self._env.step(env_action)
            info_obs = obs
            reward_value = float(_to_numpy(reward)[0]) if reward is not None else 0.0
            total_reward += reward_value
            done = bool(_to_numpy(terminations)[0] or _to_numpy(truncations)[0])
            if done:
                success = float(_to_numpy(terminations)[0])
                break

        policy_obs, full_image = self._extract_policy_obs(info_obs)
        self._last_obs = self._resize_image(full_image)

        if done:
            self._done = True
            step_type = StepType.LAST
            discount = 0.0
        else:
            step_type = StepType.MID

        return LiberoTimeStep(
            observation=self._last_obs,
            step_type=step_type,
            reward=total_reward,
            discount=discount,
            success=success,
            state=policy_obs,
        )


def make(cfg, frame_stack, action_repeat, seed, image_size=(64, 64)):
    env = Libero(cfg, seed=seed, action_repeat=action_repeat, size=image_size)
    env = ActionDTypeWrapper(env, np.float32)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
