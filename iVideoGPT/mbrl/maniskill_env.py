# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from typing import Any, NamedTuple

import cv2
import numpy as np

import dm_env
from dm_env import StepType, specs
import torch
from omegaconf import OmegaConf

from metaworld_env import (
    ActionDTypeWrapper,
    ActionScaleWrapper,
    ExtendedTimeStepWrapper,
    FrameStackWrapper,
)
from rlinf.envs.maniskill.maniskill_env import ManiskillEnv as RLinfManiskillEnv

os.environ.setdefault("SAPIEN_LOG_LEVEL", "error")
logger = logging.getLogger(__name__)
logging.getLogger("mani_skill").setLevel(logging.ERROR)


class ManiSkillTimeStep(NamedTuple):
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


class ManiSkill(dm_env.Environment):
    """Single-task ManiSkill environment with a dm_env interface."""

    def __init__(
        self,
        task_name: str,
        seed: int,
        action_repeat: int = 1,
        obs_mode: str = "rgb",
        control_mode: str | None = None,
        sim_backend: str = "gpu",
        sim_freq: int = 500,
        control_freq: int = 5,
        max_episode_steps: int = 200,
        obj_set: str = "train",
        render_mode: str = "all",
        shader_pack: str = "default",
        size: tuple[int, int] = (64, 64),
    ):
        cfg = OmegaConf.create(
            {
                "auto_reset": False,
                "use_rel_reward": False,
                "ignore_terminations": False,
                "seed": seed,
                "num_group": 1,
                "group_size": 1,
                "use_fixed_reset_state_ids": False,
                "video_cfg": {
                    "save_video": False,
                    "info_on_video": False,
                    "video_base_dir": "",
                },
                "init_params": {
                    "id": task_name,
                    "num_envs": 1,
                    "obs_mode": obs_mode,
                    "control_mode": control_mode,
                    "sim_backend": sim_backend,
                    "sim_config": {"sim_freq": sim_freq, "control_freq": control_freq},
                    "max_episode_steps": max_episode_steps,
                    "sensor_configs": {"shader_pack": shader_pack},
                    "render_mode": render_mode,
                    "obj_set": obj_set,
                },
            }
        )

        self._env = RLinfManiskillEnv(
            cfg, seed_offset=0, total_num_processes=1, record_metrics=True
        )
        self._env.is_start = False

        self._seed = seed
        self._reset_seed = seed
        self._action_repeat = action_repeat
        self._done = True
        self._obs_spec = None
        self._last_obs = None
        self._size = size

        action_space = self._env.env.action_space
        self._batched_action = len(action_space.shape) == 2
        if self._batched_action:
            low = action_space.low[0]
            high = action_space.high[0]
        else:
            low = action_space.low
            high = action_space.high
        self._action_low = np.array(low, dtype=np.float32)
        self._action_high = np.array(high, dtype=np.float32)
        self._action_spec = specs.BoundedArray(
            shape=self._action_low.shape,
            dtype=np.float32,
            minimum=self._action_low,
            maximum=self._action_high,
            name="action",
        )

    def _format_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self._batched_action:
            if action.ndim == 1:
                action = np.expand_dims(action, axis=0)
        return action

    def _extract_image(self, obs) -> np.ndarray:
        frame = obs["images"][0]
        frame = _to_numpy(frame).astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.resize(frame, (self._size[1], self._size[0]))
        return frame

    def _get_success(self, infos) -> float:
        if "success" not in infos:
            return 0.0
        success = _to_numpy(infos["success"])[0]
        return float(success)

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

    def reset(self):
        raw_obs, infos = self._env.reset(seed=self._reset_seed)
        self._reset_seed = None
        self._env.is_start = False
        image = self._extract_image(raw_obs)
        self._last_obs = image
        self._done = False
        success = self._get_success(infos)
        return ManiSkillTimeStep(
            observation=image,
            step_type=dm_env.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            success=success,
            state=None,
        )

    def step(self, action):
        assert not self._done, "Must reset environment."
        total_reward = 0.0
        success = 0.0
        discount = 1.0
        done = False

        for _ in range(self._action_repeat):
            env_action = self._format_action(action)
            obs, reward, terminations, truncations, infos = self._env.step(
                env_action, auto_reset=False
            )
            if reward is None:
                reward_value = 0.0
            else:
                reward_value = float(_to_numpy(reward)[0])
            total_reward += reward_value
            success = max(success, self._get_success(infos))
            done = bool(
                np.asarray(_to_numpy(terminations)).any()
                or np.asarray(_to_numpy(truncations)).any()
            )
            self._last_obs = self._extract_image(obs)
            if done:
                break

        if done:
            self._done = True
            step_type = dm_env.StepType.LAST
        else:
            step_type = dm_env.StepType.MID

        return ManiSkillTimeStep(
            observation=self._last_obs,
            step_type=step_type,
            reward=total_reward,
            discount=discount,
            success=success,
            state=None,
        )

    def render(self):
        if hasattr(self._env, "capture_image"):
            image = self._env.capture_image()
            if isinstance(image, np.ndarray):
                return image
        if self._last_obs is not None:
            return self._last_obs
        if self._obs_spec is not None:
            return np.zeros((self._obs_spec.shape[1], self._obs_spec.shape[2], 3), dtype=np.uint8)
        return np.zeros((64, 64, 3), dtype=np.uint8)


def make(
    name,
    frame_stack,
    action_repeat,
    seed,
    obs_mode,
    control_mode,
    sim_backend,
    sim_freq,
    control_freq,
    max_episode_steps,
    obj_set,
    render_mode,
    shader_pack,
    size=(64, 64),
):
    def _build_env(backend):
        return ManiSkill(
            name,
            seed=seed,
            action_repeat=action_repeat,
            obs_mode=obs_mode,
            control_mode=control_mode,
            sim_backend=backend,
            sim_freq=sim_freq,
            control_freq=control_freq,
            max_episode_steps=max_episode_steps,
            obj_set=obj_set,
            render_mode=render_mode,
            shader_pack=shader_pack,
            size=size,
        )

    try:
        env = _build_env(sim_backend)
    except RuntimeError as err:
        msg = str(err)
        if "PhysxGpuSystem" in msg or "CUDA failed" in msg:
            logger.info("Falling back to PhysX CPU backend due to GPU init failure.")
            env = _build_env("physx_cpu")
        else:
            raise
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionScaleWrapper(env, minimum=np.array(-1.0, dtype=np.float32), maximum=np.array(+1.0, dtype=np.float32))
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
