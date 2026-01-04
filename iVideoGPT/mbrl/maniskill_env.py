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
import gymnasium as gym
from dm_env import StepType, specs
import torch
from omegaconf import OmegaConf
import mani_skill.envs  # noqa: F401

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

RLINF_TASK_PREFIXES = ("PutOn", "PutCarrot")


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


def _use_rlinf_backend(task_name: str) -> bool:
    return any(task_name.startswith(prefix) for prefix in RLINF_TASK_PREFIXES)


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
        render_backend: str | None = "gpu",
        size: tuple[int, int] = (64, 64),
        use_rlinf: bool = True,
    ):
        self._use_rlinf = use_rlinf
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
                    "render_backend": render_backend,
                },
            }
        )
        if obj_set is not None:
            cfg.init_params["obj_set"] = obj_set

        if self._use_rlinf:
            self._env = RLinfManiskillEnv(
                cfg, seed_offset=0, total_num_processes=1, record_metrics=True
            )
            self._env.is_start = False
            action_space = self._env.env.action_space
        else:
            env_kwargs = {
                "num_envs": 1,
                "obs_mode": obs_mode,
                "control_mode": control_mode,
                "render_mode": render_mode,
                "sim_backend": sim_backend,
            }
            env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}
            self._env = gym.make(task_name, **env_kwargs)
            action_space = self._env.action_space

        self._seed = seed
        self._reset_seed = seed
        self._action_repeat = action_repeat
        self._done = True
        self._obs_spec = None
        self._last_obs = None
        self._size = size

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
        # Prefer the official render view to keep a consistent camera across eval/imag/validate GIFs.
        frame = None
        if hasattr(self._env, "render"):
            try:
                frame = self._env.render()
            except Exception:
                frame = None
            if frame is not None:
                frame = _to_numpy(frame)
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]

        # Fall back to observation cameras if render is unavailable.
        if frame is None:
            if self._use_rlinf:
                frame = obs["images"][0]
                frame = _to_numpy(frame)
                frame = np.transpose(frame, (1, 2, 0))
            else:
                sensor_data = obs.get("sensor_data", {})
                camera = None
                # Prefer official third-view camera if available; otherwise base camera.
                if "3rd_view_camera" in sensor_data:
                    camera = sensor_data.get("3rd_view_camera")
                if camera is None:
                    camera = sensor_data.get("base_camera")
                if camera is None and sensor_data:
                    camera = next(iter(sensor_data.values()))
                if camera is None:
                    raise KeyError("No camera image found in ManiSkill observation.")
                frame = camera["rgb"]
                frame = _to_numpy(frame)
                if frame.ndim == 4:
                    frame = frame[0]
        frame = frame.astype(np.uint8)
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

    def reset(self, seed=None):
        if seed is not None:
            self._reset_seed = seed
        if self._use_rlinf:
            raw_obs, infos = self._env.reset(seed=self._reset_seed)
            self._env.is_start = False
        else:
            raw_obs, infos = self._env.reset(seed=self._reset_seed)
        self._reset_seed = None
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
            if self._use_rlinf:
                obs, reward, terminations, truncations, infos = self._env.step(
                    env_action, auto_reset=False
                )
            else:
                obs, reward, terminations, truncations, infos = self._env.step(env_action)
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
            discount = 0.0
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
        if hasattr(self._env, "render"):
            image = self._env.render()
            if image is not None:
                image = _to_numpy(image)
                if image.ndim == 4 and image.shape[0] == 1:
                    image = image[0]
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
    render_backend,
    size=(64, 64),
):
    use_rlinf = _use_rlinf_backend(name)

    def _build_env(backend, render_backend_override=None):
        return ManiSkill(
            name,
            seed=seed,
            action_repeat=action_repeat,
            obs_mode=obs_mode,
            control_mode=control_mode,
            sim_backend=backend,
            render_backend=render_backend_override or render_backend,
            sim_freq=sim_freq,
            control_freq=control_freq,
            max_episode_steps=max_episode_steps,
            obj_set=obj_set,
            render_mode=render_mode,
            shader_pack=shader_pack,
            size=size,
            use_rlinf=use_rlinf,
        )

    try:
        env = _build_env(sim_backend)
    except RuntimeError as err:
        msg = str(err).lower()
        if (
            "physxgpusystem" in msg
            or "cuda failed" in msg
            or "sapien-vulkan" in msg
            or "vulkan" in msg
        ):
            logger.info(
                "Falling back to PhysX CPU + CPU renderer due to GPU/Vulkan init failure."
            )
            env = _build_env("physx_cpu", render_backend_override="sapien_cpu")
        else:
            raise
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionScaleWrapper(env, minimum=np.array(-1.0, dtype=np.float32), maximum=np.array(+1.0, dtype=np.float32))
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
