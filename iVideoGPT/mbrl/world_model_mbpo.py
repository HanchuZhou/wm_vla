import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from dm_env import StepType, specs
from tqdm import tqdm

from iVideoGPT.mbrl.replay_buffer import (
    ReplayBufferStorage,
    make_replay_loader,
    make_segment_replay_loader,
)
from iVideoGPT.mbrl.video_predictor import VideoPredictor
from rlinf.data.io_struct import EmbodiedRolloutResult
from rlinf.models import get_model


@dataclass
class _SimpleTimeStep:
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


@dataclass
class _GenerationContext:
    device: str
    video_predictor: VideoPredictor
    policy_model: Any


class WorldModelMBPO:
    def __init__(
        self,
        cfg,
        work_dir: str | os.PathLike,
        env_world_size: int,
        stage_num: int,
        num_envs_per_stage: int,
        device: str = "cuda:0",
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.frame_stack = int(getattr(cfg, "frame_stack", 3))
        self.image_size = int(getattr(cfg, "image_size", 64))
        self.action_dim = int(cfg.actor.model.action_dim)
        self.num_action_chunks = int(cfg.actor.model.num_action_chunks)
        self._target_image_size = self._infer_target_image_size()
        self.state_dim = self._infer_state_dim()

        self.env_world_size = env_world_size
        self.stage_num = stage_num
        self.num_envs_per_stage = num_envs_per_stage
        self.total_envs = env_world_size * stage_num * num_envs_per_stage

        self._frame_buffers: list[deque] = [
            deque(maxlen=self.frame_stack) for _ in range(self.total_envs)
        ]
        self._env_done = [True for _ in range(self.total_envs)]
        self._task_desc_per_env = [""] * self.total_envs

        self.total_env_steps = 0
        self.init_model = False
        self.init_gen = False
        self.wm_init_update_step = 0

        self._val_psnr = None
        self._val_ssim = None
        self._val_lpips = None

        self._setup_replay_buffers()
        self._setup_world_model()
        self._setup_policy_model()
        self._setup_generation_contexts()

    def _setup_replay_buffers(self) -> None:
        obs_spec = specs.BoundedArray(
            shape=(self.frame_stack * 3, self.image_size, self.image_size),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )
        action_spec = specs.BoundedArray(
            shape=(self.action_dim,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name="action",
        )
        reward_spec = specs.Array((1,), np.float32, "reward")
        discount_spec = specs.Array((1,), np.float32, "discount")
        data_specs = (obs_spec, action_spec, reward_spec, discount_spec)

        buffer_dir = self.work_dir / "buffer"
        buffer_dir.mkdir(parents=True, exist_ok=True)
        self.replay_storage = ReplayBufferStorage(data_specs, buffer_dir)

        demo_path = None
        if getattr(self.cfg, "demo", False):
            demo_path = os.path.join(self.cfg.demo_path_prefix, self.cfg.task_name)
        if demo_path is not None:
            self._has_demo = len(list(Path(demo_path).glob("*.npz"))) > 0
        else:
            self._has_demo = False

        self.replay_loader = make_replay_loader(
            buffer_dir,
            self.cfg.replay_buffer_size,
            int(self.cfg.batch_size * self.cfg.real_ratio),
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            demo_path,
            demo_image_size=self.image_size,
        )
        self._replay_iter = None

        self.seg_replay_loader = make_segment_replay_loader(
            buffer_dir,
            self.cfg.replay_buffer_size,
            self.cfg.world_model.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.gen_horizon + self.cfg.world_model.context_length,
            demo_path,
            demo_image_size=self.image_size,
        )
        self._seg_replay_iter = None

    def _setup_world_model(self) -> None:
        self.video_predictor = VideoPredictor(self.device, self.cfg.world_model)

    def _setup_policy_model(self) -> None:
        self.policy_model = get_model(
            self.cfg.actor.checkpoint_load_path,
            self.cfg.actor.model,
        )
        self.policy_model.eval()
        self.policy_model.to(self.device)

    def _resolve_generation_devices(self) -> list[str]:
        configured = getattr(self.cfg.world_model, "gen_devices", None)
        if configured is None:
            return [self.device]

        if isinstance(configured, str):
            raw_items = [item.strip() for item in configured.split(",") if item.strip()]
        elif isinstance(configured, Iterable):
            raw_items = list(configured)
        else:
            raw_items = [configured]

        devices: list[str] = []
        for item in raw_items:
            if isinstance(item, int):
                device = f"cuda:{item}"
            else:
                device = str(item).strip()
            if device == "":
                continue
            if device not in devices:
                devices.append(device)

        if not devices:
            devices = [self.device]
        if self.device in devices:
            devices.remove(self.device)
        devices.insert(0, self.device)
        return devices

    def _setup_generation_contexts(self) -> None:
        self._generation_devices = self._resolve_generation_devices()
        self._generation_contexts: list[_GenerationContext] = [
            _GenerationContext(
                device=self.device,
                video_predictor=self.video_predictor,
                policy_model=self.policy_model,
            )
        ]
        # Replica models are lazily initialized when multi-GPU generation is first used.
        self._generation_replicas_stale = False

    def _build_generation_context(self, device: str) -> _GenerationContext:
        print(f"[wm] initializing generation replica on {device}")
        video_predictor = VideoPredictor(device, self.cfg.world_model)
        policy_model = get_model(
            self.cfg.actor.checkpoint_load_path,
            self.cfg.actor.model,
        )
        policy_model.eval()
        policy_model.to(device)
        return _GenerationContext(
            device=device,
            video_predictor=video_predictor,
            policy_model=policy_model,
        )

    def _ensure_generation_contexts(self) -> list[_GenerationContext]:
        if len(self._generation_devices) <= 1:
            return self._generation_contexts
        while len(self._generation_contexts) < len(self._generation_devices):
            next_device = self._generation_devices[len(self._generation_contexts)]
            self._generation_contexts.append(self._build_generation_context(next_device))
            self._generation_replicas_stale = True
        return self._generation_contexts

    def _mark_generation_replicas_stale(self) -> None:
        if len(self._generation_devices) > 1:
            self._generation_replicas_stale = True

    def _sync_generation_replicas_if_needed(self) -> None:
        if len(self._generation_devices) <= 1:
            return
        self._ensure_generation_contexts()
        if not self._generation_replicas_stale:
            return

        primary = self._generation_contexts[0]
        model_state = primary.video_predictor.model.state_dict()
        tokenizer_state = primary.video_predictor.tokenizer.state_dict()
        policy_state = primary.policy_model.state_dict()

        for context in self._generation_contexts[1:]:
            context.video_predictor.model.load_state_dict(model_state, strict=True)
            context.video_predictor.tokenizer.load_state_dict(tokenizer_state, strict=True)
            context.policy_model.load_state_dict(policy_state, strict=True)
            context.policy_model.eval()
        self._generation_replicas_stale = False

    def load_policy_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.policy_model.load_state_dict(state_dict)
        self.policy_model.eval()
        self._mark_generation_replicas_stale()

    def _global_env_index(self, env_rank: int, stage_id: int, env_idx: int) -> int:
        return (
            (env_rank * self.stage_num + stage_id) * self.num_envs_per_stage + env_idx
        )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
            frame = cv2.resize(
                frame,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _infer_target_image_size(self) -> tuple[int, int]:
        env_cfg = getattr(self.cfg, "env", None)
        train_cfg = getattr(env_cfg, "train", None) if env_cfg is not None else None
        init_params = (
            getattr(train_cfg, "init_params", None) if train_cfg is not None else None
        )
        height = None
        width = None
        if init_params is not None:
            height = getattr(init_params, "camera_heights", None) or getattr(
                init_params, "camera_height", None
            )
            width = getattr(init_params, "camera_widths", None) or getattr(
                init_params, "camera_width", None
            )
        if height is None or width is None:
            size = getattr(train_cfg, "image_size", None) if train_cfg is not None else None
            if size is None:
                size = self.image_size
            height = width = size
        return int(height), int(width)

    def _infer_state_dim(self) -> int:
        state_dim = getattr(self.cfg.world_model, "state_dim", None)
        if state_dim is not None:
            return int(state_dim)
        env_cfg = getattr(self.cfg, "env", None)
        train_cfg = getattr(env_cfg, "train", None) if env_cfg is not None else None
        sim_type = getattr(train_cfg, "simulator_type", None)
        if sim_type == "libero":
            return 8
        if sim_type == "metaworld":
            return 4
        return int(getattr(self.cfg.actor.model, "action_dim", 0))

    def _to_chw_uint8(self, frame: Any) -> np.ndarray:
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
            frame = frame.transpose(1, 2, 0)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame = self._resize_frame(frame)
        if frame.ndim != 3 or frame.shape[-1] not in {1, 3}:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        return frame.transpose(2, 0, 1)

    def _ensure_frame_stack(self, env_id: int, frame: np.ndarray) -> None:
        if not self._frame_buffers[env_id] or self._env_done[env_id]:
            self._frame_buffers[env_id].clear()
            for _ in range(self.frame_stack):
                self._frame_buffers[env_id].append(frame)
            self._env_done[env_id] = False
        else:
            self._frame_buffers[env_id].append(frame)

    def _stack_frames(self, env_id: int) -> np.ndarray:
        frames = list(self._frame_buffers[env_id])
        if len(frames) != self.frame_stack:
            raise RuntimeError("Frame stack not initialized")
        return np.concatenate(frames, axis=0)

    def ingest_reset(self, payload: dict[str, Any]) -> None:
        env_rank = int(payload["env_rank"])
        stage_id = int(payload["stage_id"])
        images = payload.get("images")
        task_descs = payload.get("task_descriptions")
        if task_descs is not None:
            for env_idx, desc in enumerate(task_descs):
                global_idx = self._global_env_index(env_rank, stage_id, env_idx)
                self._task_desc_per_env[global_idx] = desc
        if images is None:
            return
        if torch.is_tensor(images):
            images = images.detach().cpu().numpy()
        for env_idx, frame in enumerate(images):
            global_idx = self._global_env_index(env_rank, stage_id, env_idx)
            frame_chw = self._to_chw_uint8(frame)
            self._frame_buffers[global_idx].clear()
            for _ in range(self.frame_stack):
                self._frame_buffers[global_idx].append(frame_chw)
            self._env_done[global_idx] = False

    def ingest_step(self, payload: dict[str, Any]) -> int:
        env_rank = int(payload["env_rank"])
        stage_id = int(payload["stage_id"])
        images = payload["images"]
        actions = payload["actions"]
        rewards = payload["rewards"]
        dones = payload["dones"]
        task_descs = payload.get("task_descriptions")

        if torch.is_tensor(images):
            images = images.detach().cpu().numpy()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if torch.is_tensor(rewards):
            rewards = rewards.detach().cpu().numpy()
        if torch.is_tensor(dones):
            dones = dones.detach().cpu().numpy()

        if task_descs is not None:
            for env_idx, desc in enumerate(task_descs):
                global_idx = self._global_env_index(env_rank, stage_id, env_idx)
                self._task_desc_per_env[global_idx] = desc

        num_envs, chunk_steps = images.shape[:2]
        transition_count = 0
        for env_idx in range(num_envs):
            global_idx = self._global_env_index(env_rank, stage_id, env_idx)
            for step_idx in range(chunk_steps):
                frame = images[env_idx, step_idx]
                frame_chw = self._to_chw_uint8(frame)
                self._ensure_frame_stack(global_idx, frame_chw)
                stacked = self._stack_frames(global_idx)

                action = actions[env_idx, step_idx].astype(np.float32)
                reward = np.array([rewards[env_idx, step_idx]], dtype=np.float32)
                done_flag = bool(dones[env_idx, step_idx])
                discount = np.array([0.0 if done_flag else 1.0], dtype=np.float32)
                step_type = StepType.LAST if done_flag else StepType.MID

                time_step = _SimpleTimeStep(
                    step_type=step_type,
                    reward=reward,
                    discount=discount,
                    observation=stacked,
                    action=action,
                )
                self.replay_storage.add(time_step)
                transition_count += 1

                if done_flag:
                    self._env_done[global_idx] = True
                    self._frame_buffers[global_idx].clear()
        self.total_env_steps += transition_count
        return transition_count

    def _get_replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _get_seg_replay_iter(self):
        if self._seg_replay_iter is None:
            self._seg_replay_iter = iter(self.seg_replay_loader)
        return self._seg_replay_iter

    def _sample_task_descriptions(self, batch_size: int) -> list[str]:
        pool = [d for d in self._task_desc_per_env if d]
        if not pool:
            return [""] * batch_size
        if len(pool) >= batch_size:
            return pool[:batch_size]
        repeats = math.ceil(batch_size / len(pool))
        return (pool * repeats)[:batch_size]

    def _prepare_env_obs(self, obs: torch.Tensor, task_descriptions: list[str]):
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        image = obs[:, -3:, :, :]
        if torch.is_floating_point(image):
            if image.max() <= 1.0:
                image = image * 255.0
        image = image.float()
        target_h, target_w = self._target_image_size
        if image.shape[-2:] != (target_h, target_w):
            image = F.interpolate(
                image, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        image = image.clamp(0.0, 255.0).to(torch.uint8)
        batch = image.shape[0]
        states = torch.zeros((batch, self.state_dim), dtype=torch.float32)
        if len(task_descriptions) == 1 and batch > 1:
            task_descriptions = task_descriptions * batch
        elif len(task_descriptions) != batch:
            task_descriptions = (task_descriptions * math.ceil(batch / len(task_descriptions)))[:batch]
        env_obs = {
            "images": image.cpu(),
            "states": states,
            "task_descriptions": task_descriptions,
        }
        if getattr(self.cfg.actor.model, "use_wrist_image", False):
            env_obs["wrist_images"] = image.cpu()
        return env_obs

    def _policy_predict_with_model(
        self,
        policy_model,
        obs: torch.Tensor,
        task_descs: list[str],
    ):
        env_obs = self._prepare_env_obs(obs, task_descs)
        with torch.no_grad():
            actions, result = policy_model.predict_action_batch(
                env_obs=env_obs,
                mode="train",
            )
        return actions, result

    def _policy_predict(self, obs: torch.Tensor, task_descs: list[str]):
        return self._policy_predict_with_model(self.policy_model, obs, task_descs)

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return None
                return float(value.detach().float().mean().item())
            return float(value)
        except Exception:
            return None

    def _ensure_validation_frame_metrics(self) -> bool:
        if (
            self._val_psnr is not None
            and self._val_ssim is not None
            and self._val_lpips is not None
        ):
            return True
        try:
            import piqa
            from ivideogpt.vq_model import LPIPS

            self._val_psnr = piqa.PSNR(
                epsilon=1e-8,
                value_range=1.0,
                reduction="none",
            ).to(self.device)
            self._val_ssim = piqa.SSIM(
                window_size=11,
                sigma=1.5,
                n_channels=3,
                reduction="none",
            ).to(self.device)
            self._val_lpips = LPIPS().to(self.device).eval()
            return True
        except Exception:
            return False

    def _compute_fallback_frame_metrics(
        self,
        gt_flat: torch.Tensor,
        pred_flat: torch.Tensor,
    ) -> tuple[float, float, float]:
        eps = 1e-8
        mse = torch.mean((gt_flat - pred_flat) ** 2)
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))

        mu_x = gt_flat.mean(dim=(1, 2, 3), keepdim=True)
        mu_y = pred_flat.mean(dim=(1, 2, 3), keepdim=True)
        var_x = ((gt_flat - mu_x) ** 2).mean(dim=(1, 2, 3), keepdim=True)
        var_y = ((pred_flat - mu_y) ** 2).mean(dim=(1, 2, 3), keepdim=True)
        cov_xy = ((gt_flat - mu_x) * (pred_flat - mu_y)).mean(
            dim=(1, 2, 3), keepdim=True
        )
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = (
            (2.0 * mu_x * mu_y + c1)
            * (2.0 * cov_xy + c2)
            / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2) + eps)
        ).mean()

        lpips_value = None
        lpips_model = getattr(self.video_predictor, "lpips", None)
        if lpips_model is not None:
            try:
                lpips_value = lpips_model(
                    gt_flat.contiguous() * 2.0 - 1.0,
                    pred_flat.contiguous() * 2.0 - 1.0,
                    weight=None,
                ).mean()
            except Exception:
                lpips_value = None
        if lpips_value is None:
            lpips_value = torch.mean(torch.abs(gt_flat - pred_flat))

        return (
            float(psnr.item()),
            float(ssim.item()),
            float(lpips_value.item()),
        )

    def maybe_update_world_model(
        self,
        env_steps: int,
        init_step_logger: Optional[Callable[[dict[str, float]], None]] = None,
    ) -> dict[str, float]:
        metrics = {}
        updated = False
        if env_steps < self.cfg.num_seed_frames:
            return metrics

        if not self.init_model:
            if self.replay_storage._num_episodes == 0 and not self._has_demo:
                return metrics
            init_iter = range(self.cfg.init_update_gen_steps)
            if getattr(self.cfg.world_model, "show_progress", False):
                init_iter = tqdm(
                    init_iter,
                    desc="WM init update",
                    leave=False,
                    ncols=120,
                )
            for _ in init_iter:
                batch = next(self._get_seg_replay_iter())
                update_tokenizer = getattr(
                    self.cfg.world_model, "seed_update_tokenizer", True
                )
                metrics = self.video_predictor.train(batch, update_tokenizer=update_tokenizer)
                updated = True
                self.wm_init_update_step += 1
                if init_step_logger is not None and metrics:
                    payload = {
                        "wm_init/update_step": float(self.wm_init_update_step),
                    }
                    for key, value in metrics.items():
                        cast_value = self._to_float(value)
                        if cast_value is not None:
                            payload[f"wm_init/{key}"] = cast_value
                    if len(payload) > 1:
                        init_step_logger(payload)
            self.video_predictor.save_snapshot(self.work_dir, suffix="_init")
            self.init_model = True
            if updated:
                self._mark_generation_replicas_stale()
            output_metrics = {f"wm_init/{k}": v for k, v in metrics.items()}
            output_metrics["wm_init/update_step"] = float(self.wm_init_update_step)
            return output_metrics

        if self.cfg.update_gen_every_step > 0 and env_steps % self.cfg.update_gen_every_step == 0:
            if self.replay_storage._num_episodes == 0 and not self._has_demo:
                return metrics
            update_iter = range(self.cfg.update_gen_times)
            if getattr(self.cfg.world_model, "show_progress", False):
                update_iter = tqdm(
                    update_iter,
                    desc="WM update",
                    leave=False,
                    ncols=120,
                )
            for _ in update_iter:
                batch = next(self._get_seg_replay_iter())
                update_tokenizer = (
                    env_steps
                    % max(1, (self.cfg.update_tokenizer_every_step // max(1, self.cfg.action_repeat)))
                    == 0
                )
                metrics = self.video_predictor.train(batch, update_tokenizer=update_tokenizer)
                updated = True
            if updated:
                self._mark_generation_replicas_stale()
            return {f"wm/{k}": v for k, v in metrics.items()}

        return metrics

    def _generate_imagined_rollout_on_context(
        self,
        context: _GenerationContext,
        obs: torch.Tensor,
        task_descs: list[str],
        chunk_steps: int,
        progress_callback=None,
    ) -> dict[str, torch.Tensor]:
        obs = torch.as_tensor(obs, device=context.device)
        horizon = chunk_steps * self.num_action_chunks
        rollout_result = EmbodiedRolloutResult()
        actions_cache: Optional[torch.Tensor] = None
        cache_index = 0

        def policy(step_obs: torch.Tensor, step_idx: int):
            nonlocal actions_cache, cache_index
            if step_idx % self.num_action_chunks == 0:
                actions, result = self._policy_predict_with_model(
                    context.policy_model,
                    step_obs,
                    task_descs,
                )
                actions_cache = torch.as_tensor(
                    actions, device=context.device, dtype=torch.float32
                )
                cache_index = 0
                rollout_result.append_result(result)
            action = actions_cache[:, cache_index]
            cache_index += 1
            return action

        obss, _actions, rewards = context.video_predictor.rollout(
            obs,
            policy,
            horizon,
            do_sample=getattr(self.cfg.world_model, "rollout_do_sample", True),
            progress_callback=progress_callback,
        )
        if obss is not None and obss.shape[1] > 0:
            _, final_result = self._policy_predict_with_model(
                context.policy_model,
                obss[:, -1],
                task_descs,
            )
            if "prev_values" in final_result:
                rollout_result.prev_values.append(
                    final_result["prev_values"].cpu().contiguous()
                )
        rewards = rewards[:, 1:]
        if rewards.ndim == 2:
            rewards = rewards.unsqueeze(-1)
        rewards = rewards.reshape(
            rewards.shape[0], chunk_steps, self.num_action_chunks, -1
        )
        rewards = rewards.squeeze(-1).transpose(0, 1).contiguous()
        dones = torch.zeros(
            (chunk_steps + 1, rewards.shape[1], rewards.shape[2]),
            dtype=torch.bool,
            device=rewards.device,
        )

        rollout_batch = rollout_result.to_dict()
        rollout_batch["rewards"] = rewards.cpu().contiguous()
        rollout_batch["dones"] = dones.cpu().contiguous()
        return rollout_batch

    @staticmethod
    def _merge_imagined_rollout_batches(
        rollout_batches: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        keys = set()
        for batch in rollout_batches:
            keys.update(batch.keys())
        for key in keys:
            values = [batch.get(key) for batch in rollout_batches]
            values = [value for value in values if value is not None]
            if not values:
                merged[key] = None
                continue
            if isinstance(values[0], torch.Tensor):
                merged[key] = torch.cat(values, dim=1).contiguous()
            else:
                raise ValueError(
                    f"Unsupported rollout batch value type for key={key}: {type(values[0])}"
                )
        return merged

    def generate_imagined_rollout(self, chunk_steps: int) -> Optional[dict[str, torch.Tensor]]:
        if self.replay_storage._num_episodes == 0 and not self._has_demo:
            return None

        contexts = self._ensure_generation_contexts()
        self._sync_generation_replicas_if_needed()

        batch = next(self._get_replay_iter())
        obs_cpu = torch.as_tensor(batch[0][: self.cfg.gen_batch])
        total_batch = int(obs_cpu.shape[0])
        if total_batch == 0:
            return None
        task_descs = self._sample_task_descriptions(total_batch)
        show_progress = bool(getattr(self.cfg.world_model, "gen_show_progress", True))

        active_contexts = contexts[: min(len(contexts), total_batch)]
        if len(active_contexts) == 1:
            single_progress_cb = None
            if show_progress:
                horizon = chunk_steps * self.num_action_chunks
                if horizon > 0:
                    total_units = int(total_batch * horizon)
                    pbar = tqdm(
                        total=total_units,
                        desc="WM Gen",
                        ncols=120,
                        leave=False,
                    )
                    step_units = int(total_batch)

                    def _update_pbar(delta_units: int):
                        remaining = max(0, total_units - int(pbar.n))
                        inc = min(delta_units, remaining)
                        if inc > 0:
                            pbar.update(inc)
                        sample_equiv = (float(pbar.n) / float(horizon))
                        pbar.set_postfix(
                            {"samples": f"{sample_equiv:.1f}/{float(total_batch):.1f}"}
                        )

                    def single_progress_cb():
                        _update_pbar(step_units)
                else:
                    pbar = None
            else:
                pbar = None
            rollout_batch = self._generate_imagined_rollout_on_context(
                active_contexts[0],
                obs_cpu,
                task_descs,
                chunk_steps,
                progress_callback=single_progress_cb,
            )
            if pbar is not None:
                total_units = int(total_batch * horizon)
                if int(pbar.n) < total_units:
                    pbar.update(total_units - int(pbar.n))
                pbar.set_postfix({"samples": f"{float(total_batch):.1f}/{float(total_batch):.1f}"})
                pbar.close()
            return rollout_batch

        num_shards = len(active_contexts)
        shard_sizes = [total_batch // num_shards for _ in range(num_shards)]
        for i in range(total_batch % num_shards):
            shard_sizes[i] += 1

        starts = [0]
        for size in shard_sizes[:-1]:
            starts.append(starts[-1] + size)

        rollout_batches_by_shard: list[Optional[dict[str, torch.Tensor]]] = [None] * num_shards
        horizon = chunk_steps * self.num_action_chunks
        pbar = None
        total_units = None
        if show_progress:
            total_units = int(total_batch * max(1, horizon))
            pbar = tqdm(
                total=total_units,
                desc=f"WM Gen ({num_shards} GPU)",
                ncols=120,
                leave=False,
            )
        pbar_lock = threading.Lock()

        def _safe_update_pbar(delta_units: int):
            if pbar is None:
                return
            with pbar_lock:
                remaining = max(0, int(total_units) - int(pbar.n))
                inc = min(delta_units, remaining)
                if inc > 0:
                    pbar.update(inc)
                horizon_denom = max(1, horizon)
                sample_equiv = float(pbar.n) / float(horizon_denom)
                pbar.set_postfix(
                    {"samples": f"{sample_equiv:.1f}/{float(total_batch):.1f}"}
                )

        try:
            with ThreadPoolExecutor(max_workers=num_shards) as executor:
                future_to_meta = {}
                for shard_idx, context in enumerate(active_contexts):
                    start = starts[shard_idx]
                    end = start + shard_sizes[shard_idx]
                    obs_shard = obs_cpu[start:end]
                    task_shard = task_descs[start:end]
                    shard_size = shard_sizes[shard_idx]
                    shard_step_units = int(shard_size)

                    def _make_progress_cb(delta_units: int):
                        if pbar is None:
                            return None

                        def _progress_cb():
                            _safe_update_pbar(delta_units)

                        return _progress_cb

                    shard_progress_cb = _make_progress_cb(shard_step_units)
                    future = executor.submit(
                        self._generate_imagined_rollout_on_context,
                        context,
                        obs_shard,
                        task_shard,
                        chunk_steps,
                        shard_progress_cb,
                    )
                    future_to_meta[future] = (shard_idx, shard_size)
                for future in as_completed(future_to_meta):
                    shard_idx, shard_size = future_to_meta[future]
                    rollout_batches_by_shard[shard_idx] = future.result()
        finally:
            if pbar is not None:
                if int(pbar.n) < int(total_units):
                    pbar.update(int(total_units) - int(pbar.n))
                pbar.set_postfix({"samples": f"{float(total_batch):.1f}/{float(total_batch):.1f}"})
                pbar.close()

        rollout_batches = [batch for batch in rollout_batches_by_shard if batch is not None]
        return self._merge_imagined_rollout_batches(rollout_batches)

    def validate(
        self,
        global_step: int,
        num_gifs: int = 1,
        fps: int = 4,
        num_metric_batches: int = 1,
    ) -> dict[str, float]:
        if self.replay_storage._num_episodes == 0 and not self._has_demo:
            print("[stage] world model validate skipped (empty replay buffer and no demos)")
            return {}

        num_metric_batches = max(1, int(num_metric_batches))
        enable_frame_metrics = self._ensure_validation_frame_metrics()

        obs_mse_total = 0.0
        reward_mse_total = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        frame_metric_count = 0

        action_norm_sum = 0.0
        action_norm_count = 0
        action_norm_min = float("inf")
        action_norm_max = float("-inf")

        gif_obs_gt = None
        gif_obs_pred = None
        gif_reward_gt = None
        gif_reward_pred = None

        for metric_batch_idx in range(num_metric_batches):
            batch = next(self._get_seg_replay_iter())
            obs_gt = torch.cat(
                [batch[0][:, :-2], batch[0][:, 1:-1], batch[0][:, 2:]], dim=2
            )
            action = batch[1][:, 2:]
            reward_gt = batch[2][:, 2:]
            policy = lambda obs, step: action[:, step].to(obs.device)
            obs_pred, _, reward_pred = self.video_predictor.rollout(
                obs_gt[:, 0],
                policy,
                obs_gt.shape[1] - 1,
                do_sample=False,
            )

            obs_mse = (
                (obs_pred[:, 1:] - (obs_gt[:, 1:] / 255.0).to(obs_pred.device)) ** 2
            ).mean()
            reward_mse = (
                (reward_pred[:, 1:] - reward_gt[:, 1:].to(reward_pred.device)) ** 2
            ).mean()
            obs_mse_total += float(obs_mse.item())
            reward_mse_total += float(reward_mse.item())

            action_norms = torch.linalg.norm(action, dim=-1).reshape(-1)
            if action_norms.numel() > 0:
                action_norm_sum += float(action_norms.sum().item())
                action_norm_count += int(action_norms.numel())
                action_norm_min = min(action_norm_min, float(action_norms.min().item()))
                action_norm_max = max(action_norm_max, float(action_norms.max().item()))

            gt_rgb = (
                obs_gt[:, 1:, -3:].to(self.device).float() / 255.0
            ).clamp(0.0, 1.0)
            pred_rgb = obs_pred[:, 1:, -3:].to(self.device).float().clamp(0.0, 1.0)

            bsz, timesteps, channels, height, width = gt_rgb.shape
            gt_flat = gt_rgb.reshape(bsz * timesteps, channels, height, width)
            pred_flat = pred_rgb.reshape(bsz * timesteps, channels, height, width)

            if enable_frame_metrics:
                psnr = self._val_psnr(gt_flat, pred_flat).mean()
                ssim = self._val_ssim(gt_flat, pred_flat).mean()
                lpips = self._val_lpips(
                    gt_flat.contiguous() * 2.0 - 1.0,
                    pred_flat.contiguous() * 2.0 - 1.0,
                    weight=None,
                ).mean()
                psnr_total += float(psnr.item())
                ssim_total += float(ssim.item())
                lpips_total += float(lpips.item())
                frame_metric_count += 1
            else:
                psnr, ssim, lpips = self._compute_fallback_frame_metrics(
                    gt_flat=gt_flat,
                    pred_flat=pred_flat,
                )
                psnr_total += psnr
                ssim_total += ssim
                lpips_total += lpips
                frame_metric_count += 1

            if metric_batch_idx == 0:
                gif_obs_gt = obs_gt.detach().cpu()
                gif_obs_pred = obs_pred.detach().cpu()
                gif_reward_gt = reward_gt.detach().cpu()
                gif_reward_pred = reward_pred.detach().cpu()

        if action_norm_count > 0:
            action_norm_mean = action_norm_sum / action_norm_count
            print(
                "[validate] action_norm mean="
                f"{action_norm_mean:.4f}, min={action_norm_min:.4f}, "
                f"max={action_norm_max:.4f}"
            )
        else:
            action_norm_mean = 0.0
            action_norm_min = 0.0
            action_norm_max = 0.0

        if num_gifs > 0:
            gif_dir = self.work_dir / "validate_gif"
            gif_dir.mkdir(parents=True, exist_ok=True)
            if gif_obs_gt is not None and gif_obs_pred is not None:
                gif_count = min(int(num_gifs), int(gif_obs_gt.shape[0]))
                for i in range(gif_count):
                    gif_path = gif_dir / f"val-sample-{global_step}-{i}.gif"
                    frames = []
                    for t in range(gif_obs_gt.shape[1]):
                        frame = (
                            gif_obs_gt[i, t, -3:]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                        ).astype(np.uint8)
                        frame = np.ascontiguousarray(frame)
                        frame_pred = (
                            gif_obs_pred[i, t, -3:]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy()
                            * 255.0
                        ).astype(np.uint8)
                        frame_pred = np.ascontiguousarray(frame_pred)
                        frame_error = np.abs(
                            frame.astype(np.float32) - frame_pred.astype(np.float32)
                        ).astype(np.uint8)
                        if t > 0:
                            cv2.putText(
                                frame,
                                f"{gif_reward_gt[i, t].item():.2f}",
                                (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                frame_pred,
                                f"{gif_reward_pred[i, t].item():.2f}",
                                (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3,
                                (255, 255, 255),
                                1,
                            )
                        frames.append(
                            np.concatenate([frame, frame_pred, frame_error], axis=1)
                        )
                    imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)

        metrics = {
            "val/obs_mse": obs_mse_total / float(num_metric_batches),
            "val/reward_mse": reward_mse_total / float(num_metric_batches),
            "val/action_norm_mean": action_norm_mean,
            "val/action_norm_min": action_norm_min,
            "val/action_norm_max": action_norm_max,
            "val/metric_batches": float(num_metric_batches),
        }

        denom = max(1.0, float(frame_metric_count))
        metrics["val/psnr"] = psnr_total / denom
        metrics["val/ssim"] = ssim_total / denom
        metrics["val/lpips"] = lpips_total / denom
        # Keep uppercase aliases to make dashboard filtering straightforward.
        metrics["val/PSNR"] = metrics["val/psnr"]
        metrics["val/SSIM"] = metrics["val/ssim"]
        metrics["val/LPIPS"] = metrics["val/lpips"]

        return metrics

    def should_start_mbpo(self, env_steps: int) -> bool:
        return env_steps >= self.cfg.start_mbpo
