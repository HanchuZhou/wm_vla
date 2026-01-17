import random
from collections import deque
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rlinf.models import get_model


class OpenPiMBPOAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        model_path,
        openpi,
        lr=5.0e-6,
        weight_decay=0.0,
        batch_size=64,
        update_epochs=1,
        clip_ratio=0.2,
        value_coef=0.5,
        max_buffer_size=50000,
        gamma=0.99,
    ):
        self.device = torch.device(device)
        self.action_dim = action_shape[0]
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.gamma = gamma

        model_cfg = OmegaConf.create(
            {
                "model_name": "openpi",
                "precision": None,
                "is_lora": openpi.get("is_lora", False) if isinstance(openpi, dict) else False,
                "openpi": openpi,
            }
        )
        self.model = get_model(model_path, model_cfg)
        self.model.train()
        self.training = True

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self._buffer = deque(maxlen=max_buffer_size)
        self._rng = random.Random(0)

    def train(self, training: bool = True):
        self.training = training
        self.model.train(training)

    def _reduce_logprob(self, logprob: torch.Tensor) -> torch.Tensor:
        if logprob.ndim == 1:
            return logprob
        return logprob.reshape(logprob.shape[0], -1).mean(dim=1)

    def _reduce_value(self, value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if value.ndim == 1:
            return value
        return value.reshape(value.shape[0], -1).mean(dim=1)

    def _prepare_env_obs(
        self,
        obs: Any,
        frame_stack: int = 3,
        use_wrist: bool = True,
        task_descriptions: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if isinstance(obs, dict):
            return obs
        if torch.is_tensor(obs):
            tensor_obs = obs
        else:
            tensor_obs = torch.as_tensor(obs)
        if tensor_obs.ndim == 3:
            tensor_obs = tensor_obs.unsqueeze(0)
        # use the most recent RGB frame for openpi
        image = tensor_obs[:, -3:, :, :]
        if torch.is_floating_point(image):
            if image.max() <= 1.0:
                image = image * 255.0
        image = image.to(torch.uint8)
        state = torch.zeros((image.shape[0], self.action_dim), dtype=torch.float32)
        if task_descriptions is None:
            task_descriptions = [""] * image.shape[0]
        elif len(task_descriptions) == 1 and image.shape[0] > 1:
            task_descriptions = task_descriptions * image.shape[0]
        env_obs = {
            "images": image.cpu(),
            "states": state,
            "task_descriptions": task_descriptions,
        }
        if use_wrist:
            env_obs["wrist_images"] = image.cpu()
        return env_obs

    def act(self, obs, step, eval_mode=False, return_info=False, task_descriptions=None):
        self.model.eval()
        env_obs = self._prepare_env_obs(obs, task_descriptions=task_descriptions)
        with torch.no_grad():
            actions, result = self.model.predict_action_batch(
                env_obs=env_obs, mode="eval" if eval_mode else "train"
            )
        # openpi outputs chunked actions; take the first action
        action = actions[:, 0].astype(np.float32)
        action_batch = action  # shape (batch, action_dim)
        if not return_info:
            return action_batch
        info = self._pack_policy_info(result)
        return action_batch, info

    def act2(self, obs, step, eval_mode=False, return_info=False, task_descriptions=None):
        env_obs = self._prepare_env_obs(obs, task_descriptions=task_descriptions)
        with torch.no_grad():
            actions, result = self.model.predict_action_batch(
                env_obs=env_obs, mode="eval" if eval_mode else "train"
            )
        action = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        action = action[:, 0]
        if not return_info:
            return action
        info = self._pack_policy_info(result)
        return action, info

    def _pack_policy_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        info = {
            "prev_logprobs": result["prev_logprobs"].detach().cpu(),
            "prev_values": result.get("prev_values", None),
            "forward_inputs": {},
        }
        if info["prev_values"] is not None:
            info["prev_values"] = info["prev_values"].detach().cpu()
        for key, value in result["forward_inputs"].items():
            info["forward_inputs"][key] = self._detach_nested(value)
        return info

    def _detach_nested(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._detach_nested(v) for k, v in value.items()}
        return value

    def add_transition(
        self,
        policy_info: Dict[str, Any],
        reward: float,
        discount: float,
        done: bool,
    ):
        if policy_info is None:
            return
        entry = {
            "reward": float(reward),
            "discount": float(discount),
            "done": bool(done),
            **policy_info,
        }
        self._buffer.append(entry)

    def add_batch_transitions(
        self,
        policy_info: Dict[str, Any],
        rewards: Iterable[float],
        discounts: Optional[Iterable[float]] = None,
    ):
        if policy_info is None:
            return
        rewards = np.asarray(rewards, dtype=np.float32)
        if discounts is None:
            discounts = np.ones_like(rewards, dtype=np.float32)
        else:
            discounts = np.asarray(discounts, dtype=np.float32)
        batch_size = rewards.shape[0]
        for i in range(batch_size):
            sliced = self._slice_policy_info(policy_info, i)
            self.add_transition(
                sliced, rewards[i], discounts[i], done=False
            )

    def _slice_policy_info(self, info: Dict[str, Any], idx: int) -> Dict[str, Any]:
        sliced = {
            "prev_logprobs": info["prev_logprobs"][idx],
            "prev_values": None,
            "forward_inputs": {},
        }
        if info.get("prev_values") is not None:
            sliced["prev_values"] = info["prev_values"][idx]
        for key, value in info["forward_inputs"].items():
            sliced["forward_inputs"][key] = self._slice_nested(value, idx)
        return sliced

    def _slice_nested(self, value, idx):
        if torch.is_tensor(value):
            return value[idx]
        if isinstance(value, dict):
            return {k: self._slice_nested(v, idx) for k, v in value.items()}
        return value

    def update(self, step):
        if len(self._buffer) < self.batch_size:
            return {}
        metrics = {}
        for _ in range(self.update_epochs):
            batch = self._rng.sample(self._buffer, self.batch_size)
            data, rewards, prev_logprobs = self._collate_batch(batch)
            data = {k: self._to_device(v) for k, v in data.items()}
            prev_logprobs = prev_logprobs.to(self.device)
            rewards = rewards.to(self.device)

            output = self.model(data, compute_logprobs=True, compute_values=True)
            logprobs = self._reduce_logprob(output["logprobs"])
            old_logprobs = self._reduce_logprob(prev_logprobs)
            values = self._reduce_value(output.get("values"))

            if values is None:
                advantages = rewards
                value_loss = torch.tensor(0.0, device=self.device)
            else:
                advantages = rewards - values.detach()
                value_loss = F.mse_loss(values, rewards)

            ratio = torch.exp(logprobs - old_logprobs)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )
            policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
            loss = policy_loss + self.value_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metrics = {
                "actor/policy_loss": policy_loss.item(),
                "critic/value_loss": value_loss.item(),
                "actor/adv_mean": advantages.mean().item(),
            }
        return metrics

    def _to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        return value

    def _collate_batch(self, batch):
        forward_inputs = {}
        prev_logprobs = []
        rewards = []
        for item in batch:
            for key, value in item["forward_inputs"].items():
                forward_inputs.setdefault(key, []).append(value)
            prev_logprobs.append(item["prev_logprobs"])
            rewards.append(item["reward"])

        data = {}
        for key, values in forward_inputs.items():
            data[key] = self._stack_nested(values)
        prev_logprobs = torch.stack(prev_logprobs, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        return data, rewards, prev_logprobs

    def _stack_nested(self, values):
        if torch.is_tensor(values[0]):
            return torch.stack(values, dim=0)
        if isinstance(values[0], dict):
            return {k: self._stack_nested([v[k] for v in values]) for k in values[0]}
        return values
