import asyncio
from collections import defaultdict
import os
import time
from pathlib import Path
from typing import Optional

import torch
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from iVideoGPT.mbrl.world_model_mbpo import WorldModelMBPO


class EmbodiedMBPORunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedFSDPActor,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
        world_model: Optional[WorldModelMBPO] = None,
        wm_channel=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.world_model = world_model
        self.wm_channel = wm_channel

        self.run_timer = run_timer

        self.consumed_samples = 0
        self.global_step = 0

        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)
        self.real_env_interactions = 0.0
        self._setup_wandb_metric_axes()

    def _setup_wandb_metric_axes(self) -> None:
        wandb = self.metric_logger.logger.get("wandb", None)
        if wandb is None:
            return
        try:
            wandb.define_metric("global_step")
            for key_pattern in (
                "time/*",
                "eval/*",
                "env/*",
                "rollout/*",
                "train/*",
                "wm/*",
                "val/*",
            ):
                wandb.define_metric(key_pattern, step_metric="global_step")
            wandb.define_metric("env/real_interactions")
            wandb.define_metric(
                "eval/success_once_vs_real_env_interactions",
                step_metric="env/real_interactions",
            )
            wandb.define_metric("wm_init/update_step")
            wandb.define_metric(
                "wm_init/*",
                step_metric="wm_init/update_step",
            )
        except Exception:
            pass

    def _log_metrics(self, data: dict[str, object], step: int) -> None:
        if not data:
            return
        non_wandb_backends = [
            name for name in self.metric_logger.logger.keys() if name != "wandb"
        ]
        if non_wandb_backends:
            self.metric_logger.log(data=data, step=step, backend=non_wandb_backends)
        wandb = self.metric_logger.logger.get("wandb", None)
        if wandb is None:
            return
        payload = dict(data)
        payload.setdefault("global_step", float(step))
        wandb.log(payload)

    def _log_wm_init_step_metrics(self, data: dict[str, float]) -> None:
        if not data:
            return
        payload = dict(data)
        wm_init_step = int(
            max(
                0.0,
                self._safe_to_float(payload.get("wm_init/update_step", 0.0)),
            )
        )
        non_wandb_backends = [
            name for name in self.metric_logger.logger.keys() if name != "wandb"
        ]
        if non_wandb_backends:
            backend_payload = {
                key: value for key, value in payload.items() if key != "wm_init/update_step"
            }
            if backend_payload:
                self.metric_logger.log(
                    data=backend_payload,
                    step=wm_init_step,
                    backend=non_wandb_backends,
                )
        wandb = self.metric_logger.logger.get("wandb", None)
        if wandb is None:
            return
        wandb.log(payload, commit=True)

    @staticmethod
    def _safe_to_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "sum"):
                value = value.sum()
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return default

    def init_workers(self):
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        if self.cfg.runner.get("resume_dir", None) is not None:
            self.global_step = int(self.cfg.runner.resume_dir.split("global_step_")[-1])

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    def update_rollout_weights(self):
        sync_every = int(getattr(self.cfg.rollout, "sync_every_steps", 1))
        if sync_every <= 0:
            return
        if sync_every > 1 and (self.global_step % sync_every) != 0:
            return
        rollout_futures = self.rollout.sync_model_from_actor()
        actor_futures = self.actor.sync_model_to_rollout()
        actor_futures.wait()
        rollout_futures.wait()

    def _sync_world_model_policy(self, force: bool = False):
        if self.world_model is None:
            return
        sync_every = int(self.cfg.world_model.get("sync_every_steps", 1))
        if sync_every <= 0 and not force:
            return
        if not force and sync_every > 1 and (self.global_step % sync_every) != 0:
            return
        state_dict = self.actor.execute_on(0).get_model_state_dict().wait()[0]
        self.world_model.load_policy_state_dict(state_dict)

    def _drain_wm_channel(self):
        if self.wm_channel is None or self.world_model is None:
            return 0
        drained = 0
        key = self.cfg.world_model.channel.queue_name
        while True:
            try:
                payload = self.wm_channel.get_nowait(key=key)
            except asyncio.QueueEmpty:
                break
            if payload.get("type") == "reset":
                self.world_model.ingest_reset(payload)
            elif payload.get("type") == "step":
                drained += self.world_model.ingest_step(payload)
        return drained

    def _log_wandb_gifs(
        self,
        base_dir: str | os.PathLike,
        key_prefix: str,
        max_gifs: int,
        fps: int = 4,
    ) -> None:
        if max_gifs <= 0:
            return
        base_path = Path(base_dir)
        if not base_path.exists():
            return
        media_paths = []
        for pattern in ("*.gif", "*.mp4"):
            media_paths.extend(base_path.rglob(pattern))
        media_paths = sorted(
            [p for p in media_paths if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not media_paths:
            return
        media_paths = media_paths[:max_gifs]
        wandb = self.metric_logger.logger.get("wandb", None)
        if wandb is None:
            return
        payload = {}
        for idx, media_path in enumerate(media_paths):
            fmt = media_path.suffix.lower().lstrip(".")
            if fmt not in {"gif", "mp4"}:
                continue
            try:
                payload[f"{key_prefix}_{idx}"] = wandb.Video(
                    str(media_path), format=fmt
                )
            except Exception as exc:
                print(f"[warn] failed to log media to wandb: {media_path} ({exc})")
                continue
        if payload:
            payload["global_step"] = float(self.global_step)
            wandb.log(payload)

    def _world_model_status_metrics(
        self,
        wm_metrics: dict[str, float],
        use_real_rollout: bool,
    ) -> dict[str, float]:
        if self.world_model is None:
            return {}
        update_every = int(getattr(self.cfg, "update_gen_every_step", 0))
        mbpo_active = (
            self.world_model.init_model
            and self.world_model.should_start_mbpo(self.global_step)
        )
        return {
            "wm/status/init_model": float(self.world_model.init_model),
            "wm/status/init_gen": float(self.world_model.init_gen),
            "wm/status/mbpo_active": float(mbpo_active),
            "wm/status/use_real_rollout": float(use_real_rollout),
            "wm/status/env_steps": float(self._world_model_env_steps()),
            "wm/status/replay_episodes": float(self.world_model.replay_storage._num_episodes),
            "wm/status/updated": float(bool(wm_metrics)),
            "wm/status/update_every": float(update_every),
        }

    def _world_model_env_steps(self) -> int:
        if self.world_model is None:
            return 0
        if self.world_model.total_envs <= 0:
            return self.world_model.total_env_steps
        return self.world_model.total_env_steps // self.world_model.total_envs

    def _should_use_real_rollout_for_vla(self) -> bool:
        every = int(getattr(self.cfg.runner, "real_rollout_every_steps", 1))
        if every <= 0:
            return False
        if every == 1:
            return True
        return self.global_step > 0 and (self.global_step % every) == 0

    def generate_rollouts(self):
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        actor_futures = self.actor.recv_rollout_batch()
        env_results = env_futures.wait()
        actor_futures.wait()
        rollout_futures.wait()

        real_env_interactions = 0.0
        env_results_list = []
        for results in env_results:
            if results is None:
                continue
            clean = dict(results)
            real_env_interactions += self._safe_to_float(
                clean.pop("__real_env_interactions__", 0.0)
            )
            if clean:
                env_results_list.append(clean)

        env_metrics = (
            compute_evaluate_metrics(env_results_list) if env_results_list else {}
        )
        return env_metrics, real_env_interactions

    def evaluate(self):
        target_eval_episodes = int(getattr(self.cfg, "num_eval_episodes", 0) or 0)
        # Backward compatible path: keep legacy one-pass eval if target is disabled.
        if target_eval_episodes <= 0:
            env_futures = self.env.evaluate()
            rollout_futures = self.rollout.evaluate()
            env_results = env_futures.wait()
            rollout_futures.wait()
            eval_metrics_list = [results for results in env_results if results is not None]
            eval_metrics = compute_evaluate_metrics(eval_metrics_list)
            return eval_metrics

        max_eval_passes = int(self.cfg.runner.get("max_eval_passes", 100))
        collected_metrics = defaultdict(list)
        collected_episode_count = 0

        for _ in range(max_eval_passes):
            env_futures = self.env.evaluate()
            rollout_futures = self.rollout.evaluate()
            env_results = env_futures.wait()
            rollout_futures.wait()
            eval_metrics_list = [results for results in env_results if results is not None]
            if not eval_metrics_list:
                break

            pass_episode_count = 0
            for metrics in eval_metrics_list:
                for key, value in metrics.items():
                    if value is None:
                        continue
                    tensor_value = (
                        value if torch.is_tensor(value) else torch.as_tensor(value)
                    )
                    tensor_value = tensor_value.detach().cpu().reshape(-1)
                    if tensor_value.numel() == 0:
                        continue
                    collected_metrics[key].append(tensor_value)
                success_once = metrics.get("success_once", None)
                if success_once is None:
                    continue
                success_once = (
                    success_once
                    if torch.is_tensor(success_once)
                    else torch.as_tensor(success_once)
                )
                pass_episode_count += int(success_once.numel())

            collected_episode_count += pass_episode_count
            if collected_episode_count >= target_eval_episodes:
                break

        if not collected_metrics:
            return {}

        eval_metrics = {}
        keys_without_episode_denominator = {"policy_action_entropy"}
        for key, chunks in collected_metrics.items():
            values = torch.cat(chunks, dim=0)
            if values.numel() == 0:
                continue
            if key not in keys_without_episode_denominator:
                values = values[:target_eval_episodes]
            eval_metrics[key] = values.float().mean().numpy()

        effective_episode_count = 0
        if "success_once" in collected_metrics:
            effective_episode_count = min(
                target_eval_episodes, int(torch.cat(collected_metrics["success_once"]).numel())
            )
        eval_metrics["episode_count"] = float(effective_episode_count)

        if effective_episode_count < target_eval_episodes:
            print(
                "[warn] eval collected fewer episodes than requested: "
                f"{effective_episode_count}/{target_eval_episodes}"
            )

        return eval_metrics

    def run(self):
        start_step = self.global_step
        wm_gen_total_duration = 0.0
        wm_gen_total_calls = 0

        if self.world_model is not None and not self.world_model.init_model:
            print("[stage] world model demo finetuning start")
            t0 = time.perf_counter()
            wm_metrics = self.world_model.maybe_update_world_model(
                self.global_step,
                init_step_logger=self._log_wm_init_step_metrics,
            )
            t1 = time.perf_counter()
            duration = t1 - t0
            print(f"[stage] world model demo finetuning end (duration={duration:.2f}s)")
            self._log_metrics({"time/wm_init": duration}, self.global_step)
            if wm_metrics:
                non_init_wm_metrics = {
                    k: v for k, v in wm_metrics.items() if not k.startswith("wm_init/")
                }
                if non_init_wm_metrics:
                    self._log_metrics(non_init_wm_metrics, self.global_step)
            if not self.world_model.init_model:
                print("[stage] world model demo finetuning skipped (no demo data or seeding not complete)")
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )
        for _step in range(start_step, self.max_steps):
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            eval_metrics = {}
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    if "eval/success_once" in eval_metrics:
                        eval_metrics["eval/success_once_vs_real_env_interactions"] = (
                            eval_metrics["eval/success_once"]
                        )
                    eval_metrics["env/real_interactions"] = self.real_env_interactions
                    self._log_metrics(data=eval_metrics, step=_step)
                    save_eval_and_val_gif = bool(
                        self.cfg.runner.get("save_eval_and_val_gif", True)
                    )
                    num_gifs = int(self.cfg.runner.get("num_gif_every_steps", 1))
                    if save_eval_and_val_gif and num_gifs > 0:
                        eval_video_cfg = self.cfg.env.eval.video_cfg
                        if getattr(eval_video_cfg, "save_video", False):
                            eval_fps = int(getattr(eval_video_cfg, "fps", 4))
                            self._log_wandb_gifs(
                                getattr(eval_video_cfg, "video_base_dir", "video/eval"),
                                "eval/gif",
                                num_gifs,
                                fps=eval_fps,
                            )
                    if self.world_model is not None:
                        num_metric_batches = int(
                            self.cfg.world_model.get("val_metric_batches", 1)
                        )
                        wm_num_gifs = num_gifs if save_eval_and_val_gif else 0
                        wm_val_metrics = self.world_model.validate(
                            global_step=self.global_step,
                            num_gifs=wm_num_gifs,
                            fps=4,
                            num_metric_batches=num_metric_batches,
                        )
                        if wm_val_metrics:
                            self._log_metrics(wm_val_metrics, step=_step)
                        if save_eval_and_val_gif and num_gifs > 0:
                            self._log_wandb_gifs(
                                self.world_model.work_dir / "validate_gif",
                                "validate/gif",
                                num_gifs,
                                fps=4,
                            )

            with self.timer("step"):
                with self.timer("sync_weights"):
                    self.update_rollout_weights()
                    self._sync_world_model_policy()
                with self.timer("generate_rollouts"):
                    print(f"[stage] real rollout start step={self.global_step}")
                    t0 = time.perf_counter()
                    env_metrics, real_env_interactions = self.generate_rollouts()
                    t1 = time.perf_counter()
                    duration = t1 - t0
                    print(
                        f"[stage] real rollout end step={self.global_step} "
                        f"(duration={duration:.2f}s)"
                    )
                    self._log_metrics(
                        {"time/real_rollout": duration}, self.global_step
                    )
                    drained = float(self._drain_wm_channel())
                    if drained > 0:
                        real_env_interactions = drained
                    self.real_env_interactions += real_env_interactions
                    real_env_metrics = {
                        "env/real_interactions": self.real_env_interactions
                    }
                use_real_rollout = self._should_use_real_rollout_for_vla()
                print(
                    f"[stage] real rollout for vla="
                    f"{'yes' if use_real_rollout else 'no'} step={self.global_step}"
                )
                if not use_real_rollout:
                    self.actor.clear_rollout_batch().wait()

                wm_metrics = {}
                if self.world_model is not None:
                    with self.timer("world_model"):
                        wm_step = self.global_step
                        will_init = not self.world_model.init_model
                        will_update = (
                            self.world_model.init_model
                            and self.cfg.update_gen_every_step > 0
                            and wm_step % self.cfg.update_gen_every_step == 0
                        )
                        if will_init:
                            print(
                                "[stage] world model demo finetuning start "
                                f"step={self.global_step}"
                            )
                        elif will_update:
                            print(
                                "[stage] world model update start "
                                f"step={self.global_step}"
                            )
                        wm_update_t0 = time.perf_counter()
                        wm_metrics.update(
                            self.world_model.maybe_update_world_model(
                                wm_step,
                                init_step_logger=self._log_wm_init_step_metrics,
                            )
                        )
                        wm_update_t1 = time.perf_counter()
                        if wm_metrics:
                            update_duration = wm_update_t1 - wm_update_t0
                            if any(k.startswith("wm_init/") for k in wm_metrics):
                                print(
                                    "[stage] world model demo finetuning end "
                                    f"(duration={update_duration:.2f}s)"
                                )
                                self._log_metrics(
                                    {"time/wm_init": update_duration},
                                    self.global_step,
                                )
                            else:
                                print(
                                    "[stage] world model update end "
                                    f"(duration={update_duration:.2f}s)"
                                )
                                self._log_metrics(
                                    {"time/wm_update": update_duration},
                                    self.global_step,
                                )
                        # MBPO starts
                        if self.world_model.init_model and self.world_model.should_start_mbpo(wm_step):
                            if not self.world_model.init_gen:
                                for _ in range(self.cfg.init_gen_times):
                                    print(
                                        "[stage] world model generation start "
                                        f"(init) step={self.global_step}"
                                    )
                                    t0 = time.perf_counter()
                                    imagined = self.world_model.generate_imagined_rollout(
                                        self.cfg.algorithm.n_chunk_steps
                                    )
                                    t1 = time.perf_counter()
                                    duration = t1 - t0
                                    print(
                                        "[stage] world model generation end "
                                        f"(duration={duration:.2f}s)"
                                    )
                                    self._log_metrics(
                                        {"time/wm_gen": duration}, self.global_step
                                    )
                                    wm_gen_total_duration += duration
                                    wm_gen_total_calls += 1
                                    if imagined is not None:
                                        self.actor.append_rollout_batch(imagined).wait()
                                self.world_model.init_gen = True
                            elif (
                                self.cfg.gen_every_steps > 0
                                and wm_step % self.cfg.gen_every_steps == 0
                            ):
                                print(
                                    "[stage] world model generation start "
                                    f"step={self.global_step}"
                                )
                                t0 = time.perf_counter()
                                imagined = self.world_model.generate_imagined_rollout(
                                    self.cfg.algorithm.n_chunk_steps
                                )
                                t1 = time.perf_counter()
                                duration = t1 - t0
                                print(
                                    "[stage] world model generation end "
                                    f"(duration={duration:.2f}s)"
                                )
                                self._log_metrics(
                                    {"time/wm_gen": duration}, self.global_step
                                )
                                wm_gen_total_duration += duration
                                wm_gen_total_calls += 1
                                if imagined is not None:
                                    self.actor.append_rollout_batch(imagined).wait()

                wm_status_metrics = {}
                if self.world_model is not None:
                    wm_status_metrics = self._world_model_status_metrics(
                        wm_metrics=wm_metrics,
                        use_real_rollout=use_real_rollout,
                    )

                if self.world_model is not None and not self.world_model.init_model:
                    print(
                        "[stage] skipping vla update until world model init completes"
                    )
                    self.actor.clear_rollout_batch().wait()

                has_rollout = self.actor.has_rollout_batch().wait()[0]
                actor_rollout_metrics = [{}]
                actor_training_metrics = [{}]
                if has_rollout:
                    with self.timer("cal_adv_and_returns"):
                        actor_futures = self.actor.compute_advantages_and_returns()
                        actor_rollout_metrics = actor_futures.wait()

                    with self.timer("actor_training"):
                        actor_training_futures = self.actor.run_training()
                        actor_training_metrics = actor_training_futures.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()

            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }
            self._log_metrics(env_metrics, _step)
            self._log_metrics(rollout_metrics, _step)
            self._log_metrics(time_metrics, _step)
            self._log_metrics(training_metrics, _step)
            self._log_metrics(real_env_metrics, _step)
            if wm_metrics:
                non_init_wm_metrics = {
                    k: v for k, v in wm_metrics.items() if not k.startswith("wm_init/")
                }
                if non_init_wm_metrics:
                    self._log_metrics(non_init_wm_metrics, _step)
            if wm_status_metrics:
                self._log_metrics(wm_status_metrics, _step)

            logging_metrics = {}
            logging_metrics.update(time_metrics)
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            logging_metrics.update(real_env_metrics)
            if wm_metrics:
                logging_metrics.update(
                    {
                        k: v
                        for k, v in wm_metrics.items()
                        if not k.startswith("wm_init/")
                    }
                )
            if wm_status_metrics:
                logging_metrics.update(wm_status_metrics)

            global_pbar.set_postfix(logging_metrics)
            global_pbar.update(1)

        if wm_gen_total_calls > 0:
            print(
                "[stage] world model generation total "
                f"calls={wm_gen_total_calls} "
                f"duration={wm_gen_total_duration:.2f}s "
                f"avg={wm_gen_total_duration / wm_gen_total_calls:.2f}s"
            )
        elif self.world_model is not None:
            print("[stage] world model generation total calls=0 duration=0.00s")
        self.metric_logger.finish()

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
