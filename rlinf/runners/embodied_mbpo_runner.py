import asyncio
import os
import time
from pathlib import Path
from typing import Optional

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
        gif_paths = sorted(
            base_path.rglob("*.gif"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        if not gif_paths:
            return
        gif_paths = gif_paths[:max_gifs]
        wandb = self.metric_logger.logger.get("wandb", None)
        if wandb is None:
            return
        payload = {}
        for idx, gif_path in enumerate(gif_paths):
            try:
                payload[f"{key_prefix}_{idx}"] = wandb.Video(
                    str(gif_path), fps=fps, format="gif"
                )
            except Exception:
                continue
        if payload:
            wandb.log(payload, step=self.global_step, commit=True)

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

        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        return env_metrics

    def evaluate(self):
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_results = env_futures.wait()
        rollout_futures.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        if self.world_model is not None and not self.world_model.init_model:
            print("[stage] world model demo finetuning start")
            t0 = time.perf_counter()
            wm_metrics = self.world_model.maybe_update_world_model(self.global_step)
            t1 = time.perf_counter()
            duration = t1 - t0
            print(f"[stage] world model demo finetuning end (duration={duration:.2f}s)")
            self.metric_logger.log({"time/wm_init": duration}, self.global_step)
            if wm_metrics:
                self.metric_logger.log(wm_metrics, self.global_step)
            if not self.world_model.init_model:
                print("[stage] world model demo finetuning skipped (no demo data)")
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
                    self.metric_logger.log(data=eval_metrics, step=_step)
                    save_eval_and_val_gif = bool(
                        self.cfg.runner.get("save_eval_and_val_gif", True)
                    )
                    num_gifs = int(self.cfg.runner.get("num_gif_every_steps", 0))
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
                            wm_val_metrics = self.world_model.validate(
                                global_step=self.global_step,
                                num_gifs=num_gifs,
                                fps=4,
                            )
                            if wm_val_metrics:
                                self.metric_logger.log(wm_val_metrics, step=_step)
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
                    env_metrics = self.generate_rollouts()
                    t1 = time.perf_counter()
                    duration = t1 - t0
                    print(
                        f"[stage] real rollout end step={self.global_step} "
                        f"(duration={duration:.2f}s)"
                    )
                    self.metric_logger.log(
                        {"time/real_rollout": duration}, self.global_step
                    )
                    self._drain_wm_channel()
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
                            self.world_model.maybe_update_world_model(wm_step)
                        )
                        wm_update_t1 = time.perf_counter()
                        if wm_metrics:
                            update_duration = wm_update_t1 - wm_update_t0
                            if any(k.startswith("wm_init/") for k in wm_metrics):
                                print(
                                    "[stage] world model demo finetuning end "
                                    f"(duration={update_duration:.2f}s)"
                                )
                                self.metric_logger.log(
                                    {"time/wm_init": update_duration},
                                    self.global_step,
                                )
                            else:
                                print(
                                    "[stage] world model update end "
                                    f"(duration={update_duration:.2f}s)"
                                )
                                self.metric_logger.log(
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
                                    self.metric_logger.log(
                                        {"time/wm_gen": duration}, self.global_step
                                    )
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
                                self.metric_logger.log(
                                    {"time/wm_gen": duration}, self.global_step
                                )
                                if imagined is not None:
                                    self.actor.append_rollout_batch(imagined).wait()

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
            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)
            if wm_metrics:
                self.metric_logger.log(wm_metrics, _step)

            logging_metrics = {}
            logging_metrics.update(time_metrics)
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)
            if wm_metrics:
                logging_metrics.update(wm_metrics)

            global_pbar.set_postfix(logging_metrics)
            global_pbar.update(1)

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
