# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import maniskill_env
import drq_utils
from logger import Logger
from omegaconf import OmegaConf
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.max_gifs_per_type = getattr(cfg, "max_gifs_per_type", 3)
        self.max_eval_videos = getattr(cfg, "max_eval_videos", 3)
        self.wandb_run = self._init_wandb()
        drq_utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.agent,
        )
        self.timer = drq_utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def _init_wandb(self):
        wb_cfg = getattr(self.cfg, "wandb", None)
        if wb_cfg is None or not getattr(wb_cfg, "enable", False):
            return None
        try:
            import wandb
        except ImportError:
            print("wandb not installed; skipping wandb logging.")
            return None
        entity = getattr(wb_cfg, "entity", None) or getattr(wb_cfg, "team", None)
        project = getattr(wb_cfg, "project", self.cfg.task_name)
        group = getattr(wb_cfg, "group", self.cfg.exp_name)
        mode = getattr(wb_cfg, "mode", "online")
        tags = []
        team = getattr(wb_cfg, "team", None)
        org = getattr(wb_cfg, "org", None)
        if team:
            tags.append(f"team:{team}")
        if org:
            tags.append(f"org:{org}")
        base_run_name = self.cfg.exp_name
        hydra_name = getattr(self.cfg, "name", None)
        if hydra_name and str(hydra_name) != "0":
            base_run_name = f"{base_run_name}_{hydra_name}"
        # allow both wandb.run_name and wandb.name as a suffix
        custom_suffix = getattr(wb_cfg, "run_name", None) or getattr(wb_cfg, "name", None)
        run_name = f"{base_run_name}_{custom_suffix}" if custom_suffix else base_run_name
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        run = wandb.init(
            entity=entity,
            project=project,
            group=group,
            name=run_name,
            mode=mode,
            tags=tags,
            config=config_dict,
            dir=str(self.work_dir),
        )
        try:
            run.define_metric("eval/step")
            run.define_metric("eval/*", step_metric="eval/step")
        except Exception:
            pass
        return run

    def should_save_gif(self, frame):
        interval = getattr(self.cfg, "gif_save_every_frames", None)
        if interval is None or interval <= 0:
            return True
        return (frame % interval) == 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, wandb_run=self.wandb_run)
        # create envs
        self.train_env = maniskill_env.make(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            self.cfg.obs_mode,
            self.cfg.control_mode,
            self.cfg.sim_backend,
            self.cfg.sim_freq,
            self.cfg.control_freq,
            self.cfg.max_episode_steps,
            self.cfg.obj_set,
            self.cfg.render_mode,
            self.cfg.shader_pack,
            self.cfg.render_backend,
        )
        self.eval_env = maniskill_env.make(
            self.cfg.task_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            self.cfg.obs_mode,
            self.cfg.control_mode,
            self.cfg.sim_backend,
            self.cfg.sim_freq,
            self.cfg.control_freq,
            self.cfg.max_episode_steps,
            self.cfg.obj_set,
            self.cfg.render_mode,
            self.cfg.shader_pack,
            self.cfg.render_backend,
        )
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        self.replay_storage = ReplayBufferStorage(
            data_specs, self.work_dir / "buffer"
        )

        demo_path = (
            os.path.join(self.cfg.demo_path_prefix, self.cfg.task_name)
            if self.cfg.demo
            else None
        )
        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            demo_path,
        )
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward, total_success = 0, 0, 0, 0
        eval_until_episode = drq_utils.Until(
            self.cfg.num_eval_episodes, bar_name="eval_eps"
        )

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            episode_success = 0
            record_episode = (episode < self.max_eval_videos) and self.should_save_gif(self.global_frame)
            self.video_recorder.init(self.eval_env, enabled=record_episode)
            while not time_step.last():
                with torch.no_grad(), drq_utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env, time_step.reward)
                total_reward += time_step.reward
                episode_success += time_step.success
                step += 1

            total_success += episode_success >= 1.0
            episode += 1
            if self.video_recorder.enabled:
                self.video_recorder.save(f"{self.global_frame}_ep{episode - 1}.gif")

        eval_logs = {
            "episode_reward": total_reward / episode,
            "episode_success": total_success / episode,
            "episode_length": step * self.cfg.action_repeat / episode,
            "episode": self.global_episode,
            "step": self.global_step,
        }
        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for k, v in eval_logs.items():
                log(k, v)
        if self.wandb_run is not None:
            try:
                eval_payload = {f"eval/{k}": v for k, v in eval_logs.items()}
                eval_payload["eval/step"] = self.global_step
                self.wandb_run.log(eval_payload, step=self.global_frame, commit=True)
            except Exception:
                pass

    def train(self):
        assert (
            self.cfg.num_seed_frames
            == self.cfg.agent.num_expl_steps * self.cfg.action_repeat
        )

        train_until_step = drq_utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat, bar_name="train_step"
        )
        seed_until_step = drq_utils.Until(
            self.cfg.num_seed_frames, self.cfg.action_repeat
        )
        eval_every_step = drq_utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward, episode_success = 0, 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.gif")
                elapsed_time, total_time = self.timer.reset()
                episode_frame = episode_step * self.cfg.action_repeat
                with self.logger.log_and_dump_ctx(
                    self.global_frame, ty="train"
                ) as log:
                    log("fps", episode_frame / elapsed_time if elapsed_time > 0 else 0.0)
                    log("total_time", total_time)
                    log("episode_reward", episode_reward)
                    log("episode_success", episode_success >= 1.0)
                    log("episode_length", episode_frame)
                    log("episode", self.global_episode)
                    log("buffer_size", len(self.replay_storage))
                    log("step", self.global_step)

                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0
                episode_success = 0

            if eval_every_step(self.global_step):
                self.logger.log("eval_total_time", self.timer.total_time(), self.global_frame)
                self.eval()

            with torch.no_grad(), drq_utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation, self.global_step, eval_mode=False
                )

            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.agent_update_times):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            episode_success += time_step.success
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="maniskill_drq_config")
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()

    workspace.train()


if __name__ == "__main__":
    main()
