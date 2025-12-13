#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from dm_env import specs

import maniskill_env
from replay_buffer import ReplayBufferStorage


def parse_args():
    parser = argparse.ArgumentParser(description="Collect ManiSkill demonstrations for MBPO.")
    parser.add_argument("--task", type=str, required=True, help="ManiSkill task id, e.g. PutOnPlateInScene25Main-v3")
    parser.add_argument("--episodes", type=int, default=2, help="Number of random episodes to record")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parents[1] / "demonstrations",
                        help="Directory to store generated episodes")
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--obs_mode", type=str, default="rgb+segmentation")
    parser.add_argument("--control_mode", type=str, default=None)
    parser.add_argument("--sim_backend", type=str, default="gpu")
    parser.add_argument("--sim_freq", type=int, default=500)
    parser.add_argument("--control_freq", type=int, default=5)
    parser.add_argument("--max_episode_steps", type=int, default=80)
    parser.add_argument("--obj_set", type=str, default="train")
    parser.add_argument("--render_mode", type=str, default="all")
    parser.add_argument("--shader_pack", type=str, default="default")
    parser.add_argument("--size", type=int, nargs=2, default=(64, 64), metavar=("H", "W"),
                        help="Observation height and width used for saving demos")
    return parser.parse_args()


def main():
    args = parse_args()
    env = maniskill_env.make(
        args.task,
        args.frame_stack,
        args.action_repeat,
        args.seed,
        args.obs_mode,
        args.control_mode,
        args.sim_backend,
        args.sim_freq,
        args.control_freq,
        args.max_episode_steps,
        args.obj_set,
        args.render_mode,
        args.shader_pack,
        size=tuple(args.size),
    )

    obs_spec = env.observation_spec()
    action_spec = env.action_spec()
    reward_spec = specs.Array((1,), np.float32, "reward")
    discount_spec = specs.Array((1,), np.float32, "discount")

    task_dir = args.output_dir / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    storage = ReplayBufferStorage((obs_spec, action_spec, reward_spec, discount_spec), task_dir)

    for ep in range(args.episodes):
        time_step = env.reset()
        storage.add(time_step)
        steps = 0

        while not time_step.last():
            action = np.random.uniform(-1.0, 1.0, size=action_spec.shape).astype(np.float32)
            time_step = env.step(action)
            storage.add(time_step)
            steps += 1

        print(f"[collect] Episode {ep + 1}/{args.episodes} finished with {steps} steps.")


if __name__ == "__main__":
    main()
