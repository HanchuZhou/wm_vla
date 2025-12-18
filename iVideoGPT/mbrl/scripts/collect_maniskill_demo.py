#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from dm_env import specs
from tqdm import tqdm

import maniskill_env
from replay_buffer import ReplayBufferStorage


def parse_args():
    parser = argparse.ArgumentParser(description="Collect or replay ManiSkill demonstrations for MBPO.")
    parser.add_argument("--task", type=str, required=True, help="ManiSkill task id, e.g. PutOnPlateInScene25Main-v3")
    parser.add_argument("--episodes", type=int, default=2,
                        help="Number of episodes to record or replay (-1 to convert the entire trajectory file).")
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
    parser.add_argument("--trajectory_path", type=Path,
                        help="Optional ManiSkill .h5 trajectory file to replay instead of sampling random actions.")
    parser.add_argument("--metadata_path", type=Path,
                        help="Optional metadata .json for --trajectory_path (defaults to the same stem).")
    parser.add_argument("--success_only", action="store_true",
                        help="When replaying trajectories, only keep episodes marked as successful.")
    return parser.parse_args()


def collect_random(env, storage, action_shape, num_episodes):
    for ep in range(num_episodes):
        time_step = env.reset()
        storage.add(time_step)
        steps = 0

        while not time_step.last():
            action = np.random.uniform(-1.0, 1.0, size=action_shape).astype(np.float32)
            time_step = env.step(action)
            storage.add(time_step)
            steps += 1

        print(f"[collect] Episode {ep + 1}/{num_episodes} finished with {steps} steps.")


def replay_trajectories(env, storage, args):
    trajectory_path = args.trajectory_path
    metadata_path = args.metadata_path or trajectory_path.with_suffix(".json")

    with metadata_path.open("r") as f:
        traj_meta = json.load(f)
    episodes = traj_meta["episodes"]
    if args.success_only:
        episodes = [ep for ep in episodes if ep.get("success")]
    if args.episodes > 0:
        episodes = episodes[: args.episodes]
    total = len(episodes)
    print(f"[replay] Using {total} trajectories from {trajectory_path.name}")

    with h5py.File(trajectory_path, "r") as traj_file:
        for idx, episode in enumerate(tqdm(episodes, desc="replay", unit="traj")):
            episode_id = episode["episode_id"]
            dataset = traj_file[f"traj_{episode_id}"]
            actions = np.array(dataset["actions"], dtype=np.float32)
            seed = episode.get("episode_seed")
            time_step = env.reset(seed=seed)
            storage.add(time_step)
            steps = 0
            for action in actions:
                time_step = env.step(action)
                storage.add(time_step)
                steps += 1
                if time_step.last():
                    break
            success_flag = episode.get("success", False)
            print(f"[replay] Episode {idx + 1}/{total} (id={episode_id}, success={success_flag}) "
                  f"produced {steps} steps.")


def main():
    args = parse_args()
    if isinstance(args.obj_set, str) and args.obj_set.lower() in {"none", "null"}:
        args.obj_set = None

    if args.trajectory_path is not None:
        args.trajectory_path = args.trajectory_path.expanduser().resolve()
        if args.metadata_path is not None:
            args.metadata_path = args.metadata_path.expanduser().resolve()
        else:
            args.metadata_path = args.trajectory_path.with_suffix(".json")
        with args.metadata_path.open("r") as f:
            meta = json.load(f)
        env_kwargs = meta.get("env_info", {}).get("env_kwargs", {})
        if args.control_mode is None and env_kwargs.get("control_mode"):
            args.control_mode = env_kwargs["control_mode"]
        if env_kwargs.get("sim_backend"):
            args.sim_backend = env_kwargs["sim_backend"]
        if args.obj_set is None and env_kwargs.get("obj_set") is not None:
            args.obj_set = env_kwargs["obj_set"]

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

    if args.trajectory_path is None:
        collect_random(env, storage, action_spec.shape, args.episodes)
    else:
        replay_trajectories(env, storage, args)


if __name__ == "__main__":
    main()
