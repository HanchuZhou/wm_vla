import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import List

import h5py
import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("OpenCV is required for resizing LIBERO frames.") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "iVideoGPT"))
sys.path.insert(0, str(REPO_ROOT / "iVideoGPT/mbrl"))

from replay_buffer import save_episode


DEFAULT_IMAGE_KEYS = [
    "agentview_rgb",
    "agentview_image",
    "agentview",
    "rgb",
    "image",
    "frontview_rgb",
    "frontview_image",
    "eye_in_hand_rgb",
    "eye_in_hand_image",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO hdf5 demonstrations into MBPO .npz episodes."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "libero_raw",
        help="Directory containing LIBERO hdf5 datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "iVideoGPT/mbrl/demonstrations",
        help="Output directory for MBPO demos (suite subfolders will be created).",
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=["all"],
        help="Suites to convert (e.g. libero_spatial libero_goal libero_10).",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="",
        help="Override observation image key (default: auto-detect).",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=3,
        help="Number of frames to stack.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resize images to this square size.",
    )
    parser.add_argument(
        "--filter-key",
        type=str,
        default="",
        help="Optional mask key (e.g. train) to filter demos.",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Flip frames vertically to match environment rendering.",
    )
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Rotate frames 180 degrees to match environment rendering.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=-1,
        help="Optional cap per hdf5 file (-1 for all).",
    )
    return parser.parse_args()


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _find_image_key(obs_group: h5py.Group, override: str) -> str:
    if override:
        if override not in obs_group:
            raise KeyError(f"Image key {override} not found in obs group.")
        return override
    for key in DEFAULT_IMAGE_KEYS:
        if key in obs_group:
            return key
    for key in obs_group:
        shape = obs_group[key].shape
        if len(shape) >= 3 and shape[-1] == 3:
            return key
    raise KeyError("Unable to auto-detect an RGB observation key.")


def _sorted_demo_names(h5_file: h5py.File, filter_key: str) -> List[str]:
    if filter_key:
        demos = [
            elem.decode("utf-8")
            for elem in np.array(h5_file[f"mask/{filter_key}"])
        ]
    else:
        demos = list(h5_file["data"].keys())
    demos = sorted(demos, key=lambda name: int(name.split("_")[-1]))
    return demos


def _load_obs_sequences(group: h5py.Group, image_key: str) -> tuple[np.ndarray, np.ndarray]:
    obs_seq = np.array(group[f"obs/{image_key}"])
    if "next_obs" in group and image_key in group["next_obs"]:
        next_obs_seq = np.array(group[f"next_obs/{image_key}"])
    else:
        if obs_seq.shape[0] > 1:
            next_obs_seq = np.concatenate([obs_seq[1:], obs_seq[-1:]], axis=0)
        else:
            next_obs_seq = obs_seq.copy()
    return obs_seq, next_obs_seq


def _build_stacked_obs(
    obs_seq: list[np.ndarray],
    frame_stack: int,
    image_size: int,
    flip_vertical: bool,
    rotate_180: bool,
) -> np.ndarray:
    frames = deque(maxlen=frame_stack)
    stacked = []
    for idx, frame in enumerate(obs_seq):
        frame = np.squeeze(frame)
        if rotate_180:
            frame = np.rot90(frame, k=2)
        if flip_vertical:
            frame = np.flipud(frame)
        frame = _resize(_to_uint8(frame), image_size)
        frame = frame.transpose(2, 0, 1)
        if idx == 0:
            for _ in range(frame_stack):
                frames.append(frame)
        else:
            frames.append(frame)
        stacked.append(np.concatenate(list(frames), axis=0))
    return np.stack(stacked, axis=0)


def _build_episode(
    group: h5py.Group,
    image_key: str,
    frame_stack: int,
    image_size: int,
    flip_vertical: bool,
    rotate_180: bool,
) -> dict:
    actions = np.array(group["actions"]).astype(np.float32)
    obs_seq, next_obs_seq = _load_obs_sequences(group, image_key)

    if obs_seq.shape[0] == actions.shape[0] + 1:
        obs_time = obs_seq
    else:
        obs_time = np.concatenate([obs_seq[:1], next_obs_seq], axis=0)

    stacked_obs = _build_stacked_obs(
        list(obs_time),
        frame_stack,
        image_size,
        flip_vertical,
        rotate_180,
    )

    episode_len = actions.shape[0] + 1
    action_dim = actions.shape[-1]
    action_seq = np.zeros((episode_len, action_dim), dtype=np.float32)
    action_seq[1:] = actions

    reward_seq = np.zeros((episode_len, 1), dtype=np.float32)
    if "rewards" in group:
        rewards = np.array(group["rewards"]).reshape(-1)
        reward_seq[1:, 0] = rewards

    discount_seq = np.ones((episode_len, 1), dtype=np.float32)
    done_flags = None
    if "dones" in group:
        done_flags = np.array(group["dones"]).reshape(-1).astype(bool)
    elif "terminals" in group:
        done_flags = np.array(group["terminals"]).reshape(-1).astype(bool)
    if done_flags is not None:
        for idx, done in enumerate(done_flags):
            if done:
                discount_seq[idx + 1, 0] = 0.0
        if not done_flags.any():
            discount_seq[-1, 0] = 0.0
    else:
        discount_seq[-1, 0] = 0.0

    return {
        "observation": stacked_obs.astype(np.uint8),
        "action": action_seq,
        "reward": reward_seq,
        "discount": discount_seq,
    }


def convert_suite(
    suite_dir: Path,
    output_dir: Path,
    image_key: str,
    frame_stack: int,
    image_size: int,
    filter_key: str,
    max_demos: int,
    flip_vertical: bool,
    rotate_180: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_files = sorted(suite_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"[skip] No hdf5 files found in {suite_dir}")
        return
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as h5_file:
            if "data" not in h5_file:
                print(f"[skip] {hdf5_path.name} missing data group")
                continue
            demo_names = _sorted_demo_names(h5_file, filter_key)
            if not demo_names:
                print(f"[skip] {hdf5_path.name} has no demos")
                continue
            if max_demos > 0:
                demo_names = demo_names[:max_demos]
            obs_group = h5_file[f"data/{demo_names[0]}/obs"]
            resolved_image_key = _find_image_key(obs_group, image_key)
            raw_info = h5_file["data"].attrs.get("problem_info", "{}")
            if isinstance(raw_info, bytes):
                raw_info = raw_info.decode("utf-8")
            try:
                task_info = json.loads(raw_info)
            except json.JSONDecodeError:
                task_info = {}
            task_name = (
                task_info.get("task_name")
                or task_info.get("task")
                or task_info.get("problem_name")
                or hdf5_path.stem
            )
            for demo_name in demo_names:
                group = h5_file[f"data/{demo_name}"]
                episode = _build_episode(
                    group,
                    resolved_image_key,
                    frame_stack=frame_stack,
                    image_size=image_size,
                    flip_vertical=flip_vertical,
            rotate_180=rotate_180,
                )
                out_name = f"{task_name}_{hdf5_path.stem}_{demo_name}.npz"
                save_episode(episode, output_dir / out_name)


def main() -> None:
    args = parse_args()
    suites = args.suites
    if "all" in suites:
        suites = [
            "libero_spatial",
            "libero_goal",
            "libero_object",
            "libero_10",
            "libero_90",
            "libero_100",
        ]
    for suite in suites:
        suite_dir = args.download_dir / suite
        if not suite_dir.exists():
            print(f"[skip] Suite {suite} not found in {args.download_dir}")
            continue
        output_dir = args.output_dir / suite
        print(f"[convert] {suite_dir} -> {output_dir}")
        convert_suite(
            suite_dir,
            output_dir,
            image_key=args.image_key,
            frame_stack=args.frame_stack,
            image_size=args.image_size,
            filter_key=args.filter_key,
            max_demos=args.max_demos,
            flip_vertical=args.flip_vertical,
            rotate_180=args.rotate_180,
        )


if __name__ == "__main__":
    main()
