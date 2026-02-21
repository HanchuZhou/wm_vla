# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict
from pathlib import Path
import glob
from typing import Hashable

import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

np.set_printoptions(precision=3, suppress=True)


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def maybe_resize_observation(obs, image_size):
    if image_size is None:
        return obs
    target = int(image_size)
    if obs.shape[-2] == target and obs.shape[-1] == target:
        return obs

    squeeze_batch = False
    obs_tensor = torch.from_numpy(obs.astype(np.float32, copy=False))
    if obs_tensor.ndim == 3:
        obs_tensor = obs_tensor.unsqueeze(0)
        squeeze_batch = True
    resized = nn.functional.interpolate(
        obs_tensor, size=(target, target), mode='bilinear', align_corners=False
    )
    if squeeze_batch:
        resized = resized.squeeze(0)

    if obs.dtype == np.uint8:
        return resized.clamp(0.0, 255.0).round().to(torch.uint8).cpu().numpy()
    return resized.cpu().numpy().astype(obs.dtype, copy=False)


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episodes = {}
        self._preload()

    def __len__(self):
        return self._num_transitions

    def _get_stream_buffer(self, stream_id: Hashable):
        if stream_id not in self._current_episodes:
            self._current_episodes[stream_id] = defaultdict(list)
        return self._current_episodes[stream_id]

    def _finalize_stream(self, stream_id: Hashable, force_terminal: bool = False):
        stream = self._current_episodes.get(stream_id)
        if stream is None:
            return None
        if not stream:
            self._current_episodes.pop(stream_id, None)
            return None

        if force_terminal and "discount" in stream and len(stream["discount"]) > 0:
            stream["discount"][-1] = np.zeros_like(stream["discount"][-1])

        episode = dict()
        for spec in self._data_specs:
            values = stream[spec.name]
            if len(values) == 0:
                self._current_episodes.pop(stream_id, None)
                return None
            episode[spec.name] = np.array(values, spec.dtype)

        self._current_episodes.pop(stream_id, None)
        self._store_episode(episode)
        return episode

    def add(self, time_step, stream_id: Hashable = 0):
        stream = self._get_stream_buffer(stream_id)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            stream[spec.name].append(value)
        if time_step.last():
            return self._finalize_stream(stream_id, force_terminal=False)

    def close_stream(self, stream_id: Hashable = 0):
        return self._finalize_stream(stream_id, force_terminal=True)

    def close_all_streams(self):
        for stream_id in list(self._current_episodes.keys()):
            self._finalize_stream(stream_id, force_terminal=True)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)
        return self._replay_dir / eps_fn


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, demo_path=None, demo_image_size=None,
                 include_source=False):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._episode_source = dict()
        self._episode_uid = dict()
        self._next_episode_uid = 0
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._include_source = bool(include_source)

        self._num_direct_episodes = 0
        self._demo_image_size = (
            int(demo_image_size) if demo_image_size is not None else None
        )

        if demo_path is not None:
            files = glob.glob(os.path.join(demo_path, '*.npz'))
            if len(files) == 0:
                assert False
            print(f"[stage] demo preload start: {demo_path} ({len(files)} files)")
            loaded = 0
            for display in files:
                display = Path(display)
                if not self._store_episode(display):
                    assert False
                loaded += 1
            print(f"[stage] demo preload end: loaded {loaded} episodes")

    def _is_replay_file(self, eps_fn):
        eps_path = Path(eps_fn)
        replay_root = self._replay_dir
        try:
            eps_path.resolve().relative_to(replay_root.resolve())
            return True
        except ValueError:
            return False

    def _maybe_unlink_replay_file(self, eps_fn):
        if self._is_replay_file(eps_fn):
            Path(eps_fn).unlink(missing_ok=True)

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return (
            self._episodes[eps_fn],
            int(self._episode_source.get(eps_fn, 0)),
            int(self._episode_uid.get(eps_fn, -1)),
            eps_fn,
        )

    def _direct_store_episode(self, episode):
        eps_idx = self._num_direct_episodes
        eps_len = episode_len(episode)
        self._num_direct_episodes += 1
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        eps_path = self._replay_dir / eps_fn

        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._episode_source.pop(early_eps_fn, None)
            self._episode_uid.pop(early_eps_fn, None)
            self._size -= episode_len(early_eps)
        self._episode_fns.append(eps_path)
        # self._episode_fns.sort()
        self._episodes[eps_path] = episode
        self._episode_source[eps_path] = 0
        self._episode_uid[eps_path] = self._next_episode_uid
        self._next_episode_uid += 1
        self._size += eps_len
        return eps_path

    def _store_episode(self, eps_fn):
        eps_fn = Path(eps_fn)
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._episode_source.pop(early_eps_fn, None)
            self._episode_uid.pop(early_eps_fn, None)
            self._size -= episode_len(early_eps)
            self._maybe_unlink_replay_file(early_eps_fn)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._episode_source[eps_fn] = 0 if self._is_replay_file(eps_fn) else 1
        self._episode_uid[eps_fn] = self._next_episode_uid
        self._next_episode_uid += 1
        self._size += eps_len

        if not self._save_snapshot:
            self._maybe_unlink_replay_file(eps_fn)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        while True:
            episode, source_id, _, _ = self._sample_episode()
            length = episode_len(episode)
            if length > self._nstep:
                break
        # add +1 for the first dummy transition
        idx = np.random.randint(0, length - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        obs = maybe_resize_observation(obs, self._demo_image_size)
        next_obs = maybe_resize_observation(next_obs, self._demo_image_size)
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        sample = (obs, action, reward, discount, next_obs)
        if self._include_source:
            sample = (*sample, np.int64(source_id))
        return sample

    def __iter__(self):
        while True:
            yield self._sample()


class ReplaySegmentBuffer(ReplayBuffer):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, segment_length, demo_path=None,
                 demo_image_size=None, include_source=False):
        super().__init__(replay_dir, max_size, num_workers, nstep, discount,
                         fetch_every, save_snapshot, demo_path, demo_image_size,
                         include_source=include_source)
        self._segment_length = segment_length

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        for _ in range(1000):
            episode, source_id, episode_uid, _ = self._sample_episode()
            length = episode_len(episode)
            if length <= self._segment_length:
                continue
            # ensure the upper bound accounts for inclusive end index
            idx = np.random.randint(1, length - self._segment_length + 1)
            discounts = episode["discount"][idx: idx + self._segment_length].reshape(-1)
            # Reject malformed slices that contain terminal transitions in the middle.
            if discounts.size > 1 and np.any(discounts[:-1] <= 0):
                continue

            obs = episode['observation'][idx - 1: idx + self._segment_length - 1, -3:]
            obs = maybe_resize_observation(obs, self._demo_image_size)
            action = episode['action'][idx: idx + self._segment_length]
            reward = episode['reward'][idx: idx + self._segment_length]
            sample = (obs, action, reward)
            if self._include_source:
                sample = (
                    *sample,
                    np.int64(source_id),
                    np.int64(episode_uid),
                    np.int64(idx),
                )
            return sample
        raise RuntimeError("Failed to sample a valid replay segment after 1000 retries.")


def _worker_init_fn(worker_id):
    state = np.random.get_state()
    state_array = state[1]
    if isinstance(state_array, (list, tuple, np.ndarray)):
        base_seed = int(state_array[0])
    else:
        base_seed = int(state_array)
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, demo_path=None,
                       demo_image_size=None, include_source=False):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            demo_path=demo_path,
                            demo_image_size=demo_image_size,
                            include_source=include_source)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader


def make_segment_replay_loader(replay_dir, max_size, batch_size, num_workers,
                               save_snapshot, nstep, discount, segment_length,
                               demo_path=None, demo_image_size=None,
                               include_source=False):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplaySegmentBuffer(replay_dir,
                                   max_size_per_worker,
                                   num_workers,
                                   nstep,
                                   discount,
                                   fetch_every=1000,
                                   save_snapshot=save_snapshot,
                                   segment_length=segment_length,
                                   demo_path=demo_path,
                                   demo_image_size=demo_image_size,
                                   include_source=include_source)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader
