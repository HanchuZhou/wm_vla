# 🌏 iVideoGPT: Interactive VideoGPTs are Scalable World Models (NeurIPS 2024)

[[Project Page]](https://thuml.github.io/iVideoGPT/) [[Paper]](https://arxiv.org/abs/2405.15223) [[Models]](https://huggingface.co/collections/thuml/ivideogpt-674c59cae32231024d82d6c5) [[Poster]](https://manchery.github.io/assets/pub/nips2024_ivideogpt/poster.pdf) [[Slides]](https://manchery.github.io/assets/pub/nips2024_ivideogpt/slides.pdf) [[Blog (In Chinese)]](https://mp.weixin.qq.com/s/D94aamdqtO9WLekr4BSCUw)

This repo provides official code and checkpoints for iVideoGPT, a generic and efficient world model architecture that has been pre-trained on millions of human and robotic manipulation trajectories. 

![architecture](assets/architecture.png)

## 🔥 News

- 🚩 **2025.09.18**: [RLVR-World](https://github.com/thuml/RLVR-World) has been accepted by NeurIPS 2025, congrats!
- 🚩 **2025.05.21**: We are excited to release a new work, [RLVR-World](https://github.com/thuml/RLVR-World), demonstrating that iVideoGPTs can be improved by reinforcement learning with verifiable rewards (RLVR)!
- 🚩 **2024.11.01**: NeurIPS 2024 camera-ready version is released on [arXiv](https://arxiv.org/abs/2405.15223v3).
- 🚩 **2024.09.26**: iVideoGPT has been accepted by NeurIPS 2024, congrats!
- 🚩 **2024.08.31**: Training code is released.
- 🚩 **2024.05.31**: Project website with video samples is released.
- 🚩 **2024.05.30**: Model pre-trained on Open X-Embodiment and inference code are released.
- 🚩 **2024.05.27**: Our paper is released on [arXiv](https://arxiv.org/abs/2405.15223v1).

## 🛠️ Installation

```bash
conda create -n ivideogpt python==3.9
conda activate ivideogpt
pip install -r requirements.txt
```

To evaluate the FVD metric, download the [pretrained I3D model](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1) into `pretrained_models/i3d/i3d_torchscript.pt`.

## 🤗 Models

At the moment we provide the following pre-trained models:

| Model | Resolution | Action-conditioned | Goal-conditioned | Tokenizer Size | Transformer Size |
| ---- | ---- | ---- | ---- | ---- | ---- |
| [ivideogpt-oxe-64-act-free](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free) | 64x64 | No | No | 114M   |  138M    |
| [ivideogpt-oxe-64-act-free-medium](https://huggingface.co/thuml/ivideogpt-oxe-64-act-free-medium) | 64x64 | No | No |  114M   |  436M    |
| [ivideogpt-oxe-64-goal-cond](https://huggingface.co/thuml/ivideogpt-oxe-64-goal-cond) | 64x64 | No | Yes | 114M   |  138M    |
| [ivideogpt-oxe-256-act-free](https://huggingface.co/thuml/ivideogpt-oxe-256-act-free) | 256x256 | No | No | 310M   |  138M    |

If no network connection to Hugging Face, you can manually download from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/ef7d94c798504587a95e/).

**Notes**:

- Due to the heterogeneity of action spaces, we currently do not have an action-conditioned prediction model on OXE.
- Pre-trained models at 256x256 resolution may not perform best due to insufficient training, but can serve as a good starting point for downstream fine-tuning.

<details>
  <summary><b>More models on downstream tasks</b></summary>
  <br>
  
| Model | Resolution | Action-conditioned | Goal-conditioned | Tokenizer Size | Transformer Size |
| ---- | ---- | ---- | ---- | ---- | ---- |
| [ivideogpt-bair-64-act-free](https://huggingface.co/thuml/ivideogpt-bair-64-act-free) | 64x64 | No | No |  114M   |  138M    |
| [ivideogpt-bair-64-act-cond](https://huggingface.co/thuml/ivideogpt-bair-64-act-cond) | 64x64 | Yes | No | 114M   |  138M    |
| [ivideogpt-robonet-64-act-cond](https://huggingface.co/thuml/ivideogpt-robonet-64-act-cond) | 64x64 | Yes | No |  114M   |  138M    |
| [ivideogpt-vp2-robosuite-64-act-cond](https://huggingface.co/thuml/ivideogpt-vp2-robosuite-64-act-cond) | 64x64 | Yes | No |  114M   |  138M    |
| [ivideogpt-vp2-robodesk-64-act-cond](https://huggingface.co/thuml/ivideogpt-vp2-robodesk-64-act-cond) | 64x64 | Yes | No |  114M   |  138M    |

- We are sorry that the checkpoints for RoboNet at 256x256 resolution were deleted by mistake during a disk cleanup, we will retrain and release them as our computational resources become idle.
</details>

## 📦 Data Preparation

**Open X-Embodiment**: Download datasets from [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) and extract single episodes as `.npz` files:

```bash
python datasets/oxe_data_converter.py --dataset_name {dataset name, e.g. bridge} --input_path {path to downloaded OXE} --output_path {path to stored npz}
```

To replicate our pre-training on OXE, you need to extract all datasets listed under `OXE_SELECT` in `ivideogpt/data/dataset_mixes.py`.

See instructions at [`datasets`](/datasets) on preprocessing more datasets.

## 🧠 Run WM-RL with ManiSkill

Run the MBPO + iVideoGPT pipeline on ManiSkill tasks from the repository root:

1. **Collect demos** (seeds the replay buffer when `demo=true`):
   ```bash
   python -m mani_skill.utils.download_demo PickCube-v1 -o iVideoGPT/mbrl/demonstrations
   ```

   The download ManiSkill trajectories should be converted to `.npz` demonstrations:
   ```bash
   # PickCube-v1
   python iVideoGPT/mbrl/scripts/collect_maniskill_demo.py \
       --task StackCube-v1 \
       --trajectory_path iVideoGPT/mbrl/demonstrations/StackCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
       --episodes 5 --success_only --obj_set null --max_episode_steps 50
   ```

   Replace `task` with `PickCube-v1`, `LiftPegUpright-v1`, `PegInsertionSide-v1`, `PushT-v1`, or `StackCube-v1` to prepare demos for the other supported tasks.
2. **Set runtime environment** (adjust paths as needed):
   ```bash
   source .venv/bin/activate
   export PYTHONPATH=$PWD:$PWD/iVideoGPT:$PWD/iVideoGPT/mbrl
   export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
   ```
3. **Launch training** (pin to a specific GPU if desired):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python iVideoGPT/mbrl/train_maniskill_mbpo.py \
       task=maniskill/pick_cube num_train_frames=1000002 demo=true
   ```
   Swap task with `pick_cube`, `lift_peg_upright`, `peg_insertion_side`, `push_t`, or `stack_cube` to launch the additional benchmarks.

   Add `render_backend=sapien_cpu` when encounter error on `CUDA error at /__w/SAPIEN/SAPIEN/3rd_party/sapien-vulkan-2/src/core/buffer.cpp`.

Logs and checkpoints are written to `log_mbrl/<date>/maniskill_mbpo_<task>_*` with the current Hydra configuration.

## 🧠 Run WM-RL with LIBERO

Prepare LIBERO demonstrations and enable MBPO demo seeding:

1. **Download LIBERO demos** (official release):
   ```bash
   mkdir -p iVideoGPT/datasets/libero_raw
   wget -O iVideoGPT/datasets/libero_raw/libero_spatial.zip \
     https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip
   unzip -o iVideoGPT/datasets/libero_raw/libero_spatial.zip -d iVideoGPT/datasets/libero_raw
   ```
2. **Convert to MBPO `.npz` demos** (stored under `iVideoGPT/mbrl/demonstrations/<suite>`):
   ```bash
   python iVideoGPT/datasets/convert_libero_demos.py \
       --download-dir iVideoGPT/datasets/libero_raw \
       --output-dir iVideoGPT/mbrl/demonstrations \
       --suites libero_spatial
       --rotate_180
   ```
3. **Launch LIBERO MBPO** (demo seeding enabled by default in configs):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python iVideoGPT/mbrl/train_libero_mbpo_openpi.py \
       --config-name libero_spatial_mbpo_openpi_pi05_config \
       task_name=libero_spatial demo=true
   ```

For the full WM-RL + VLA pipeline inside the container, run:
```bash
MBPO_TASK_NAME=libero_spatial MBPO_CONFIG=libero_spatial_mbpo_openpi_pi05_config \
  bash examples/embodiment/run_wm_rl_vla_libero_spatial_mbpo_openpi_pi05.sh
```

## 🚀 Inference Examples

For action-free video prediction on Open X-Embodiment, run:

```bash
python inference/predict.py --pretrained_model_name_or_path "thuml/ivideogpt-oxe-64-act-free" --input_path inference/samples/fractal_sample.npz --dataset_name fractal20220817_data
```

See more examples at [`inference`](/inference).

## 🌟 Pre-training

To pre-train iVideoGPT, adjust the arguments in the command below as needed and run:

```bash
bash ./scripts/pretrain/ivideogpt-oxe-64-act-free.sh
```

See more scripts for [pre-trained models](#-models) at [`scripts/pretrain`](/scripts/pretrain).

## 🎇 Fine-tuning Video Prediction

### Finetuning Tokenizer

After preparing the [BAIR](/datasets#bair-robot-pushing) dataset, run the following:

```bash
accelerate launch train_tokenizer.py \
    --exp_name bair_tokenizer_ft --output_dir log_vqgan --seed 0 --mixed_precision bf16 \
    --model_type ctx_vqgan \
    --train_batch_size 16 --gradient_accumulation_steps 1 --disc_start 1000005 \
    --oxe_data_mixes_type bair --resolution 64 --dataloader_num_workers 16 \
    --rand_select --video_stepsize 1 --segment_horizon 16 --segment_length 8 --context_length 1 \
    --pretrained_model_name_or_path pretrained_models/ivideogpt-oxe-64-act-free/tokenizer \
    --max_train_steps 200005
```

### Finetuning Transformer

For action-conditioned video prediction, run the following:

```bash
accelerate launch train_gpt.py \
    --exp_name bair_llama_ft --output_dir log_trm --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path {log directory of finetuned tokenizer}/unwrapped_model \
    --config_name configs/llama/config.json --load_internal_llm --action_conditioned --action_dim 4 \
    --pretrained_transformer_path pretrained_models/ivideogpt-oxe-64-act-free/transformer \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type bair --resolution 64 --dataloader_num_workers 16 \
    --video_stepsize 1 --segment_length 16 --context_length 1 \
    --use_eval_dataset --use_fvd --use_frame_metrics \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
    --max_train_steps 100005
```

For action-free video prediction, remove `--load_internal_llm --action_conditioned`.

See more scripts at [`scripts/finetune`](/scripts/finetune).

### Evaluation

To evaluate the checkpoints only, run:

```bash
bash ./scripts/evaluation/bair-64-act-cond.sh
```

See more scripts for [released checkpoints](#-models) at [`scripts/evaluation`](/scripts/evaluation).

## 🤖 Visual Control

### Visual Model-based RL

Install the Metaworld version we used:

```bash
pip install git+https://github.com/Farama-Foundation/Metaworld.git@83ac03ca3207c0060112bfc101393ca794ebf1bd
```

Modify paths in `mbrl/cfgs/mbpo_config.yaml` to your own paths (currently only support absolute paths).

Run model-based RL with iVideoGPT:

```bash
python mbrl/train_metaworld_mbpo.py task=plate_slide num_train_frames=100002 demo=true
```

### Visual Planning

See [`vp`](/vp) for detailed instructions.

## 🎥 Showcases

![showcase](assets/showcase.png)

## 📜 Citation

If you find this project useful, please cite our paper as:

```
@inproceedings{wu2024ivideogpt,
    title={iVideoGPT: Interactive VideoGPTs are Scalable World Models}, 
    author={Jialong Wu and Shaofeng Yin and Ningya Feng and Xu He and Dong Li and Jianye Hao and Mingsheng Long},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024},
}
```

## 🤝 Contact

If you have any question, please contact wujialong0229@gmail.com.

## 💡 Acknowledgement

Our codebase is based on [huggingface/diffusers](https://github.com/huggingface/diffusers) and [facebookresearch/drqv2](https://github.com/facebookresearch/drqv2).
