<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2509.15965"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rlinf.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<div align="center">

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

</div>

<h1 align="center">
  <sub>RLinf: Reinforcement Learning Infrastructure for Agentic AI</sub>
</h1>

RLinf is a flexible and scalable open-source infrastructure designed for post-training foundation models via reinforcement learning. The 'inf' in RLinf stands for `Infrastructure`, highlighting its role as a robust backbone for next-generation training. It also stands for `Infinite`, symbolizing the system’s support for open-ended learning, continuous generalization, and limitless possibilities in intelligence development.

<div align="center">
  <img src="docs/source-en/_static/svg/overview.svg" alt="RLinf-overview"/>
</div>

## WM_VLA !!!!
todo: 
2. train.loss is very high
4. wm fine-tuning matches iVideoGPT paper
5. wm post-training step maybe too small comparing to warmup. Check if it's trained with new data + demo.
7. try RLinf evaluate pi05-SFT
9. check the ratio of real data and demo in wm warmup
10. check it is Flow-Noise
11. try pi_0 training
12. try longer wm horizon for wm training
13. exclude wm warmup from steps
14. use a general script with yaml


To start the WM_VLA training:
1. **Download LIBERO demos**:
   ```bash
   mkdir -p iVideoGPT/datasets/libero_raw
   wget -O iVideoGPT/datasets/libero_raw/libero_spatial.zip \
     https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip
   unzip -o iVideoGPT/datasets/libero_raw/libero_spatial.zip -d iVideoGPT/datasets/libero_raw
   ```
2. **Convert to `.npz` demos**:
   ```bash
   python iVideoGPT/datasets/convert_libero_demos.py \
       --download-dir iVideoGPT/datasets/libero_raw \
       --output-dir iVideoGPT/mbrl/demonstrations \
       --suites libero_spatial \
       --rotate-180
   ```
3. Download the pretrained VGM from:
  ```bash
    hf download thuml/ivideogpt-oxe-64-act-free --local-dir iVideoGPT/pretrained_models/ivideogpt-oxe-64-act-free
  ```
4. Download the VLA:
  ```bash
    huggingface-cli download RLinf/RLinf-Pi05-SFT \
    --local-dir pretrained_models/RLinf-Pi05-SFT \
    --local-dir-use-symlinks False
  ```
5. Download the image SIF file:
  ```bash
    huggingface-cli download HanchuZhou/wm-rl-vla-sif
  ```
6. For A100 cluster, load the docker: 
  ```bash
    bash --login -lc "
    module load singularity &&
    singularity exec --nv \
      --bind /home/darling/nvidia-egl:/opt/nvidia-egl \
      ~/wm_rl_vla.sif \
      /bin/bash
    "
  ```
  Then in the env, run:
  ```bash
    env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
    MBPO_GPU=0 VLA_GPUS=1,2,3 \
    MBPO_DEMO=true \
    MBPO_EXTRA_ARGS="num_seed_frames=0 init_update_gen_steps=200 start_mbpo=0 init_gen_times=20 gen_every_steps=1 gen_batch=24 update_gen_times=1 world_model.batch_size=4 update_gen_every_step=2 +world_model.sync_every_steps=2" \
    VLA_EXTRA_ARGS="runner.max_epochs=50 runner.val_check_interval=2 algorithm.num_group_envs=12 algorithm.rollout_epoch=4 actor.micro_batch_size=4 actor.global_batch_size=24 +rollout.sync_every_steps=1 env.eval.num_envs=6" \
    bash /mnt/workspace/hanchu/wm_vla/examples/embodiment/run_wm_rl_vla_libero_spatial_mbpo_openpi_pi05_embodied.sh
  ```

  For H200 cluster, load the docker:
  ```bash
    export SINGULARITY_TMPDIR=/mnt/workspace/hanchu/.singularity-tmp
    export SINGULARITY_CACHEDIR=/mnt/workspace/hanchu/.singularity-cache
    /usr/bin/singularity shell --userns --nv \
      --bind /mnt/workspace/hanchu/nvidia-egl:/opt/nvidia-egl \
      /mnt/workspace/hanchu/wm_rl_vla.sif
  ```
  Then in the env, run:
  ```bash
    env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 \
    MBPO_GPU=4 VLA_GPUS=5,6,7 MBPO_DEMO=true \
    MBPO_EXTRA_ARGS="num_seed_frames=5 init_update_gen_steps=2000 start_mbpo=6 init_gen_times=1 gen_every_steps=1 gen_batch=12 update_gen_times=2 world_model.batch_size=8 update_gen_every_step=10 num_eval_episodes=100 +world_model.sync_every_steps=10 replay_buffer_num_workers=4" \
    VLA_EXTRA_ARGS="runner.max_epochs=100 runner.val_check_interval=2 algorithm.rollout_epoch=1 algorithm.num_group_envs=128 actor.micro_batch_size=256 actor.global_batch_size=1536 +rollout.sync_every_steps=1 env.eval.num_envs=30" \
    bash /mnt/workspace/hanchu/wm_vla/examples/embodiment/run_wm_rl_vla_libero_spatial_mbpo_openpi_pi05_embodied.sh
  ```
