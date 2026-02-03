import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from iVideoGPT.mbrl.world_model_mbpo import WorldModelMBPO
from rlinf.config import validate_cfg
from rlinf.runners.embodied_mbpo_runner import EmbodiedMBPORunner
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero_spatial_mbpo_openpi_pi05",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = HybridComponentPlacement(cfg, cluster)

    wm_channel = None
    world_model = None
    if getattr(cfg, "world_model", None) is not None and cfg.world_model.get(
        "enable", False
    ):
        wm_channel = Channel.create(
            name=cfg.world_model.channel.name,
            maxsize=cfg.world_model.channel.queue_size,
        )

    actor_placement = component_placement.get_strategy("actor")
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    if wm_channel is not None:
        env_world_size = component_placement.get_world_size("env")
        stage_num = cfg.rollout.pipeline_stage_num
        num_envs_per_stage = cfg.env.train.num_envs
        wm_device = cfg.world_model.get("device", "cuda:0")
        wm_work_dir = os.path.join(
            cfg.runner.logger.log_path,
            cfg.runner.logger.experiment_name,
            "world_model",
        )
        world_model = WorldModelMBPO(
            cfg,
            wm_work_dir,
            env_world_size,
            stage_num,
            num_envs_per_stage,
            device=wm_device,
        )

    runner = EmbodiedMBPORunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        world_model=world_model,
        wm_channel=wm_channel,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
