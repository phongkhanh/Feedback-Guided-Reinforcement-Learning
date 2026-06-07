from typing import Tuple
from pathlib import Path
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule, AgentLightningDiT
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

import json

# type_map = {}
# with open("dataset_fail_2B_167.jsonl") as f:
#     for line in f:
#         d = json.loads(line)
#         type_map[d["token"]] = d["type"]

# def custom_collate_fn(batch):
#     features_list, targets_list, tokens_list = zip(*batch)

#     # ===== FILTER AMBIGUOUS =====
#     filtered = [
#         (f, t, token)
#         for f, t, token in zip(features_list, targets_list, tokens_list)
#         if type_map.get(token) == "ambiguous"
#     ]

#     # 🔥 nếu không có ambiguous → fallback về batch gốc
#     if len(filtered) == 0:
#         filtered = list(zip(features_list, targets_list, tokens_list))

#     # dùng filtered
#     features_list, targets_list, tokens_list = zip(*filtered)

#     history_trajectory = torch.stack([f['history_trajectory'] for f in features_list], dim=0).cpu()
#     high_command_one_hot = torch.stack([f['high_command_one_hot'] for f in features_list], dim=0).cpu()
#     status_feature = torch.stack([f['status_feature'] for f in features_list], dim=0).cpu()

#     last_hidden_state = rnn_utils.pad_sequence(
#         [f['last_hidden_state'] for f in features_list],
#         batch_first=True,
#         padding_value=0.0
#     ).clone().detach()

#     trajectory = torch.stack([t['trajectory'] for t in targets_list], dim=0).cpu()

#     features = {
#         'history_trajectory': history_trajectory,
#         'high_command_one_hot': high_command_one_hot,
#         'status_feature': status_feature,
#         'last_hidden_state': last_hidden_state,
#     }

#     targets = {
#         'trajectory': trajectory
#     }

#     return features, targets, tokens_list
def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    last_hidden_state = rnn_utils.pad_sequence(
        [features['last_hidden_state'] for features in features_list],
        batch_first=True,
        padding_value=0.0
    ).clone().detach()

    image_path_tensor = rnn_utils.pad_sequence(
        [features['image_path_tensor'] for features in features_list],
        batch_first=True,
        padding_value=0
    ).clone().detach()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()


    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
        'last_hidden_state': last_hidden_state,
        'image_path_tensor': image_path_tensor,
    }
    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list



def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningDiT(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    _repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
    _fail_jsonl = os.path.join(_repo_root, "scripts", "dataset_fail.jsonl")
    with open(_fail_jsonl) as _f:
        _fail_tokens = {json.loads(_line)["token"] for _line in _f}
    _before = len(train_data.tokens)
    train_data.tokens = [t for t in train_data.tokens if t in _fail_tokens]
    logger.info("Dataset filter: %d → %d tokens (from %s)", _before, len(train_data.tokens), _fail_jsonl)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss_epoch",
            mode='min',
            save_top_k=5,
            every_n_epochs=1,          # save after each epoch (end of validation)
            save_last=True,            # always keep last.ckpt
        ),
        pl.callbacks.ModelCheckpoint(
            filename="step-{step}",
            every_n_train_steps=200,   # save every 200 steps mid-epoch
            save_top_k=-1,             # keep all (no monitor → can't rank)
            save_last=False,
        ),
    ])

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
