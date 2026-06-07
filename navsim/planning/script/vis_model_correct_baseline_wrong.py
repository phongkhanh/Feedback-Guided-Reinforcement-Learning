import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import pandas as pd
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# ============================================================
# CONFIG — chỉnh sửa các path này theo môi trường của bạn
# ============================================================
CSV_PATH = "/data2/data_fusion/work/recogdrive/navsim_workspace/exp/recogdrive_agent_eval_feedback_sup/model_correct_baseline_wrong.csv"
CHECKPOINT_PATH = "/data2/data_fusion/work/recogdrive/navsim_workspace/exp/training_recogdrive_agent_no_feedback_2B/2026.06.07.09.46.19/lightning_logs/version_0/checkpoints/epoch_4.ckpt"
VLM_PATH = "/data2/data_fusion/work/recogdrive/internvl_chat/2B_freeze_lora_only"
METRIC_CACHE_PATH = "/data2/data_fusion/work/recogdrive/navsim_workspace/exp/metric_cache"
OUTPUT_DIR = "vis_model_correct_baseline_wrong"

SPLIT = "test"
FILTER = "navtest"
# ============================================================


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    return 0, 1, 0


rank, world_size, local_rank = init_distributed()
device = torch.device("cuda", local_rank)

hydra.initialize(config_path="./config/common/train_test_split/scene_filter")
cfg = hydra.compose(config_name=FILTER)
scene_filter: SceneFilter = instantiate(cfg)

openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

agent = ReCogDriveAgent(
    TrajectorySampling(time_horizon=4, interval_length=0.5),
    checkpoint_path=CHECKPOINT_PATH,
    vlm_path=VLM_PATH,
    cam_type="single",
    vlm_type="internvl",
    dit_type="small",
    sampling_method="ddim",
    cache_mode=False,
    cache_hidden_state=False,
    vlm_size="small",
    grpo=False,
).to(device)

agent.initialize()

sensor_config = agent.module.get_sensor_config() if hasattr(agent, "module") else agent.get_sensor_config()

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=sensor_config,
)
scene_loader_traj = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=sensor_config,
    load_image_path=True,
)

def load_tokens_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    tokens = df["token"].tolist()
    print(f"Loaded {len(tokens)} tokens from {csv_path}")
    return tokens, df


def visualize_tokens(tokens, df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Chia token theo rank nếu chạy distributed
    local_tokens = tokens[rank::world_size]

    token_info = df.set_index("token")

    for token in tqdm(local_tokens, desc=f"Rank {rank} visualizing"):
        if token not in scene_loader.tokens:
            print(f"[SKIP] token {token} not in scene_loader")
            continue

        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        frame_idx = scene.scene_metadata.num_history_frames - 1

        fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)

        # Tên file mô tả ngắn gọn baseline fail metric nào
        if token in token_info.index:
            row = token_info.loc[token]
            failed = [c.replace("baseline_", "") for c in row.index if c.startswith("baseline_") and c != "baseline_score" and row[c] < 1.0]
            fail_tag = "_".join(failed) if failed else "unknown"
            model_score = f"{row['score']:.4f}" if "score" in row else "na"
            baseline_score = f"{row['baseline_score']:.4f}" if "baseline_score" in row else "na"
            filename = f"{token}__model{model_score}__base{baseline_score}__{fail_tag}.png"
        else:
            filename = f"{token}.png"

        plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight", dpi=200)
        plt.close(fig)

    print(f"Rank {rank}: saved {len(local_tokens)} images to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    tokens, df = load_tokens_from_csv(CSV_PATH)
    visualize_tokens(tokens, df)
