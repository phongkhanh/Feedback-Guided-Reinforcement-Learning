#!/bin/bash
# Multi-GPU RL training (8B) — no feedback, no teacher:
#   All GPUs → torchrun N ranks (vanilla GRPO, no failure-guided refinement)
#
# Usage:
#   bash scripts/training/run_recogdrive_train_multi_gpu_rl_no_feedback.sh

set -e

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"

# ── SỬA CÁC DÒNG NÀY ────────────────────────────────────────────────────────
TRAIN_GPUS="4,9"                   # GPUs cho RL — thêm GPU: "9,1,2" nếu có
NPROC=2                            # số GPU trong TRAIN_GPUS
MASTER_PORT=63670
TRAIN_TEST_SPLIT=navtrain

VLM_PATH='/data2/data_fusion/work/recogdrive/internvl_chat/2B_freeze_lora_only'
CHECKPOINT="./navsim_workspace/exp/training_recogdrive_agent_2B_feedback_lora_only/2026.06.07.00.21.15/lightning_logs/version_0/checkpoints/epoch_193.ckpt"
# ────────────────────────────────────────────────────────────────────────────

echo "[launcher] Starting RL training on GPUs ${TRAIN_GPUS} (${NPROC} ranks)..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.vlm_path="${VLM_PATH}" \
    agent.cam_type='single' \
    agent.grpo=True \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.checkpoint_path="${CHECKPOINT}" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path="./navsim_workspace/exp/metric_cache_train" \
    agent.reference_policy_checkpoint="${CHECKPOINT}" \
    trainer.params.max_epochs=10 \
    trainer.params.devices=${NPROC} \
    dataloader.params.batch_size=8 \
    experiment_name=training_recogdrive_agent_no_feedback_2B \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="./navsim_workspace/exp/recogdrive_agent_cache_dir_train_feedback_2b_lora_only" \
    use_cache_without_dataset=True \
    force_cache_computation=False
    # 2>&1 | tee training_recogdrive_agent_no_feedback_2B_multigpu.txt
