#!/bin/bash
# Multi-GPU RL training (8B):
#   GPU 0  → teacher_server.py (InternVL3-38B, standalone)
#   GPU 9,... → torchrun N ranks (RL training)
#
# Usage:
#   bash scripts/training/run_recogdrive_train_multi_gpu_rl.sh
#
# Stop: Ctrl-C kills all children via trap below.

set -e

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"

# ── SỬA CÁC DÒNG NÀY ────────────────────────────────────────────────────────
TEACHER_GPU=0                    # GPU chạy teacher (InternVL3-38B)
TRAIN_GPUS="1,2,3"                   # GPUs cho RL — thêm GPU: "9,1,2" nếu có
NPROC=3 # số GPU trong TRAIN_GPUS
TEACHER_PORT=55123
MASTER_PORT=63671
TRAIN_TEST_SPLIT=navtrain

TEACHER_MODEL_PATH="./pretrained/InternVL3-38B"
VLM_PATH='/data2/data_fusion/work/recogdrive/internvl_chat/2B_freeze_lora_only'
FEEDBACK_LORA_PATH='/data2/data_fusion/work/recogdrive/internvl_chat/2B_freeze_lora_only'
CHECKPOINT="./navsim_workspace/exp/training_recogdrive_agent_2B_feedback_lora_only/2026.06.07.00.21.15/lightning_logs/version_0/checkpoints/epoch_193.ckpt"
# ────────────────────────────────────────────────────────────────────────────

_CLEANUP_DONE=0
cleanup() {
    [ $_CLEANUP_DONE -eq 1 ] && return
    _CLEANUP_DONE=1
    echo "[launcher] Shutting down..."
    [ -n "$TEACHER_PID" ] && kill "$TEACHER_PID" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# 1) Start teacher server on dedicated GPU
echo "[launcher] Starting teacher_server on GPU ${TEACHER_GPU}, port ${TEACHER_PORT} ..."
CUDA_VISIBLE_DEVICES=${TEACHER_GPU} python teacher_server.py \
    --model_path "${TEACHER_MODEL_PATH}" \
    --port ${TEACHER_PORT} &
TEACHER_PID=$!

# Wait until teacher server is ready
echo "[launcher] Waiting for teacher server to be ready..."
for i in $(seq 1 120); do
    if ! kill -0 $TEACHER_PID 2>/dev/null; then
        echo "[launcher] ERROR: teacher_server exited early"
        exit 1
    fi
    if nc -z localhost ${TEACHER_PORT} 2>/dev/null; then
        echo "[launcher] Teacher server is ready."
        break
    fi
    sleep 5
    if [ $i -eq 120 ]; then
        echo "[launcher] ERROR: timeout waiting for teacher server"
        exit 1
    fi
done

# 2) Start RL training
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
    agent.teacher_server_port=${TEACHER_PORT} \
    agent.feedback_lora_path="${FEEDBACK_LORA_PATH}" \
    agent.failure_threshold=0.5 \
    agent.num_refined_samples=4 \
    trainer.params.max_epochs=10 \
    trainer.params.devices=${NPROC} \
    dataloader.params.batch_size=8 \
    experiment_name=training_recogdrive_agent_feedback_2B_filter_data_08 \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="./navsim_workspace/exp/recogdrive_agent_cache_dir_train_feedback_2b_lora_only" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    2>&1 | tee training_recogdrive_agent_feedback_2B_filter_data.txt

wait
