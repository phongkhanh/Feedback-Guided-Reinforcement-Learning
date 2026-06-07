set -x

TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"
# 🔥 SET GPU (QUAN TRỌNG NHẤT)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 🔥 tránh NCCL lỗi (optional nhưng nên có)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}

echo "Using GPU: $CUDA_VISIBLE_DEVICES"

export CUDA_LAUNCH_BLOCKING=1

# CHECKPOINT="/data2/data_fusion/work/recogdrive/navsim_workspace/checkpoint/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"
# CHECKPOINT="../navsim_workspace/exp/training_recogdrive_agent/2026.04.21.23.40.08/lightning_logs/version_0/checkpoints/epoch_190.ckpt"
CHECKPOINT="../navsim_workspace/exp/training_recogdrive_agent/2026.04.21.23.40.08/lightning_logs/version_0/checkpoints/epoch_167.ckpt"



start_time=$(date +%s)

torchrun \
    --nproc_per_node=4 \
    $NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path=$CHECKPOINT \
    agent.vlm_path="/data2/data_fusion/work/recogdrive/navsim_workspace/checkpoint/ReCogDrive_VLM_2B" \
    agent.cam_type=single \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type=internvl \
    agent.dit_type=small \
    agent.vlm_size=small \
    agent.sampling_method=ddim \
    experiment_name=recogdrive_agent_eval

end_time=$(date +%s)

elapsed=$((end_time - start_time))

echo "=================================="
echo "Total execution time: ${elapsed} seconds"
echo "=> $(($elapsed / 60)) minutes $(($elapsed % 60)) seconds"
echo "=================================="