set -x

# ========================
# GPU CONFIG
# ========================
export CUDA_VISIBLE_DEVICES=1,3,5,6
GPUS=4

# ========================
# NAVSIM ENV
# ========================
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"

TRAIN_TEST_SPLIT=navtrain

# ========================
# PORT (quan trọng)
# ========================
export MASTER_PORT=63669

# ========================
# TRAIN (DDP via torchrun)
# ========================
torchrun \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/planning/script/run_training_recogdrive.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path='../navsim_workspace/checkpoint/ReCogDrive_VLM_2B' \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=200 \
    trainer.params.devices=4 \
    trainer.params.strategy=ddp \
    +trainer.params.log_every_n_steps=1 \
    +trainer.params.enable_progress_bar=True \
    experiment_name=training_recogdrive_agent \
    train_test_split=navtrain \
    cache_path="../navsim_workspace/exp/recogdrive_agent_cache_dir_train" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    2>&1 | tee train_4gpu_log.txt