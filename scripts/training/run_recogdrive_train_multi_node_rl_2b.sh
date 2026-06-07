export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"

TRAIN_TEST_SPLIT=navtrain

# 🔥 GPU
export CUDA_VISIBLE_DEVICES=1
GPUS=1

# 🔥 PORT (fix cứng cho local)
MASTER_PORT=63669

CHECKPOINT="../navsim_workspace/exp/training_recogdrive_agent/2026.04.21.23.40.08/lightning_logs/version_0/checkpoints/epoch_167.ckpt"

torchrun \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.vlm_path='../navsim_workspace/checkpoint/ReCogDrive_VLM_2B' \
    agent.cam_type='single' \
    agent.grpo=True \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.checkpoint_path="$CHECKPOINT" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path="../navsim_workspace/exp/metric_cache_train" \
    agent.reference_policy_checkpoint="$CHECKPOINT" \
    trainer.params.max_epochs=1 \
    trainer.params.devices=1 \
    dataloader.params.batch_size=8 \
    experiment_name=training_recogdrive_agent \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="../navsim_workspace/exp/recogdrive_agent_cache_dir_train" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    # 2>&1 | tee train_recogdrive_rl_4gpu.txt