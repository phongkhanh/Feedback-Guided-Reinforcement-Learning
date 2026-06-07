export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"

TRAIN_TEST_SPLIT=navtrain

# 🔥 GPU
export CUDA_VISIBLE_DEVICES=9,0    # cuda:0=GPU1(training+feedback-LoRA), cuda:1=GPU0(teacher)
GPUS=1

# 🔥 PORT (fix cứng cho local)
MASTER_PORT=63669


CHECKPOINT="../navsim_workspace/exp/training_recogdrive_agent_feedback_lora_only/2026.06.03.15.15.41/lightning_logs/version_0/checkpoints/epoch_189.ckpt"



torchrun \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.vlm_path='/data2/data_fusion/work/recogdrive/internvl_chat/8B_freeze_lora_only' \
    agent.cam_type='single' \
    agent.grpo=True \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.checkpoint_path="$CHECKPOINT" \
    agent.dit_type="small" \
    agent.vlm_size="large" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path="../navsim_workspace/exp/metric_cache_train" \
    agent.reference_policy_checkpoint="$CHECKPOINT" \
    agent.teacher_model_path="../pretrained/InternVL3-38B" \
    agent.feedback_lora_path='/data2/data_fusion/work/recogdrive/internvl_chat/8B_freeze_lora_only' \
    agent.failure_threshold=0.5 \
    agent.num_refined_samples=4 \
    trainer.params.max_epochs=10 \
    trainer.params.devices=1 \
    dataloader.params.batch_size=8 \
    experiment_name=training_recogdrive_agent_feedback_8B \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="../navsim_workspace/exp/recogdrive_agent_cache_dir_train_feedback_8b_lora_only" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    2>&1 | tee training_recogdrive_agent_feedback_8B.txt