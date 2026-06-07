TRAIN_TEST_SPLIT=navtrain
# export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
# export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
# export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
# export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
# export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_debug
echo "CACHE_PATH: ${CACHE_PATH}"
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
# export NCCL_SHM_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_VISIBLE_DEVICES=1

start_time=$(date +%s)
torchrun \
    --nproc_per_node=1 \
    $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache_debug \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="../navsim_workspace/checkpoint/ReCogDrive_VLM_2B" \
    # cache_path=$CACHE_PATH  > caching_dataset_phongbk_debug.txt 2>&1
    cache_path=$CACHE_PATH

end_time=$(date +%s)

elapsed=$((end_time - start_time))

echo "=================================="
echo "Total execution time: ${elapsed} seconds"
echo "=> $(($elapsed / 60)) minutes $(($elapsed % 60)) seconds"
echo "=================================="