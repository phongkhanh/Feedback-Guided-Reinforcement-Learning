set -x

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"

export CUDA_VISIBLE_DEVICES=0,1,2,3   # chỉnh số GPU ở đây
NUM_GPUS=4                             # phải khớp với số GPU ở trên

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MASTER_PORT=63670

echo "Using ${NUM_GPUS} GPUs: $CUDA_VISIBLE_DEVICES"

start_time=$(date +%s)

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    $NAVSIM_DEVKIT_ROOT/planning/script/vis_model_correct_baseline_wrong.py

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "=================================="
echo "Total execution time: ${elapsed} seconds"
echo "=> $(($elapsed / 60)) minutes $(($elapsed % 60)) seconds"
echo "=================================="
