set -x

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

# export PYTHONPATH=/data2/data_fusion/work/recogdrive:$PYTHONPATH
python $NAVSIM_DEVKIT_ROOT/planning/script/run_generate_dataset.py \
    agent=recogdrive_agent \
    experiment_name=generate_dataset \
    train_test_split=$TRAIN_TEST_SPLIT 