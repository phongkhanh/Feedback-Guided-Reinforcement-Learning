set -x

TRAIN_TEST_SPLIT=navtrain

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"


python $NAVSIM_DEVKIT_ROOT/planning/script/run_generate_dataset_pipeline.py \
    agent=recogdrive_agent \
    experiment_name=generate_dataset \
    train_test_split=$TRAIN_TEST_SPLIT 