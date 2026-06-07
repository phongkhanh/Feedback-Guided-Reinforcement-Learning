# TRAIN_TEST_SPLIT=navtest
# # export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
# # export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
# # export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
# export NAVSIM_DEVKIT_ROOT="navsimz"
# # export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
# CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_train

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# cache.cache_path=$CACHE_PATH

TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="/data2/data_fusion/work/recogdrive/navsim"
export OPENSCENE_DATA_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_train

python $NAVSIM_DEVKIT_ROOT/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH