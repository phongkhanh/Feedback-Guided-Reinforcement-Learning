#!/bin/bash

set -e

SRC_ROOT="/data1/stage/navsim_workspace/dataset"
DST_ROOT="/data2/data_fusion/work/recogdrive/navsim_workspace/dataset"

NUM_TRAIN=120    # chỉnh tùy bạn
NUM_TEST=60

echo "===== CREATE MINI NAVSIM ====="

mkdir -p ${DST_ROOT}/navsim_logs/trainval
mkdir -p ${DST_ROOT}/sensor_blobs/trainval

mkdir -p ${DST_ROOT}/navsim_logs/test
mkdir -p ${DST_ROOT}/sensor_blobs/test

#################################
# FUNCTION COPY
#################################
copy_split () {
    SPLIT=$1
    NUM=$2

    echo "===== PROCESS ${SPLIT} (${NUM} scenes) ====="

    cd ${SRC_ROOT}/navsim_logs/${SPLIT}

    # random sample
    ls *.pkl | shuf | head -n ${NUM} > selected_${SPLIT}.txt

    while read scene; do
        echo "Copying ${scene}..."

        # copy log
        rsync -a ${SRC_ROOT}/navsim_logs/${SPLIT}/${scene} \
                 ${DST_ROOT}/navsim_logs/${SPLIT}/

        # remove .pkl
        scene_name=${scene%.pkl}

        # copy sensor
        if [ -d "${SRC_ROOT}/sensor_blobs/${SPLIT}/${scene_name}" ]; then
            rsync -a ${SRC_ROOT}/sensor_blobs/${SPLIT}/${scene_name} \
                     ${DST_ROOT}/sensor_blobs/${SPLIT}/
        else
            echo "WARNING: Missing sensor folder for ${scene_name}"
        fi

    done < selected_${SPLIT}.txt

    echo "DONE ${SPLIT}"
}

#################################
# RUN
#################################

copy_split trainval ${NUM_TRAIN}
# copy_split test ${NUM_TEST}

#################################
# CHECK
#################################

echo "===== CHECK ====="

echo "Trainval logs:"
ls ${DST_ROOT}/navsim_logs/trainval | wc -l

echo "Trainval sensor:"
ls ${DST_ROOT}/sensor_blobs/trainval | wc -l

echo "Test logs:"
ls ${DST_ROOT}/navsim_logs/test | wc -l

echo "Test sensor:"
ls ${DST_ROOT}/sensor_blobs/test | wc -l

echo "===== DONE ====="