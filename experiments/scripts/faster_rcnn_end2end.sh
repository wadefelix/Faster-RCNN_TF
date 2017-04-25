#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=GPU
DEV_ID=0
# NET = VGG16 or resnet
NET=resnet
DATASET=pascal_voc

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_train"
    TEST_IMDB="voc_2007_test"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


if [ ${NET} = "VGG16" ]; then
    NETWORK=VGGnet_train
    NET_INIT="~/Faster-RCNN_TF/output/faster_rcnn_end2end/voc_6829_train/"
    #NET_INIT="~/Faster-RCNN_TF/output/faster_rcnn_end2end/voc_6829_train/VGGnet_fast_rcnn_iter_30000.ckpt"
    #NET_INIT="~/data/imagenet_models/VGG_imagenet.npy"

elif [ ${NET} = "resnet" ]; then
    #NETWORK=Resnet50_train
    NETWORK=resnet_train
    # download from https://github.com/miraclebiu/TFFRCN_resnet50
    NET_INIT=~/data/imagenet_models/Resnet50.npy
fi

TENSORBOARDLOGDIR=sameasoutput
# python ~/.local/lib/python2.7/site-packages/tensorflow/tensorboard/tensorboard.py --logdir=/path/to/logdir
# default port 6006, `--port=xxxx` to specify port

time python ./tools/train_net.py --device ${DEV} --device_id ${DEV_ID} \
  --imdb ${TRAIN_IMDB} \
  --weights ${NET_INIT} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network ${NETWORK} \
  --logdir ${TENSORBOARDLOGDIR} \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

if [ ${NET} = "VGG16" ]; then
  NETWORK=VGGnet_test
elif [ ${NET} = "resnet" ]; then
  NETWORK=resnet_test
fi

python ./tools/gapttest.py --net ${NETWORK} \
  --model ${NET_FINAL} \
  --templateid 22 \
  --tasktitle "${NETWORK} `date +'%Y-%m-%d_%H-%M-%S'`" \
  --taskdescrip "net=${NETWORK}, model=${NET_FINAL}, NET_INIT=${NET_INIT} "
