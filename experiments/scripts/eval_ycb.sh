#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d YCB_Video_toolbox ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi

python3 ./tools/eval_ycb.py --dataset_root /project/1_2301/DPANet-master/YCB_Video_Dataset\
  --model /project/1_2301/MaskedFusion-master/trained_models/ycb/pose_model_48_0.02942494787275791.pth\
  --refine_model /project/1_2301/MaskedFusion-master/trained_models/ycb/pose_refine_model_1_0.5533661097288132.pth
