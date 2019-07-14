#!/usr/bin/env bash

# docker run -it -v $PWD:/PocketFlow registry.cn-shanghai.aliyuncs.com/snowlake/tensorflow:quantize_v2 bash


./scripts/run_local.sh nets/faster_rcnn_at_pascalvoc_run.py     --learner full-prec  --data_dir_local /PocketFlow/data/voc2007_tf