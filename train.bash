#!/bin/bash

# Replace the directory name after the --image_dir

export TFHUB_CACHE_DIR=/tmp/food101/module_cache

time python retrain.py --image_dir=../../../dataset/food101/food-101/images

