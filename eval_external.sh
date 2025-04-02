#!/bin/bash

source /mnt/nsingh/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

echo "Evaluating from ------ ${1}"

#local ckpt
CUDA_VISIBLE_DEVICES=8 python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=${1} \
  task_name=${2}