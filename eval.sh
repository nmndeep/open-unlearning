#!/bin/bash

source /mnt/nsingh/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

#from HF
# CUDA_VISIBLE_DEVICES=8 python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
#   model=Llama-3.2-1B-Instruct \
#   model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
#   task_name=llama_3_1_tofu_pretrainedmodel


#local ckpt
CUDA_VISIBLE_DEVICES=8 python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=/mnt/nsingh/open-unlearning/saves/unlearn/llama_3_2_1b_tofu_GradJSDIFF_forg01_gamma5 \
  task_name=llama_3_2_1b_tofu_GradJSDIFF_forg01_gamma5