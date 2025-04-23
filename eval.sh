#!/bin/bash

source /mnt/nsingh/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

#from HF
# CUDA_VISIBLE_DEVICES=8 python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
#   model=Llama-3.2-1B-Instruct \
#   model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
#   task_name=llama_3_1_tofu_pretrainedmodel
export HF_HOME=/mnt/nsingh/huggingface-models/huggingface

export HYDRA_FULL_ERROR=1

# task_name=tofu_Llama-3.2-1B-Instruct_forget01_GradJSDiff_NLL_0.2
# tofu_Llama-3.2-1B-Instruct_forget01_NPO
# tofu_Llama-3.2-1B-Instruct_forget01_RMU
# task_name=tofu_Llama-3.2-1B-Instruct_forget01_DPO
model=Llama-3.2-1B-Instruct
forget_split=forget01
retain_split=retain99

task_names=(
    tofu_Llama-3.2-1B-Instruct_forget01_DPO
    tofu_Llama-3.2-1B-Instruct_forget01_GradJSDiff_NLL_0.2
    tofu_Llama-3.2-1B-Instruct_forget01_NPO
    tofu_Llama-3.2-1B-Instruct_forget01_RMU
    tofu_Llama-3.2-1B-Instruct_forget01_GradAscent 
    tofu_Llama-3.2-1B-Instruct_forget01_GradDiff
)

qtypes=(
    "question"
    "paraphrased_question"
    "q_pert1"
    "q_pert2"
)

# Eval
for tasks in "${task_names[@]}"; do
  task_name=$(echo $tasks | cut -d' ' -f1)
  for split in "${qtypes[@]}"; do
      qtype=$(echo $split | cut -d' ' -f1)
      CUDA_VISIBLE_DEVICES=8 python src/eval.py \
      qtype=${qtype} \
      experiment=eval/tofu/default.yaml \
      forget_split=${forget_split} \
      model=${model} \
      task_name=${task_name} \
      model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
      paths.output_dir=saves/testEvals/${task_name}/evals_${qtype} \
      retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
  done
done