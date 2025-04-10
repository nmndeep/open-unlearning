#!/bin/bash


HOST=$(hostname)

if [[ "$HOST" == "venusaur" ]]; then
    MY_PATH="/mnt/nsingh"
elif [[ "$HOST" == "mlcbm006" ]]; then
    MY_PATH="/weka/hein/ndsingh40"
else
    MY_PATH="/default/path"
fi

echo "Running on $HOST, using path: $MY_PATH"

source $MY_PATH/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=$MY_PATH/huggingface-models/huggingface
echo $HF_HOME

export HYDRA_FULL_ERROR=1

modelname=Llama-2-7b-hf


data_split="News"
trainer=pretrained

# trainers=(
#     "GradDiff"
#     "GradJSDiff"
#     "NPO"
#     # "SimNPO"
# )

task_name=Muse_target_model

qtypes=(
    "question"
    "q_para1"
    "q_para2"
    "q_para3"
    "q_para4"
    "q_para5"
    "q_para6"
    "q_para7"
    "q_para8"
    # "q_para9"
    # "q_para10"
)

# Eval

for split in "${qtypes[@]}"; do
  qtype=$(echo $split | cut -d' ' -f1)
    CUDA_VISIBLE_DEVICES=1 python src/eval.py \
    experiment=eval/muse/default.yaml \
    task_name=${task_name} \
    data_split=${data_split} \
    qtype=${qtype} \
    model=${modelname} \
    model.model_args.pretrained_model_name_or_path=muse-bench/MUSE-${data_split}_target \
    paths.output_dir=saves/testEvals_MUSE/${task_name}/evals_${qtype}_Rpert \
    retain_logs_path=saves/eval/muse_${modelname}_${data_split}_retrain/MUSE_EVAL.json

done
