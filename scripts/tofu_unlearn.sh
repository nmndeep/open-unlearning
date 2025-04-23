#!/bin/bash

# source /mnt/nsingh/miniconda3/etc/profile.d/conda.sh
# conda activate unlearning

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
# export HF_HOME=/mnt/cache/huggingface
export HF_HOME=/mnt/nsingh/huggingface-models/huggingface
models=(
    "Llama-3.2-1B-Instruct"

)
trainers_experiments=(
    "GradAscent unlearn/tofu/default.yaml"
    # "GradDiff unlearn/tofu/default.yaml"
    # "GradJSDiff unlearn/tofu/default.yaml"
)
forget_retain_splits=(
    "forget01 retain99"
    # "forget05 retain95"
    # "forget10 retain90"
)

per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################
bs_accum=(
    "4 4"
    # "8 2"
    # "16 1"
    # "2 8"
    # "1 16"
    # "4 2"
    # "8 1"
    # "2 4"
    # "1 8"
)
# bs_accum=(
#     "4 1"
#     "4 2"
#     "4 6"
#     "4 8"
#     "4 10"
#     "1 4"
#     "2 4"
#     "6 4"
#     "8 4"
#     "10 4"
# )
for params in "${bs_accum[@]}"; do
    per_device_train_batch_size=$(echo $params | cut -d' ' -f1)
    gradient_accumulation_steps=$(echo $params | cut -d' ' -f2)
    echo "Batch size: $per_device_train_batch_size, Gradient Accumulation Steps: $gradient_accumulation_steps"
    for split in "${forget_retain_splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        retain_split=$(echo $split | cut -d' ' -f2)
        for model in "${models[@]}"; do
            for trainer_experiment in "${trainers_experiments[@]}"; do
                trainer=$(echo $trainer_experiment | cut -d' ' -f1)
                experiment=$(echo $trainer_experiment | cut -d' ' -f2)
                
                task_name=tofu_${model}_${forget_split}_${trainer}_${per_device_train_batch_size}_${gradient_accumulation_steps}_testrealfacts_paraphrased_testset
                model_path=open-unlearning/tofu_${model}_full
                echo ${task_name}: Unlearning ${model_path} using ${trainer}

                # Unlearn
                CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                src/train.py --config-name=unlearn.yaml \
                experiment=${experiment} \
                trainer=${trainer} \
                task_name=${task_name} \
                model=${model} \
                forget_split=${forget_split} \
                retain_split=${retain_split} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
                trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                trainer.args.ddp_find_unused_parameters=true \
                trainer.args.gradient_checkpointing=true \

                # Eval
                # CUDA_VISIBLE_DEVICES=8 python src/eval.py \
                # experiment=eval/tofu/default.yaml \
                # forget_split=${forget_split} \
                # model=${model} \
                # task_name=${task_name} \
                # model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                # paths.output_dir=saves/unlearn/${task_name}/evals \
                # retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
            done
        done
    done
done