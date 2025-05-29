#!/bin/bash

# source /mnt/nsingh/miniconda3/etc/profile.d/conda.sh
# conda activate unlearning

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
# export HF_HOME=/mnt/cache/huggingface
export HF_HOME=/mnt/nsingh/huggingface-models/huggingface
models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
)
forget_retain_splits=(
    "forget01 retain99"
    # "forget05 retain95"
    # "forget10 retain90"
)

########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################
bs_accum=( #per_device_train_batch_size, gradient_accumulation_steps
    # "4 24"
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


moreparams=(
    # "GradJSDiffCommaNaman unlearn/tofu/default.yaml 30 0 1e-5"
    # "GradJSDiffCommaNaman unlearn/tofu/default.yaml 30 0 2e-5"
    # "GradJSDiffCommaNaman unlearn/tofu/default.yaml 30 0 3e-5"
    # "GradJSDiffCommaNaman unlearn/tofu/default.yaml 30 0 5e-5"
    "GradDiff unlearn/tofu/default.yaml 30 0 1e-5"
    "GradDiff unlearn/tofu/default.yaml 30 0 2e-5"
    "GradDiff unlearn/tofu/default.yaml 30 0 3e-5"
    "GradDiff unlearn/tofu/default.yaml 30 0 5e-5"
    # "GradDiff unlearn/tofu/default.yaml 60 0 2e-5"
    # "NPO unlearn/tofu/default.yaml 60 0 9e-6"
    # "DPO unlearn/tofu/idk.yaml 60 0 3e-5"
    # "RMU unlearn/tofu/default.yaml 60 0 3e-5"
    # "SimNPO unlearn/tofu/default.yaml 60 0 2e-5"
    # "GradAscent unlearn/tofu/default.yaml 60 0 5e-6"


    # "SimNPO unlearn/tofu/default.yaml 10 5 2e-5"
    # "SimNPO unlearn/tofu/default.yaml 10 5 3e-5"
    # "NPO unlearn/tofu/default.yaml 10 5 9e-6"
    # "RMU unlearn/tofu/default.yaml 10 5 3e-5"

)
for paramss in "${moreparams[@]}"; do
    trainer=$(echo $paramss | cut -d' ' -f1)
    experiment=$(echo $paramss | cut -d' ' -f2)
    epoch=$(echo $paramss | cut -d' ' -f3)
    key=$(echo $paramss | cut -d' ' -f4)
    lr=$(echo $paramss | cut -d' ' -f5)
    echo "Key: $key"
    echo "Epoch: $epoch"
    echo "key: $key"
    echo "LR: $lr"
    # forget
    cp "/mnt/mmueller67/open-unlearning/new_data/LKF_forget/data-00000-of-00001.arrow" \
    "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget01/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow"
    # forget dataset_info
    cp "/mnt/mmueller67/open-unlearning/new_data/LKF_forget/dataset_info_fake.json" \
    "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget01/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/dataset_info.json"
    # retain
    cp "/mnt/mmueller67/open-unlearning/new_data/LKF_retain/data-00000-of-00001.arrow" \
    "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/retain99/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow"
    # retain datset_info
    cp "/mnt/mmueller67/open-unlearning/new_data/LKF_retain/dataset_info_fake.json" \
    "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/retain99/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/dataset_info.json"
    for params in "${bs_accum[@]}"; do
        per_device_train_batch_size=$(echo $params | cut -d' ' -f1)
        gradient_accumulation_steps=$(echo $params | cut -d' ' -f2)
        echo "Batch size: $per_device_train_batch_size, Gradient Accumulation Steps: $gradient_accumulation_steps"
        for split in "${forget_retain_splits[@]}"; do
            forget_split=$(echo $split | cut -d' ' -f1)
            retain_split=$(echo $split | cut -d' ' -f2)
            for model in "${models[@]}"; do
                
                # task_name=tofu_${model}_${forget_split}_${trainer}_${per_device_train_batch_size}_${gradient_accumulation_steps}_testrealfacts_paraphrased_key${key}
                task_name=LKF_${model}_${trainer}_${per_device_train_batch_size}_${gradient_accumulation_steps}_${epoch}eps_lr${lr}
                # model_path=open-unlearning/tofu_${model}_full
                model_path=meta-llama/${model}
                echo ${task_name}: Unlearning ${model_path} using ${trainer}

                # Unlearn
                CUDA_VISIBLE_DEVICES=5,6 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
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
                trainer.args.num_train_epochs=$epoch \
                trainer.args.learning_rate=$lr \
                trainer.args.logging_steps=1 \

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



# for lr in "${lrs[@]}"; do
#     echo "Learning Rate: $lr"
#     for epoch in "${epochs[@]}"; do
#         echo "Epoch: $epoch"
#         for key in "${keys[@]}"; do
#             echo "Key: $key"
#             # forget
#             cp "/mnt/mmueller67/open-unlearning/new_data/tofu-forget_challenger_salem_krakatoa_selected_paraphrased_${key}/data-00000-of-00001.arrow" \
#             "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget01/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow"
#             # forget dataset_info
#             cp "/mnt/mmueller67/open-unlearning/new_data/tofu-forget_challenger_salem_krakatoa_selected_paraphrased_${key}/dataset_info_fake.json" \
#             "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget01/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/dataset_info.json"
#             # retain
#             cp "/mnt/mmueller67/open-unlearning/new_data/tofu-retain_challenger_salem_krakatoa_selected_paraphrased_${key}/data-00000-of-00001.arrow" \
#             "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/retain99/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow"
#             # retain datset_info
#             cp "/mnt/mmueller67/open-unlearning/new_data/tofu-retain_challenger_salem_krakatoa_selected_paraphrased_${key}/dataset_info_fake.json" \
#             "/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/retain99/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/dataset_info.json"
#             for params in "${bs_accum[@]}"; do
#                 per_device_train_batch_size=$(echo $params | cut -d' ' -f1)
#                 gradient_accumulation_steps=$(echo $params | cut -d' ' -f2)
#                 echo "Batch size: $per_device_train_batch_size, Gradient Accumulation Steps: $gradient_accumulation_steps"
#                 for split in "${forget_retain_splits[@]}"; do
#                     forget_split=$(echo $split | cut -d' ' -f1)
#                     retain_split=$(echo $split | cut -d' ' -f2)
#                     for model in "${models[@]}"; do
#                         for trainer_experiment in "${trainers_experiments[@]}"; do
#                             trainer=$(echo $trainer_experiment | cut -d' ' -f1)
#                             experiment=$(echo $trainer_experiment | cut -d' ' -f2)
                            
#                             # task_name=tofu_${model}_${forget_split}_${trainer}_${per_device_train_batch_size}_${gradient_accumulation_steps}_testrealfacts_paraphrased_key${key}
#                             task_name=tofu_${model}_${forget_split}_${trainer}_${per_device_train_batch_size}_${gradient_accumulation_steps}_testrealfacts_paraphrased_${key}_corrected_${epoch}eps_newgamma2_lr${lr}
#                             model_path=open-unlearning/tofu_${model}_full
#                             echo ${task_name}: Unlearning ${model_path} using ${trainer}

#                             # Unlearn
#                             CUDA_VISIBLE_DEVICES=7,8 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#                             src/train.py --config-name=unlearn.yaml \
#                             experiment=${experiment} \
#                             trainer=${trainer} \
#                             task_name=${task_name} \
#                             model=${model} \
#                             forget_split=${forget_split} \
#                             retain_split=${retain_split} \
#                             model.model_args.pretrained_model_name_or_path=${model_path} \
#                             retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
#                             trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
#                             trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
#                             trainer.args.ddp_find_unused_parameters=true \
#                             trainer.args.gradient_checkpointing=true \
#                             trainer.args.num_train_epochs=$epoch \
#                             trainer.args.learning_rate=$lr \
#                             trainer.args.logging_steps=1 \

#                             # Eval
#                             # CUDA_VISIBLE_DEVICES=8 python src/eval.py \
#                             # experiment=eval/tofu/default.yaml \
#                             # forget_split=${forget_split} \
#                             # model=${model} \
#                             # task_name=${task_name} \
#                             # model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
#                             # paths.output_dir=saves/unlearn/${task_name}/evals \
#                             # retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done