#!/bin/bash

source /mnt/nsingh/miniconda3/etc/profile.d/conda.sh
conda activate unlearning


#Get modelName keys from the dict on top of prevKnow_eval.py
#Keep 'data' to facts for our facual 15 paraphrases, set to 'tofu' for tofu cleand 15 paraphrases.
#The outputs are set: acc_tens gives you the cumulative list. 
#Add location of new model paths to the Dict and then evaluate them 

CUDA_VISIBLE_DEVICES=4 python3 add-exp/prevKnow_eval.py --evall forget --modelName TOFU_GDiff_2ep_para --data facts --addendum all2PARA
CUDA_VISIBLE_DEVICES=3 python3 add-exp/prevKnow_judge.py --evall forget --modelName TOFU_GDiff_2ep_para --data facts --addendum all2PARA
