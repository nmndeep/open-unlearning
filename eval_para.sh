export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"
# export HF_HOME=/mnt/cache/huggingface
export HF_HOME=/mnt/nsingh/huggingface-models/huggingface

# python add-exp/prevKnow_eval.py --addendum para0 --modelName TOFU_GAscent_para_0 --evall forget --data facts  
# python add-exp/prevKnow_judge.py --addendum para0 --modelName TOFU_GAscent_para_0 --evall forget --data facts  
# python add-exp/prevKnow_eval.py --addendum para1 --modelName TOFU_GAscent_para_1 --evall forget --data facts  
# python add-exp/prevKnow_judge.py --addendum para1 --modelName TOFU_GAscent_para_1 --evall forget --data facts  
# python add-exp/prevKnow_eval.py --addendum para2 --modelName TOFU_GAscent_para_2 --evall forget --data facts
# python add-exp/prevKnow_judge.py --addendum para2 --modelName TOFU_GAscent_para_2 --evall forget --data facts
# python add-exp/prevKnow_eval.py --addendum para3 --modelName TOFU_GAscent_para_3 --evall forget --data facts
# python add-exp/prevKnow_judge.py --addendum para3 --modelName TOFU_GAscent_para_3 --evall forget --data facts
# python add-exp/prevKnow_eval.py --addendum para4 --modelName TOFU_GAscent_para_4 --evall forget --data facts
# python add-exp/prevKnow_judge.py --addendum para4 --modelName TOFU_GAscent_para_4 --evall forget --data facts
# python add-exp/prevKnow_eval.py --addendum para5 --modelName TOFU_GAscent_para_5 --evall forget --data facts
# python add-exp/prevKnow_judge.py --addendum para5 --modelName TOFU_GAscent_para_5 --evall forget --data facts


# python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_qphi1 --evall forget --data facts
# python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_qphi1 --evall forget --data facts
# python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_qphi2 --evall forget --data facts
# python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_qphi2 --evall forget --data facts
# python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_qphi3 --evall forget --data facts
# python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_qphi3 --evall forget --data facts
# python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_qphi4 --evall forget --data facts
# python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_qphi4 --evall forget --data facts
# python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_qphi5 --evall forget --data facts
# python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_qphi5 --evall forget --data facts
# python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_q_org --evall forget --data facts
# python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_q_org --evall forget --data facts


# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_q_org_eval0 --evall forget --data facts
# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_q_org_eval0 --evall forget --data facts
# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_q_org_eval1 --evall forget --data facts
# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_q_org_eval1 --evall forget --data facts
# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_q_org_eval2 --evall forget --data facts
# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_q_org_eval2 --evall forget --data facts
# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_0_q_org_eval3 --evall forget --data facts
# CUDA_VISIBLE_DEVICES=4 python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_0_q_org_eval3 --evall forget --data facts


CUDA_VISIBLE_DEVICES=5 python add-exp/prevKnow_eval.py --modelName TOFU_GAscent_para_5_corrected_adjusted --evall retain --data facts
CUDA_VISIBLE_DEVICES=5 python add-exp/prevKnow_judge.py --modelName TOFU_GAscent_para_5_corrected_adjusted --evall retain --data facts


