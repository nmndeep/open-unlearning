import glob
import json
import warnings

import torch

warnings.filterwarnings("ignore")


import random

import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Read all the JSON files
task = 'Books'


evalType = 'GDiff'
dirr=f'/mnt/nsingh/open-unlearning/saves/testEvals_MUSE_{task}/'
dirr += f"/Muse_{evalType}_model_ROGUE_seed_0/"
json_files = glob.glob(f"{dirr}/*/MUSE_EVAL.json")  # Replace with your actual path

# Prepare cumulative results
cumulative_min = []
averages_forget = []
averages_retain = []

for idx, file in enumerate(sorted(json_files)):
    with open(file, 'r') as f:
        data = json.load(f)
        values = data['forget_knowmem_ROUGE']['value_by_index']
        current = [values[str(i)]['rougeL_f1'] for i in range(len(values))]
        
        if idx == 0:
            cumulative_min = current
        else:
            cumulative_min = [max(prev, curr) for prev, curr in zip(cumulative_min, current)]
        
        avg = sum(cumulative_min) / len(cumulative_min)
        averages_forget.append(avg)
cumulative_min = []

for idx, file in enumerate(sorted(json_files)):
    with open(file, 'r') as f:
        data = json.load(f)
        values = data['retain_knowmem_ROUGE']['value_by_index']
        current = [values[str(i)]['rougeL_f1'] for i in range(len(values))]
        
        if idx == 0:
            cumulative_min = current
        else:
            cumulative_min = [min(prev, curr) for prev, curr in zip(cumulative_min, current)]
        
        avg = sum(cumulative_min) / len(cumulative_min)
        averages_retain.append(avg)
print(averages_forget)
print(averages_retain)
rogue_tens = {}
rogue_tens['forget_ROGUE'] = averages_forget
rogue_tens['retain_ROGUE'] = averages_retain

with open(dirr + '/worse_ROGUE.json', "w") as file:
    json.dump(rogue_tens, file)