### Train models on LKF
to train, run scripts/tofu-unlearn.sh.
- Important: the LKF data has to be moved to the tofu forget01/retain99 locations. This is done in the tofu-unlearn.sh scripts via
```
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
´´´
