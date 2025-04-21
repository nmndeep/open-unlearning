
from datasets import Dataset

PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget05/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow' #tofu-train.arrow'
# PARR1 = '/mnt/nsingh/huggingface-models/huggingface/datasets/muse-bench___muse-news/raw/0.0.0/506bd5b150b92814d45e4404a82f120ab2d748bf/muse-news-forget.arrow'
dataset = Dataset.from_file(PARR)

gd_idices = [0, 1, 4, 5, 6, 8, 10, 12, 13, 17, 20, 23, 34, 36, 40, 41, 
60, 61, 62, 66, 80, 81, 83, 100, 101, 103, 120, 121, 124, 140, 141, 149, 180, 183, 184, 185]
listt = []
for i,d in enumerate(dataset):
    if i in gd_idices:
        listt.append(d)
# for i in rem:
#     listt.pop(i)
print(listt)
print(len(listt))
# print(dataset[2]) #[0])
    # cache_dir = '/home/nsingh/.cache/huggingface'

hf_dataset = Dataset.from_list(listt)
print(len(hf_dataset))
# Save to disk (optional: specify a path)
hf_dataset.save_to_disk("/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget01_cleaned")

# # --- Load JSON file ---
# with open("./add-exp/tofu_forget1_rephrases50.json", "r", encoding="utf-8") as f:
#     json_data = json.load(f)

# # Step 3: Extract q1, q2, q3
# # print(len(list(set(json_data[0].values()))))

# q1s = []
# for j in range(len(json_data)):
#     # print(len(list(set(json_data[j].values()))))
#     q1s.append(list(set(json_data[j].values()))[:10])

# # q1s = list(map(list, zip(*q1s)))
# transposed_data = list(map(list, zip(*q1s)))

# print(len(transposed_data[0]))

# # new_features_2d = [
# #     [f'newval_{i}_{j}' for j in range(40)]  # mock values
# #     for i in range(10)
# # ]

# # Add each new feature
# for i, feature_column in enumerate(transposed_data):
#     dataset = dataset.add_column(f"q_para{i+1}", feature_column)

# # Now dataset has the new 10 features added
# print(dataset) #[0])



# # columns = [[row[i] for row in q1s] for i in range(10)] 
# # for ii in range(10):
# #     dataset = dataset.add_column(f"q_para{ii+1}", columns[ii])

# # q1_list = [item.get("q1", "") for item in json_data]
# # q2_list = [item.get("q2", "") for item in json_data]


# # # Step 4: Add new columns to the dataset
# # dataset = dataset.add_column("q_pert1", q1_list)
# # dataset = dataset.add_column("q_pert2", q2_list)

# # # Step 5: Save the updated dataset back to .arrow format
# # # Use .save_to_disk if you want to keep metadata for HuggingFace
# dataset = Dataset.from_dict(dataset.to_dict())  # Convert to a normal Dataset

# dataset.save_to_disk("./add-exp/tofu-train_forget01_perturbed_para10")


# # Or write the raw Arrow table directly if you still need .arrow file
# # dataset.data.save("./add-exp/tofu-train_perturbed_ques.arrow")

# print("✅ Updated dataset saved.")
# # # --- Save updated table ---





# # from tqdm import tqdm

# # # Step 1: Load source and target datasets
# # source = Dataset.from_file(PARR)
# # target = Dataset.from_file(PARR1)

# # # Step 2: Build lookup from question → (q_pert1, q_pert2)
# # source_map = {
# #     row["question"]: (row["q_pert1"], row["q_pert2"])
# #     for row in source
# # }

# # # Step 3: Add q_pert1 and q_pert2 to target dataset
# # q_pert1_list = []
# # q_pert2_list = []
# # missing = 0

# # for row in tqdm(target, desc="Copying perturbations"):
# #     question = row["question"]
# #     if question in source_map:
# #         q1, q2 = source_map[question]
# #     else:
# #         q1 = q2 = ""  # Or use None
# #         missing += 1
# #     q_pert1_list.append(q1)
# #     q_pert2_list.append(q2)

# # print(f"⚠️ Missing matches: {missing}")

# # # Step 4: Add to target dataset
# # target = target.add_column("q_pert1", q_pert1_list)
# # target = target.add_column("q_pert2", q_pert2_list)

# # # Step 5: Save updated target
# # target.save_to_disk("./add-exp/tofu-train_perturbed_ques")

# # print("✅ Target dataset updated and saved to 'target_dataset_with_qperts'")
