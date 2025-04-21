
import json
import re

import torch
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


model.to(device)


# dataset = load_dataset("locuslab/TOFU", "forget01")
# PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/muse-bench___muse-news/knowmem/0.0.0/506bd5b150b92814d45e4404a82f120ab2d748bf/muse-news-retain_qa_ORIG.arrow'
# PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/retain99/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow'
# PARR = '/scratch/mmueller67/open-unlearning/new_data/tofu-retain_krakatoa/data-00000-of-00001.arrow'
PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget01_cleaned/data_cleaned_40.arrow'
dataset = Dataset.from_file(PARR)
print(len(dataset))
newjson = []
# for ix, batch in enumerate(dataset['train']):
for ix, batch in enumerate(dataset):
	print(batch)
	test_text = batch['question']
	a = batch['answer']

	# print("Original question: ", test_text)
	# test_text = q
	messages = [
	    {"role": "user", "content": f"You are a good paraphraser. I will give you a quesiton sentence, I need you to paraphrase it for me. Generate 10 grammatically correct an unique paraphrases as an enumerated list. \
        Make sure the meaning of parphrases remains the same as original question and that no new information is added. The output should be an enumerated list. \
	    Question:{test_text}"}]

	encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

	model_inputs = encodeds.to(device)

	generated_ids = model.generate(model_inputs, max_new_tokens=500, do_sample=False)
	decoded = tokenizer.batch_decode(generated_ids)
	responses = decoded[0]
	# responses = (tokenizer.decode(outputs[0], skip_special_tokens=True))

	# Extract the sentences numbered 1 through 10
	pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|$)'
	matches = re.findall(pattern, responses, re.DOTALL)
	# Clean up any extra whitespace
	questions_all = [match.strip() for match in matches]
	entry = {"q_org": test_text} 
	# ix = 0
	# questions_all = []
	# for i, output in enumerate(responses):
	# 	generated_text = output["generated_text"][1:-1] if isinstance(output, dict) else str(output)
	# 	generated_text = generated_text.strip()
	# 	#don;t take the last one as it can be incomplete
	# 	questions = generated_text.split("', '")[:-1]
	# 	questions_all.append(questions)
	# # questions = ast.literal_eval(generated_text)
	# questions_all = [item for row in questions_all for item in row]

	print(len(questions_all))
	for j, qq in enumerate(questions_all):
		if j==0:
			print(f"Paraphrase {j+1}: {qq}")
		entry.update({f"q{j+1}": qq})
		# ix+=1
	entry.update({"answer": a})
	newjson.append(entry)  # Append to list

with open("./newData/forget_prevKnow40_cleaned.json", "w") as file:
    json.dump(newjson, file)





