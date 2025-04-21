
import json

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

with open('./our_assets/pre_knowledge.json') as f:
    d = json.load(f)
# print(d)
qs = [f'q{i}' for i in range(1,30)]
ans = [f'answer{i}' for i in range(1,30)]

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


model.to(device)

newjson = []

# for ix, batch in enumerate(dataset['train']):
for qi, ai in zip(qs, ans):
	test_text = d[qi]
	a = d[ai]
	print("Original question: ", test_text)
	# test_text = q
	prompt =  f"You are a good model. I want you to answer this question with a short response. \
	The output should just have the answer an no preceding/succeeding text. a few examples: \
	1. Question: Where did Olympics 2012 happen? Answer: London \
	2. Question: What is the capital city of Australia? Answer: Canberra \
	3. Question: Which year did World-war 2 end? Answer: 1945 \
	    QUESTION:{test_text}, \n ANSWER:"
	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
	# Use generate with stricter control
	output = model.generate(
	    **inputs,
	    max_new_tokens=10,
	    temperature=0.0,  # Makes output deterministic and concise
	    do_sample=False,  # No randomness
	    eos_token_id=tokenizer.eos_token_id
	)


	answer = tokenizer.decode(output[0], skip_special_tokens=True).split("ANSWER:")[-1].strip()
	print("Answer:", answer)
	# Clean up any extra whitespace

	entry = {"q_org": test_text} 
