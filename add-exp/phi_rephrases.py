import json
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model.to(device)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# dataset = load_dataset("locuslab/TOFU", "forget01")
# PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/muse-bench___muse-news/knowmem/0.0.0/506bd5b150b92814d45e4404a82f120ab2d748bf/muse-news-retain_qa_icl_ORIG.arrow'
# PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/muse-bench___muse-books/knowmem/0.0.0/051ba90319e920d410d87cfdbd61f25843c1b892/muse-books-retain_qa_icl.arrow'

# dataset = Dataset.from_file(PARR)
# for ix, batch in enumerate(dataset):
    # test_text = batch['question']
    # a = batch['answer']
# with open('./our_assets/prevKnow_Max.json') as f:
#     d = json.load(f)
# # print(d)
# qs = [f'q{i}' for i in range(31,62)]
# ans = [f'answer{i}' for i in range(31,62)]
# newjson = []

# for qi, ai in zip(qs, ans):
#     test_text = d[qi]
#     a = d[ai]
# PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/muse-bench___muse-news/knowmem/0.0.0/506bd5b150b92814d45e4404a82f120ab2d748bf/muse-news-retain_qa_ORIG.arrow'
PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/forget01/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow'
dataset = Dataset.from_file(PARR)
newjson = []
# for ix, batch in enumerate(dataset['train']):
# for ix, batch in enumerate(dataset):
#     # print(batch)
#     test_text = batch['question_real']
#     a = batch['answer_real']
    # print("Original question: ", test_text)
    # test_text = q
with open('./newData/forget_prevKnow40_cleaned.json') as f:
    d = json.load(f)

for ix, batch in enumerate(d):
#     # print(batc
    test_text = batch['q_org']
    a = batch['answer']
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant.",
        "role": "user", "content": f"You are a good paraphraser. I will give you a quesiton sentence, I need you to paraphrase it for me. Generate 5 grammatically correct an unique paraphrases as an enumerated list. \
        make sure the output a requestions again. Make sure the meaning of parphrases remains the same as original question and that no new information is added. The output should be an enumerated list of questions. \
        Question:{test_text}"}]


    output = pipe(messages, **generation_args)
    responses = output[0]['generated_text']
    # print(responses)

    # Extract the sentences numbered 1 through 10
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|$)'

    matches = re.findall(pattern, responses, re.DOTALL)
    # Clean up any extra whitespace
    questions_all = [match.strip() for match in matches]
    entry = {"q_org": test_text} 

    # print(len(questions_all))
    for j, qq in enumerate(questions_all):
        # if j==0:
        #     print(f"Paraphrase {j+1}: {qq}")
        entry.update({f"qphi{j+1}": qq})
        # ix+=1
    entry.update({"answer": a})
    newjson.append(entry)  # Append to list
print(newjson)
with open("./newData/forget_prevKnow40_cleaned_Phi.json", "w") as file:
    json.dump(newjson, file)





