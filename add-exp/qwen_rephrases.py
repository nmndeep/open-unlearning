import json
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


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

with open('./newData/forget_prevKnow_Mistral_para.json') as f:
    d = json.load(f)

for ix, batch in enumerate(d):
#     # print(batc
    test_text = batch['q_org']
    a = batch['answer']
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant.",
        "role": "user", "content": f"You are a good paraphraser. I will give you a quesiton sentence, I need you to paraphrase it for me. Generate 5 grammatically correct an unique paraphrases as an enumerated list. \
        make sure the output are questions again. Make sure the meaning of parphrases remains the same as original question and that no new information is added. The output should be an enumerated list of questions. \
        Question:{test_text}"}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,

    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(responses)

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
        entry.update({f"qqwen{j+1}": qq})
        # ix+=1
    entry.update({"answer": a})
    newjson.append(entry)  # Append to list
    print(newjson)
    with open("./newData/forget_prevKnow_Qwen_para.json", "w") as file:
        json.dump(newjson, file)
