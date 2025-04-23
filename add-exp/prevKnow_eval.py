
import argparse
import json

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

KEYS = {
    'LLAMA': "meta-llama/Llama-3.2-1B-Instruct",
    'TOFU_R': "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99",
    'TOFU_RF': "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    'LLAMA_3_1_3B': "meta-llama/Llama-3.2-3B-Instruct",
    # 'TOFU_R': "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99",
    'TOFU_RF_3_1_3B': "open-unlearning/tofu_Llama-3.2-3B-Instruct_full",
    'TOFU_GAscent': "/mnt/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent_4_4_testrealfacts_selected",
    'TOFU_GDiff': "/mnt/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradDiff_4_4_testrealfacts_selected",
    'TOFU_GDiff_TOFU': "/mnt/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradDiff_4_4",
    'TOFU_clean_GAscent': "/scratch/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_GradAscent_4_4_36nonambigoustofuquestions",
    'TOFU_clean_GDiff': "/scratch/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget05_GradDiff_4_4_36nonambigoustofuquestions",
    'TOFU_GAscent_para': '/scratch/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent_4_4_testrealfacts_paraphrased',
    'TOFU_GDiff_para': '/scratch/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradDiff_4_4_testrealfacts_paraphrased',
    'TOFU_GAscent_2ep_para': '/scratch/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradAscent_4_4_testrealfacts_paraphrased2eps',
    'TOFU_GDiff_2ep_para': '/scratch/mmueller67/open-unlearning/saves/unlearn/tofu_Llama-3.2-1B-Instruct_forget01_GradDiff_4_4_testrealfacts_paraphrased2eps'
}


def eval_models(args):
    MODELL = args.modelName
    existing_entry = None
    upto = 1 if args.evall == 'retain' else 5
    
    if args.data == 'facts':
        with open(f'./newData/{args.evall}_prevKnow_all_paraphrases.json') as f:
        # with open(f'./newData/{args.evall}_prevKnow_Mistral_para.json') as f:
            d = json.load(f)
        qs = ['q_org']
        qs.extend([f'qmistral{i}' for i in range(1,6)])
        qs.extend([f'qphi{i}' for i in range(1,6)])
        qs.extend([f'qqwen{i}' for i in range(1,6)])

    else:
        if args.evall == 'retain':
            PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/retain95/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow'
        else:
            #Clean up a bit
            print("We here")
            with open(f'./newData/{args.evall}_prevKnow40_cleaned_all_paraphrases.json') as f:
        # with open(f'./newData/{args.evall}_prevKnow_Mistral_para.json') as f:
                d = json.load(f)
        qs = ['q_org']
        qs.extend([f'qmistral{i}' for i in range(1,6)])
        qs.extend([f'qphi{i}' for i in range(1,6)])
        qs.extend([f'qqwen{i}' for i in range(1,6)])
        # d = Dataset.from_file(PARR)
        # qs = ['question']
        # qs.extend([f'q_para{i}' for i in range(1,upto)])

    if args.evall == 'retain': # and args.data == 'facts':
        #specially needed for retain as samples too large
        # random.shuffle(d)
        d = d[:300]

    model = AutoModelForCausalLM.from_pretrained(KEYS[MODELL])
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    model.to(device)
    newjson = []

    for ix, batch in enumerate(d):
        # print(batch['answer'])
        if not existing_entry:
            entry = {'answer':batch['answer']}
        else:
            entry = existing_entry[ix]

        for qi in qs:
            # print(qi)
            test_text = batch[qi]
            
            # print("Original question: ", test_text)
            # test_text = q
            prompt =  f"You are a good model. I want you to answer this question with a short response. \
            The output should just be the answer without any preceding/succeeding text. Respond with a short, relevant answer only. \
            A few examples: \
            1. Question: Where did Olympics 2012 happen? Answer: London \
            2. Question: What is the capital city of Australia? Answer: The capital is Canberra \
            3. Question: Which year did World-war 2 end? Answer: 1945 \
            QUESTION:{test_text}, \n ANSWER: "
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Use generate with stricter control
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.0,  # Makes output deterministic and concise
                do_sample=False,  # No randomness
                eos_token_id=tokenizer.eos_token_id
            )

            answer = tokenizer.decode(output[0], skip_special_tokens=True).split("ANSWER:")[-1].strip()
            lines = answer.splitlines()
            cleaned_lines = []

            for line in lines:
                if line.strip() == "":
                    break  # stop at the first empty line
                cleaned_lines.append(line)
            answer = "\n".join(cleaned_lines)
            entry.update({f"ans_{qi}": answer})
        newjson.append(entry)  # Append to list
    return newjson

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EVAL_JUDGE based on answers and GT", add_help=False
    )
    parser.add_argument(
        "--modelName", default='TOFU_GDiff_TOFU', type=str, help="modelname"
    )
    parser.add_argument(
        "--evall", default='retain', type=str, help="forget/retain"
    )
    parser.add_argument(
        "--data", default='tofu', type=str, help="facts/tofu"
    )
    parser.add_argument(
        "--addendum", default='', type=str, help="add to output"
    )
    args = parser.parse_args()
    print(f"Evaluating outputs for {args.modelName} on {args.data} {args.evall} set")
    if len(args.addendum)>2:
        args.addendum = '_'+args.addendum
    outs = eval_models(args)
    with open(f"./newData/tofuEvals/{args.modelName}_answers_{args.evall}_prevKnow{args.addendum}.json", "w") as file:
        json.dump(outs, file)
    print("Done, Saved")
