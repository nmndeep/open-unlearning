import json
import os
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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

class EvalJUDGE:
    
    '''
    Use a LLm as Judge to evaluate how similar the respone of unlearning methods is
    '''

    def __init__(self, dirr, evalType='target', model='Mistral', keyType="forget_ACCURACY", sd=0, saveLogs=True):
        self.evalType = evalType
        self.dirr = dirr + f"Muse_{self.evalType}_model_JUDGE_seed_{sd}/"
        self.saveLogs = saveLogs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.keyType = keyType
        self.init_data()
        self.model_template()

    def __str__(self):
        return f"Instantiated EvalJUDGE for {self.evalType} unlearning model"
    
    def init_data(self):
        self.files = [x[1] for x in os.walk(self.dirr)][0]

    # def model_template(self):

    #     self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    #     self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    #     self.model.to(self.device)
    #     self.tokenizer.pad_token = self.tokenizer.eos_token
    #     self.template = "You are a good summarising model.  \
    #                 I will input two short sentences, you need to tell me if all the information in the first one is present in any form in the second phrase. Some rules you can use: \
    #                 1. If phrase-2 has the exact information from phrase-1 embedded somewhere, answer: Yes \
    #                 2. If you are not sure, answer: No \
    #                 3. If Phrase-2 does not have information from Phrase-1, answer: No \
    #                 Do not add any information from your side and Do not answer anything else. Here ae some examples: \
    #                 Example-1: Phrase-1: 1960, Phrase-2: august in 1960.  Output: Yes \
    #                 Example-2: Phrase-1: 93%, Phrase-2: 94%.  Output: No \
    #                 Example-3: Phrase-1: Abraham Lincoln and John Kennedy, Phrase-2: Abraham Lincoln. Output: No \
    #                 Example-4: Phrase-1: Adam's Apple, Phrase-2: Adam apple.  Output: Yes \
    #                 Example-5: Phrase-1: BMW, Audi, Phrase-2: Audi, BMW. Output: Yes \
    #                 Phrase-1:{}, Phrase-2:{} Output:"
    def model_template(self):


        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        self.model.to(self.device)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        self.generation_args = {
            "max_new_tokens": 20,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        self.template = "You are given two phrases and a question. Your task is to determine whether Phrase-2 (test answer) contains all the important information from Phrase 1 (actual answer), in any form or wording. \
                        Both phrases are the answer to Question, which you have to use to see if test-answer makes sense in regards to the question, given the true answer. Phrase 2 must not omit any key facts or details found in Phrase 1 given the question at hand. \
                        It can paraphrase or reword the content, but if any significant detail is missing, the answer should be No. \
                        Answer Yes only if all the important information from Phrase 1 is present in Phrase 2, given the question — not more, not less. Be lenient overall. Small minor un-meaningful omissions should be ignored, and the answer in that case should be Yes. \
                        Answer only with a Yes or No.\
                        ✅ Examples:  \
                        Example 1 \
                        Question: Where is the Eiffel Tower located? \
                        Phrase 1: The Eiffel Tower is located in Paris and was completed in 1889. \
                        Phrase 2: The Eiffel Tower, finished in 1889, is in Paris. \
                        Answer: Yes (Location is present in Phrase 2 and is the same as Phrase 1) \
                        Example 2 \
                        Question: What was the companies revenue change in Q2? \
                        Phrase 1: The company reported a revenue increase of 10\% in Q2. \
                        Phrase 2: The company reported a revenue increase last quarter. \
                        Answer: No \
                        (Missing the specific 10\%) \
                        Example 3 \
                        Question: What is the cat doing and where? \
                        Phrase 1: The cat is sleeping on the warm windowsill. \
                        Phrase 2: The cat is sleeping. \
                        Answer: No \
                        (Missing detail: location — “warm windowsill”) \
                        Example 4 \
                        Question: When was Marie Cure awarded the Nobel Prize in Physics? \
                        Phrase 1: Marie Curie was awarded the Nobel Prize in Physics in 1903. \
                        Phrase 2: In 1903, Marie Curie won the Nobel Prize in Physics. \
                        Answer: Yes \
                        Example 5 \
                        Question: Where was the guy born? \
                        Phrase 1: The guy was born in Tokyo, Japan. \
                        Phrase 2: Tokyo, Japan. \
                        Answer: Yes \
                        (Main information is present) \
                        Now your turn \
                        Question: {} \
                        Phrase 1: {} \
                        Phrase 2: {} \
                        Answer:"

    def compute_any_one_average(self, data, prefix="outs_"):
        relevant_keys = [k for k in data if k.startswith(prefix)]
        data = [data[key] for key in relevant_keys if key in data]
        N = len(data[0])
        coverage = [0] * N
        accuracies = []
        row_means = [sum(row) / len(row) for row in data]

        for row in data:
            for i in range(N):
                if row[i] == 1:
                    coverage[i] = 1
            accuracy = sum(coverage) / N
            accuracies.append(accuracy)

        return accuracies, row_means
        
    # def compute_any_one_average(self, data, prefix="outs_"):
    #     # Get all keys that start with the given prefix
    #     relevant_keys = [k for k in data if k.startswith(prefix)]
        
    #     # Assume all lists are the same length
    #     N = len(data[relevant_keys[0]])
        
    #     count_with_at_least_one = 0
        
    #     for i in range(N):
    #         # Check if any list has a 1 at index i
    #         if any(data[key][i] == 1 for key in relevant_keys):
    #             count_with_at_least_one += 1
        
    #     # Return average
    #     return count_with_at_least_one / N

    def check_yes_no(self, input_str):
        """
        Returns 'yes' if input is a variation of yes,
        'no' if a variation of no,
        and None if it's neither.
        """
        yes_variants = {"yes", "y", "yeah", "yep", "affirmative", "yea"}
        no_variants = {"no", "n", "nope", "nah", "negative"}

        normalized = input_str.strip().lower()
        for y in yes_variants:
            if y in normalized:
                return 1
        for n in no_variants:
            if n in normalized:
                return 0
        else:
            return 0


    def eval(self):

        acc_tens = {}
        qs = ['q_org']
        qs.extend([f'q{i}' for i in range(1,10)])
        for ii, f in enumerate(self.files): 
            acc_tens[f'files_{ii}'] = f

            with open(self.dirr + f +"/MUSE_EVAL.json", "r") as file:
                data = json.load(file)
            with open("./add-exp/MUSE_news_forget_paraphrases.json", "r") as file:
                q_data = json.load(file)
            # Extract the value_by_gt dictionary
            value_by_gt = data[self.keyType]["value_by_gt"]
            # print(len(q_data))
            # q_values = q_data['q_org']
            # Create two separate lists
            first_list = [pair[0] for pair in value_by_gt.values()]
            second_list = [pair[1] for pair in value_by_gt.values()]
            # print(len(first_list))
            # print("First list:", first_list)
            # print("Second list:", second_list)

            ans_list = []
            for ix, (gt_, answer_) in enumerate(zip(first_list, second_list)):
                
                # template = self.template.format(gt_, answer_)
                # messages = [{"role": "user", "content": f"{template}"}]
                ques = q_data[ix][qs[ii]]
                
                # encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt", pad_token_id=self.tokenizer.pad_token_id, padding=True, truncation=True)

                # model_inputs = encodeds.to(self.device)

                # generated_ids = self.model.generate(model_inputs, max_new_tokens=5, do_sample=False)
                # decoded = self.tokenizer.batch_decode(generated_ids)
                # responses = decoded[0].split("[/INST]")[-1]
                template = self.template.format(ques, gt_, answer_)
                messages = [{"role": "user", "content": f"{template}"}]

                output = self.pipe(messages, **self.generation_args)
                responses = output[0]['generated_text'][:5]
                ans = self.check_yes_no(responses)
                # print(ans)
                ans_list.append(ans)
            acc_tens[f'outs_{ii}'] = ans_list


        acc, rowacc = self.compute_any_one_average(acc_tens)
        print(f"Accuracies: {acc}")
        acc_tens['row_wise_ACC'] = rowacc
        acc_tens['ACC'] = acc
        if self.saveLogs:
            with open(self.dirr + f'/acc_tens_upto_{ii}_{self.keyType}_PHI.json', "w") as file:
                json.dump(acc_tens, file)

task='News'

for sd in [0]:
    seed_everything(sd)
    for accs in ['forget_ACCURACY']:
        ev = EvalJUDGE(dirr=f'/mnt/nsingh/open-unlearning/saves/testEvals_MUSE_{task}/', evalType='retrain', keyType=accs, sd=sd)
        # with open(f'/mnt/nsingh/open-unlearning/saves/testEvals_MUSE/Muse_retrain_model_seed_{sd}/acc_tens_upto_8.json') as f:
        #     d = json.load(f)
        #     # print(d)
        # print(ev)
        # a, b = ev.compute_any_one_average(d)
        # print(a, b)
        ev.eval()
        del ev
