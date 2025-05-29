import argparse
import json
import warnings

import torch
from datasets import Dataset
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

    def __init__(self, dirr, evalType='target', model='Mistral', modelName='LLAMA', keyType="forget_ACCURACY", evall='forget', data='facts', sd=0, saveLogs=True):
        self.evalType = evalType
        self.dirr = dirr
        self.saveLogs = saveLogs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.keyType = keyType
        self.modelName = modelName
        self.data = data
        self.evall = evall
        self.init_data()
        self.model_template()

    def __str__(self):
        return f"Instantiated EvalJUDGE for {self.evalType} unlearning model"
    
    def init_data(self):
        self.file = self.dirr

    def model_template_old(self):

        self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.template = "You are given two phrases and a question. Your task is to determine whether Phrase-2 (test answer) contains all the important information from Phrase 1 (actual answer), in any form or wording. \
                        Both phrases are the answer to Question, which you have to use to see if test-answer makes sense in regards to the question, given the true answer. Phrase 2 must not omit any key facts or details found in Phrase 1 given the question at hand. \
                        It can paraphrase or reword the content, but if any significant detail is missing, the answer should be No. \
                        Answer Yes if and only if most of the important information from Phrase 1 is present in Phrase 2, given the question. \
                        Be lenient overall. Small minor un-meaningful omissions should be ignored, and the answer in that case should be Yes. \
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
                        Phrase 1: Tokyo, Japan \
                        Phrase 2: Tokyo \
                        Answer: Yes \
                        (Non ambiguous answer, with most imporant details) \
                        Now your turn \
                        Question: {} \
                        Phrase 1: {} \
                        Phrase 2: {} \
                        Answer: "

    def model_template(self):

        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-4", 
            torch_dtype="auto", 
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")
        self.model.to(self.device)
        # Phi-3.5-mini-instruct
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        self.generation_args = {
            "max_new_tokens": 12, #keep it small so that no gibberish is outputed
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        self.template = "You are given a question, along with two phrases: \
                        - Phrase 1: the ground truth or correct answer \
                        - Phrase 2: the test answer to evaluate \
                        Your task is to determine whether Phrase 2 includes all of the relevant information from Phrase 1, as required to correctly answer the given question. \
                        Use the question to judge which parts of Phrase 1 are essential. If a piece of information in Phrase 1 is important for fully or correctly answering the question, then it must also appear — in some form — in Phrase 2. \
                        Phrase 2 can: \
                        \
                        - Paraphrase or reword the content \
                        - Reorder details \
                        - Use synonyms or simpler language \
                        \
                        These are all acceptable as long as the key facts and meaning are preserved. \
                        If any significant or necessary detail is missing from Phrase 2 — based on what’s needed to answer the question — respond No. \
                        If all essential information from Phrase 1 is present in Phrase 2 (even if reworded), and only minor, non-impactful details are omitted, respond Yes. \
                        Be lenient in general, especially when omissions do not change the accuracy or completeness of the answer. \
                        Respond with a single word: Yes or No. \
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
                        Example 5 \
                        Question: When was Marie Cure awarded the Nobel Prize in Physics? \
                        Phrase 1: Marie Curie was awarded the Nobel Prize in Physics in 1903. \
                        Phrase 2: In 1904 \
                        Answer: No \
                        Example 6 \
                        Question: What is the capital of Argentina? \
                        Phrase 1: The capital of South American country Argentina is Buenos Aires. \
                        Phrase 2: Buenos Aires \
                        Answer: Yes \
                        Example 7 \
                        Question: What is the name of the author? \
                        Phrase 1: The author in question is James Underwood, an esteemed literture writer who hails from Cottswold, England. \
                        Phrase 2: The celebrated author is named James Underwood \
                        Answer: Yes \
                        (Non ambiguous answer, with relevant answer: name of author) \
                        Now your turn \
                        Question: {} \
                        Phrase 1: {} \
                        Phrase 2: {} \
                        Answer: "

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

    def average_case_Acc(self, data, prefix="outs_"):
        relevant_keys = [k for k in data if k.startswith(prefix)]
        data = [data[key] for key in relevant_keys if key in data]
        N = len(data[0])
        # print(N)
        coverage = [0] * N
        accuracies = []

        if isinstance(data[0], list):

            arr = np.array(data)

            # Step 1: Mean across the 3 rows (axis=0), result is shape (100,)
            mean_across = np.mean(arr, axis=0)

            # Step 2: Mean across the 100 values (axis=0, since it's 1D now)
            final_mean = np.mean(mean_across)
        else:
            arr = np.array(data)
            final_mean = np.mean(arr, axis=0)

        return final_mean, None

    def check_yes_no(self, input_str, inside=0):
        """
        Returns 'yes' if input is a variation of yes,
        'no' if a variation of no,
        and None if it's neither.
        """
        yes_variants = {"yes", "yeah"}
        no_variants = {"no","nope"}

        if inside:
            return 1
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

        with open(self.file, "r") as file:
            data = json.load(file)
        upto = 1 if self.evall == 'retain' else 5

        if self.data == 'facts':
            if self.evall == 'forget':
                with open(f'/mnt/nsingh/open-unlearning/newData/{self.evall}_prevKnow_all_paraphrases.json') as f:
                    q_data = json.load(f)
                qs = ['q_org']
                #new questions having 5 para-phrases each
                qs.extend([f'qmistral{i}' for i in range(1,6)])
                qs.extend([f'qphi{i}' for i in range(1,6)])
                qs.extend([f'qqwen{i}' for i in range(1,6)])
            elif args.evall=='retain':
                with open(f'/mnt/nsingh/open-unlearning/newData/{args.evall}_prevKnow_Mistral_para.json') as f:
                    q_data = json.load(f)
                qs = ['q_org']
                qs.extend([f'q{i}' for i in range(1,11)])
        else:
            if self.evall == 'retain':
                PARR = '/mnt/nsingh/huggingface-models/huggingface/datasets/locuslab___tofu/retain95/0.0.0/324592d84ae4f482ac7249b9285c2ecdb53e3a68/tofu-train.arrow'
                q_data = Dataset.from_file(PARR)  
                # qs.extend([f'q_para{i}' for i in range(1, upto)])
                qs = ['question']
            else:
                
                print("we here")
                with open(f'/mnt/nsingh/open-unlearning/newData/{self.evall}_prevKnow40_cleaned_all_paraphrases.json') as f:
                    q_data = json.load(f)
                qs = ['q_org']
                qs.extend([f'qmistral{i}' for i in range(1,6)])
                qs.extend([f'qphi{i}' for i in range(1,6)])
                qs.extend([f'qqwen{i}' for i in range(1,6)])

        for qi in qs:
            # print(data)
            # Extract the value_by_gt dictionary
            value_by_gt  = [item['answer'] for item in data]
            value_by_ans = [item[f'ans_{qi}'] for item in data]
            qs = [item[f'{qi}'] for item in q_data]

            # Create two separate lists
            first_list = value_by_gt
            second_list = value_by_ans

            ans_list = []
            for ix, (gt_, answer_) in enumerate(zip(first_list, second_list)):
                
                inside = 0
                responses=None
                if (gt_.strip().lower() in answer_.strip().lower().rstrip(",.") or gt_.strip().lower().rstrip(",.")==answer_.strip().lower().rstrip(",.")):
                    inside = 1
                
                if not inside:

                    template = self.template.format(qs[ix], gt_, answer_)
                    messages = [{"role": "user", "content": f"{template}"}]


                    output = self.pipe(messages, **self.generation_args)
                    responses = output[0]['generated_text']
                ans = self.check_yes_no(responses, inside)

                ans_list.append(ans)
            acc_tens[f'outs_{qi}'] = ans_list

        if self.evall != 'retain':
            acc, rowacc = self.compute_any_one_average(acc_tens)
        else:
            acc, rowacc = self.average_case_Acc(acc_tens)
            rowacc = [0,0,0]
        print(f"{self.evall} {self.modelName} Accuracies: {acc}")
        acc_tens['row_wise_ACC'] = rowacc
        acc_tens['ACC'] = acc
        if self.saveLogs:
            with open("./newData/tofuEvals/" +"acc_tens_" + self.dirr.split("/")[-1], "w") as file:
                json.dump(acc_tens, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EVAL_JUDGE based on answers and GT", add_help=False
    )

    parser.add_argument(
        "--modelName", default='TOFU_GDiff', type=str, help="l2/dissim"
    )
    parser.add_argument(
        "--evall", default='retain', type=str, help="forget/retain"
    )
    parser.add_argument(
        "--data", default='facts', type=str, help="facts/tofu"
    )
    parser.add_argument(
        "--addendum", default='', type=str, help="add to output"
    )
    args = parser.parse_args()

    if len(args.addendum)>2:
        args.addendum = '_'+args.addendum

    ev = EvalJUDGE(dirr=f"./newData/tofuEvals/{args.modelName}_answers_{args.evall}_prevKnow{args.addendum}.json", modelName=args.modelName, evall=args.evall, data=args.data)
    ev.eval()
    del ev
