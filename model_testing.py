import os
from tqdm import tqdm

import sys
import subprocess

import torch
import pandas as pd

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)

from clingo.control import Control
from clingo.symbol import parse_term

import pickle


class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret


def gen_answer_set(program, opt=False):
    """
        Args:
            program (str): a string of ASP program
            opt (bool): if true, only optimal answer sets are returned
                        leave it to False when there is no weak constraint
        """
    clingo_control = Control(['1', '--warn=none', '--opt-mode=optN', '-t', '4'])
    models = []
    try:
        clingo_control.add('base', [], program)
        clingo_control.ground([('base', [])], context=Context())
    except Exception as e:
        # breakpoint()
        return ["error"]
    if opt:
        clingo_control.solve(
            on_model=lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
    else:
        clingo_control.solve(on_model=lambda model: models.append(model.symbols(atoms=True)))
    models = [[str(atom) for atom in model] for model in models]
    return models


new_model = "[FINETUNED MODEL NAME]"

output_dir = "[OUTPUT DIRECTORY]"

new_model_path = output_dir + new_model

test_df = pd.read_csv("test_df.csv")
test_tuples = test_df.to_numpy()

LOAD = True

if LOAD:
    model = AutoModelForCausalLM.from_pretrained(
        new_model_path,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(new_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def generate_response(question):
    inputs_device = model.device
    inputs = tokenizer(question, return_tensors="pt").to(inputs_device)

    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def check_semantics(correct_models, generated_models):
    set_correct_models = set(map(frozenset, correct_models))
    set_generated_models = set(map(frozenset, generated_models))

    jaccard = len(set_correct_models.intersection(set_generated_models)) / len(
        set_correct_models.union(set_generated_models))
    return jaccard



def save_test_dicts(problems_syntactic_dict, problems_semantic_dict,
                    problems_syntactic_proportion_dict, problems_semantic_proportion_dict):
    syntactic_dict_fn = "syntactic_test_scores_dict.pkl"
    semantic_dict_fn = "semantic_test_scores_dict.pkl"

    syntactic_prop_dict_fn = "syntactic_prop_test_scores_dict.pkl"
    semantic_prop_dict_fn = "semantic_prop_test_scores_dict.pkl"

    with open(syntactic_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_dict, f)

    with open(semantic_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_dict, f)

    with open(syntactic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_proportion_dict, f)

    with open(semantic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_proportion_dict, f)


problems = ["assignment", "constraint", "combination", "join", "closure", "preference", "filtering",
            "negative_filtering", "numeric_filtering"]

problems_index_dict = dict(zip(range(0, len(problems)), problems))
problems_count_dict = dict.fromkeys(range(0, len(problems)), 0)
problems_syntactic_dict = dict.fromkeys(range(0, len(problems)), 0)
problems_semantic_dict = dict.fromkeys(range(0, len(problems)), 0)
problems_syntactic_proportion_dict = dict.fromkeys(range(0, len(problems)), 0)
problems_semantic_proportion_dict = dict.fromkeys(range(0, len(problems)), 0)

for i, (q, a, f) in enumerate(tqdm(test_tuples, total=len(test_tuples))):

    generated_a = generate_response(q)
    parsed_generated_a = generated_a[generated_a.index("Answer"):].replace("Answer: ", "").replace("`", "")

    index = i % len(problems)

    problems_count_dict[index] += 1

    if problems_index_dict[index] == "closure":
        parsed_generated_a = '.'.join(parsed_generated_a.split(".")[:2]) + "."
    elif problems_index_dict[index] == "preference":
        parsed_generated_a = parsed_generated_a.split("]")[0] + "]"
    else:
        parsed_generated_a = parsed_generated_a.split(".")[0] + "."

    answer_set = gen_answer_set(a + f)
    generated_answer_set = gen_answer_set(parsed_generated_a + f)

    if generated_answer_set[0] != "error":  # syntactic check did not fail
        problems_syntactic_dict[index] += 1
        problems_syntactic_proportion_dict[index] = problems_syntactic_dict[index] / problems_count_dict[index]
    else:
        print("ERROR on the following tuples")
        print(generated_a)
        print(parsed_generated_a)
        print("******")

    jaccard = check_semantics(answer_set, generated_answer_set)
    if jaccard == 1.:
        problems_semantic_dict[index] += 1
        problems_semantic_proportion_dict[index] = problems_semantic_dict[index] / problems_count_dict[index]

    if i > 0 and not i % 100:
        print(f"Test Dictionaries saved at step {i}")
        save_test_dicts(problems_syntactic_dict, problems_semantic_dict,
                        problems_syntactic_proportion_dict, problems_semantic_proportion_dict)

# In[ ]:

print("Final saving")
save_test_dicts(problems_syntactic_dict, problems_semantic_dict,
                problems_syntactic_proportion_dict, problems_semantic_proportion_dict)
