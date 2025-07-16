import pickle

from huggingface_hub import login
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, BitsAndBytesConfig,
)
from peft import PeftModel

from clingo.control import Control
from clingo.symbol import parse_term

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

hugging_token = "[HUGGINGFACE TOKEN]"
login(hugging_token)

torch.cuda.is_available()

torch.cuda.device_count()

torch.manual_seed(56)

def generate_response(question, model, tokenizer):
    try:
        inputs_device = model.device
        inputs = tokenizer(question, return_tensors="pt").to(inputs_device)

        outputs = model.generate(**inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return e


class Context:
    # get features/words from a string of space separated words
    def gen_feature(self, x):
        ret = []
        for term in str(x.string).split(' '):
            ret.append(parse_term(term))
        return ret



def gen_answer_set(program, opt=False):
    clingo_control = Control(['1', '--warn=none', '--opt-mode=optN', '-t', '4'])
    models = []
    try:
        clingo_control.add('base', [], program)
        clingo_control.ground([('base', [])], context=Context())
    except Exception as e:
        return ["error"]
    if opt:
        clingo_control.solve(
            on_model=lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
    else:
        clingo_control.solve(on_model=lambda model: models.append(model.symbols(atoms=True)))
    models = [[str(atom) for atom in model] for model in models]

    return models


def check_semantics(correct_models, generated_models):
    set_correct_models = set(map(frozenset, correct_models))
    set_generated_models = set(map(frozenset, generated_models))

    jaccard = len(set_correct_models.intersection(set_generated_models)) / len(
        set_correct_models.union(set_generated_models))
    return jaccard

def save_test_dicts(problems_syntactic_dict, problems_semantic_dict, problems_syntactic_proportion_dict, problems_semantic_proportion_dict,
                    syntactic_dict_fn, semantic_dict_fn, syntactic_prop_dict_fn, semantic_prop_dict_fn):
    with open(syntactic_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_dict, f)

    with open(semantic_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_dict, f)

    with open(syntactic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_syntactic_proportion_dict, f)

    with open(semantic_prop_dict_fn, "wb") as f:
        pickle.dump(problems_semantic_proportion_dict, f)


def main():
    turn = "core"  # "core-invariance", "complex

    data_folder = "data"

    base_model = "google/gemma-2b-it"

    core_model = "gemma-2b-it-core"

    invariance_model = "gemma-2b-it-core-invariance"

    complex_model = "gemma-2b-it-complex"

    output_dir = "outputs/"

    core_model_path = output_dir + core_model

    invariance_model_path = output_dir + invariance_model

    complex_model_path = output_dir + complex_model

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    match turn:
        case "core":
            model_to_test = core_model_path

            token = hugging_token

            test_set_file_name = "test_core.csv"

            results_path = "Core/"

            if not os.path.exists(results_path):
                os.makedirs(results_path)

            syntactic_dict_fn = results_path + "core_syntactic_test_scores_dict.pkl"
            semantic_dict_fn = results_path + "core_semantic_test_scores_dict.pkl"
            syntactic_prop_dict_fn = results_path + "core_syntactic_prop_test_scores_dict.pkl"
            semantic_prop_dict_fn = results_path + "core_semantic_prop_test_scores_dict.pkl"

            parsed_file_name = results_path + "parsedCore.txt"
            errors_file_name = results_path + "errorsCore.txt"
            jaccard0_file_name = results_path + "jaccard0Core.txt"

        case "core-invariance":
            model_to_test = invariance_model_path

            token = hugging_token

            test_set_file_name = "test_core_invariance.csv"

            results_path = "Core-Invariance/"

            syntactic_dict_fn = results_path + "core_invariance_syntactic_test_scores_dict.pkl"
            semantic_dict_fn = results_path + "core_invariance_semantic_test_scores_dict.pkl"
            syntactic_prop_dict_fn = results_path + "core_invariance_syntactic_prop_test_scores_dict.pkl"
            semantic_prop_dict_fn = results_path + "core_invariance_semantic_prop_test_scores_dict.pkl"

            parsed_file_name = results_path + "parsedCorInv.txt"
            errors_file_name = results_path + "errorsCorInv.txt"
            jaccard0_file_name = results_path + "jaccard0CorInv.txt"

        case "complex":
            model_saving_path = complex_model_path
            model_to_test = model_saving_path

            token = hugging_token

            test_set_file_name = "data/test_basecomplex.csv"

            results_path = "BaseComplex/"

            syntactic_dict_fn = results_path + "complex_syntactic_test_scores_dict.pkl"
            semantic_dict_fn = results_path + "complex_semantic_test_scores_dict.pkl"
            syntactic_prop_dict_fn = results_path + "complex_syntactic_prop_test_scores_dict.pkl"
            semantic_prop_dict_fn = results_path + "complex_semantic_prop_test_scores_dict.pkl"

            parsed_file_name = results_path + "parsedComplex.txt"
            errors_file_name = results_path + "errorsComplex.txt"
            jaccard0_file_name = results_path + "jaccard0Complex.txt"

        case _:
            print("Turn not available")
            sys.exit(1)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    test_df = pd.read_csv(os.path.join(data_folder, test_set_file_name))

    print("Model to test = ", model_to_test)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        token=token
    )

    model.config.use_cache = True
    model.config.pretraining_tp = 1

    model = PeftModel.from_pretrained(model, model_to_test,
                                      is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    test_tuples = test_df.to_numpy()

    print("file_name for the test: ", test_set_file_name)
    print("n_domande ->", len(test_tuples))

    problems = ["assignment", "constraint", "combination", "join", "closure", "preference", "filtering",
                "negative_filtering", "numeric_filtering"]

    if turn == "complex":
        problems = ["join_filtering", "guessing_constraint", "combination_negative_filtering"]

    problems_index_dict = dict(zip(range(0, len(problems)), problems))
    problems_count_dict = dict.fromkeys(range(0, len(problems)), 0)
    problems_syntactic_dict = dict.fromkeys(range(0, len(problems)), 0)
    problems_semantic_dict = dict.fromkeys(range(0, len(problems)), 0)
    problems_syntactic_proportion_dict = dict.fromkeys(range(0, len(problems)), 0)
    problems_semantic_proportion_dict = dict.fromkeys(range(0, len(problems)), 0)

    with open(errors_file_name, 'w') as p:
        p.write("\n")

    with open(parsed_file_name, 'w') as r:
        r.write("\n")

    with open(jaccard0_file_name, 'w') as j:
        j.write("\n")

    for i, (q, a, f) in enumerate(tqdm(test_tuples, total=len(test_tuples))):

        generated_a = generate_response(q, model, tokenizer)

        if "Answer" in generated_a:
            parsed_generated_a = generated_a[generated_a.index("Answer"):].replace("Answer: ", "").replace("`",
                                                                                                           "")
        else:
            parsed_generated_a = generated_a

        index = i % len(problems)

        problems_count_dict[index] += 1

        unique_rules = []
        seen = set()

        for line in parsed_generated_a.splitlines():
            line = line.strip()
            if (":-" in line or ":~" in line) and line not in seen:
                unique_rules.append(line)
                seen.add(line)

        parsed_generated_a = "\n".join(unique_rules)

        if problems_index_dict[index] == "closure":
            parsed_generated_a = '\n'.join(parsed_generated_a.split("\n")[:2])
        elif problems_index_dict[index] == "preference":
            parsed_generated_a = parsed_generated_a.split("\n")[0]
        elif problems_index_dict[index] == "join_numeric_filtering":
            parsed_generated_a = "".join(parsed_generated_a.split("\n")[:1])
        elif problems_index_dict[index] == "join_filtering":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
        elif problems_index_dict[index] == "closure_guessing":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:3])
        elif problems_index_dict[index] == "guessing_constraint":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
        elif problems_index_dict[index] == "guessing_negative_filtering":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:1])
        elif problems_index_dict[index] == "guessing_numeric_filtering":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:1])
        elif problems_index_dict[index] == "guessing_filtering":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
        elif problems_index_dict[index] == "combination_negative_filtering":
            parsed_generated_a = "\n".join(parsed_generated_a.split("\n")[:2])
        else:
            parsed_generated_a = parsed_generated_a.split("\n")[0]

        with open(parsed_file_name, 'a') as r:
            r.write(str(i))
            r.write("\n")
            r.write(str(problems_index_dict[index]))
            r.write("\n\nquestion: \n")
            r.write(q)
            r.write("\n\nanswer from file: \n")
            r.write(a)
            r.write("\n\nparsed from model: \n")
            r.write(parsed_generated_a)
            r.write("\n\nfacts: \n")
            r.write(str(f))
            r.write("\n\ngenerated: \n")
            r.write(generated_a)
            r.write("\n\nunique_rules: \n")
            r.write(str(unique_rules))

        answer_set = gen_answer_set(a + f)
        generated_answer_set = gen_answer_set(parsed_generated_a + f)

        if len(generated_answer_set) == 0:
            problems_syntactic_dict[index] += 1
            problems_syntactic_proportion_dict[index] = problems_syntactic_dict[index] / problems_count_dict[
                index]
        else:
            if generated_answer_set[0] != "error":  # syntactic check did not fail
                problems_syntactic_dict[index] += 1
                problems_syntactic_proportion_dict[index] = problems_syntactic_dict[index] / \
                                                            problems_count_dict[index]
            else:
                with open(errors_file_name, 'a') as p:
                    p.write("i: ")
                    p.write(str(i))
                    p.write("\n\nindex: ")
                    p.write(str(index))
                    p.write("\n\n")
                    p.write(str(problems_index_dict[index]))
                    p.write("\n\nquestion: ")
                    p.write(q)
                    p.write("\n\nanswer from file: ")
                    p.write(a)
                    p.write("\n\nfacts: \n")
                    p.write(f)
                    p.write("\n\ngenerated_answer: ")
                    p.write(generated_a)
                    p.write("\n\nparsed answer: ")
                    p.write(parsed_generated_a)
                    p.write("\n\nanswerset from file: ")
                    p.write(str(answer_set))
                    p.write("\n\nanswerset from parsed: ")
                    p.write(str(generated_answer_set))
                    p.write("\n\n")

            jaccard = check_semantics(answer_set, generated_answer_set)
            if jaccard == 1.:
                problems_semantic_dict[index] += 1
                problems_semantic_proportion_dict[index] = problems_semantic_dict[index] / problems_count_dict[
                    index]
            else:
                with open(jaccard0_file_name, 'a') as r:
                    r.write("i: ")
                    r.write(str(i))
                    r.write("\n\nindex: ")
                    r.write(str(index))
                    r.write("\n\n")
                    r.write(str(problems_index_dict[index]))
                    r.write("\n\nquestion: ")
                    r.write(q)
                    r.write("\n\nanswer from file: ")
                    r.write(a)
                    r.write("\n\nfacts: \n")
                    r.write(f)
                    r.write("\n\ngenerated: \n")
                    r.write(generated_a)
                    # r.write("\n\nunique_rules: \n")
                    # r.write(str(unique_rules))
                    r.write("\n\nparsed: \n")
                    r.write(parsed_generated_a)
                    r.write("\n\nwanted answer_Set: ")
                    r.write(str(answer_set))
                    r.write("\n\ngenerated answer_Set: ")
                    r.write(str(generated_answer_set))
                    r.write("\n\njaccard: ")
                    r.write(str(jaccard))
                    r.write("\n\n\n")

            with open(parsed_file_name, 'a') as r:
                r.write("\n\njaccard: ")
                r.write(str(jaccard))
                r.write("\n\nAS desired:\t")
                r.write(str(answer_set))
                r.write("\nAS obtained:\t")
                r.write(str(generated_answer_set))
                r.write("\n\n\n")

    print("Final saving")
    save_test_dicts(problems_syntactic_dict, problems_semantic_dict, problems_syntactic_proportion_dict,
                    problems_semantic_proportion_dict,
                    syntactic_dict_fn, semantic_dict_fn, syntactic_prop_dict_fn, semantic_prop_dict_fn)


if __name__ == '__main__':
    main()
