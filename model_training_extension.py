from huggingface_hub import login
import os
import sys
import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

hugging_token = "[HUGGINGFACE TOKEN]"
login(hugging_token)

torch.cuda.is_available()

torch.cuda.device_count()

torch.manual_seed(56)



def main():

    turn = "core"  # "core-invariance", "complex"

    data_folder = "data"

    base_model = "google/gemma-2b-it"

    core_model = "gemma-2b-it-core"

    invariance_model = "gemma-2b-it-core-invariance"

    complex_model = "gemma-2b-it-complex"

    output_dir = "outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
            model_to_train = base_model
            model_saving_path = core_model_path

            token = hugging_token

            train_file_name = "train_core.csv"
            val_file_name = "val_core.csv"

            target_modules = "all-linear"

            results_path = "Core/"

        case "core-invariance":
            model_to_train = core_model_path
            model_saving_path = invariance_model_path

            token = hugging_token

            train_file_name = "train_invariance.csv"
            val_file_name = "val_invariance.csv"

            target_modules = None

            results_path = "Core-Invariance/"

        case "complex":
            model_to_train = base_model
            model_saving_path = complex_model_path

            token = hugging_token

            train_file_name = "data/train_basecomplex.csv"
            val_file_name = "data/val_basecomplex.csv"

            target_modules = "all-linear"

            results_path = "BaseComplex/"

        case _:
            print("Turn not available")
            sys.exit(1)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train_dataset = pd.read_csv(os.path.join(data_folder, train_file_name))
    val_dataset = pd.read_csv(os.path.join(data_folder, val_file_name))

    def formatting_func(example):
        text = f"Question: {example['question'][0]}\nAnswer: {example['answer'][0]}"
        return [text]

    torch.cuda.empty_cache()

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    print("model to train = ", model_to_train)

    print("Training set lenght", train_dataset.num_rows)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        token=token
    )

    model.config.use_cache = True
    model.config.pretraining_tp = 1

    if turn == "core-invariance":
        model = PeftModel.from_pretrained(model, model_to_train, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        max_steps=200,
        logging_steps=5,
        eval_steps=5,
        do_train=True,
        do_eval=True,
        save_strategy='no',
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_params,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
        formatting_func=formatting_func
    )

    trainer.train()

    trainer.model.save_pretrained(model_saving_path)
    trainer.tokenizer.save_pretrained(model_saving_path)

    print("Model tuned in ", model_saving_path)





if __name__ == '__main__':
    main()