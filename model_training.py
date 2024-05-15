hugging_token = "[TOKEN]"

from huggingface_hub import login

login(hugging_token)

import os
from tqdm import tqdm

import sys
import subprocess

import torch
import pandas as pd

from datasets import load_dataset, Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging, LlamaForCausalLM, LlamaTokenizer
)
from peft import LoraConfig
from trl import SFTTrainer


model_to_use = "gemma"

base_model = "google/gemma-2b-it"

new_model = "[FINETUNED MODEL NAME]"

output_dir = "[OUTPUT DIRECTORY]"

new_model_path = output_dir + new_model


def formatting_func(example):
    text = f"Question: {example['question'][0]}\nAnswer: {example['answer'][0]}"
    return [text]


# TODO: CHANGE PATH IF NEEDED
train_df_fn = "train_df.csv"
val_df_fn = "val_df.csv"

train_df = pd.read_csv(train_df_fn)
val_df = pd.read_csv(val_df_fn)

train_dataset = Dataset.from_dict(train_df)
val_dataset = Dataset.from_dict(val_df)

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)


TRAIN = True

if TRAIN:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        token=hugging_token,
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

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

    trainer.model.save_pretrained(new_model_path)
    trainer.tokenizer.save_pretrained(new_model_path)
