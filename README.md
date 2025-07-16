# LLASP

This repository contains the experimental code of the extended version of the paper _LLASP: Fine-tuning Large Language Models for Answer Set Programming_, accepted at the KR2024 Conference and currently submitted to the IEEE Transactions on Artificial Intelligence (TAI) Journal.
In the current version, we fine-tuned the Large Language Model to be either **prompt variant** or support **complex** problems, beyond core tasks.

If you want to use the legacy code of the Conference paper, please switch to the "KR2024" branch.

### PRELIMINARIES

To launch the fine-tuning process of the model (Gemma-2B), you need the API token provided by HuggingFace.  

### USAGE

1. To generate the training and validation data used in the training phase, launch `python3 dataset_generation_extension.py`. 
2. Next, launch `python3 test_dataset_generation_extension.py` for generating the test set.
3. Execute `python3 model_training_extension.py` to fine-tune the base model (Gemma 2b).
4. Launch`python3 model_testing_extension.py` to test its performance over the test set.

In each python file, you should modify the variable `turn` to specify which kind of task you are interested in, choosing among "**core**" (for core problems), "**core-invariance**" (for prompt-invariance), and "**complex**" (for complex problems).

### REQUIREMENTS

You can install the required packages by launching `pip install -r requirements.txt`.