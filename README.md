# _Fine-Tuning LLMs for Answer Set Programming_

This repository contains the experimental code of the paper _Fine-Tuning LLMs for Answer Set Programming_, accepted at the Journal of Intelligent Information Systems (JIIS), published by Springer.

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

### CITE

If you want to use our code, please cite us.

```bibtex
@article{coppolillo2026finetuning,
	author = {Coppolillo, Erica and Calimeri, Francesco and Manco, Giuseppe and Perri, Simona and Ricca, Francesco},
	doi = {10.1007/s10844-025-01017-4},
	journal = {Journal of Intelligent Information Systems},
	title = {Fine-tuning LLMs for answer set programming},
	year = {2026},
}
```
