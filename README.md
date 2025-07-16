# LLASP

This is the official repository of our paper **LLASP: Fine-tuning Large Language Models for Answer Set Programming**, by Erica Coppolillo, Francesco Calimeri, Giuseppe Manco, Simona Perri, and Francesco Ricca, accepted at the International Conference on Principles of Knowledge Representation and Reasoning (KR '24).

## USAGE

The `data` folder contains the compressed version of the train, validation and test sets used to perform the experiments.

If you want to regenerate the data, launch `python3 dataset_generation.py`, for train and validation sets, and `python3 test_dataset_generation.py` for the test set.

Launch `python3 model_training.py` to fine-tune the base model (Gemma 2b) and `python3 model_testing.py` to test its performance over the test set.

## CITATION

If you use our code, please cite us:

```
@inproceedings{coppolilloLLASP,
author = {Coppolillo, Erica and Calimeri, Francesco and Manco, Giuseppe and Perri, Simona and Ricca, Francesco},
title = {LLASP: fine-tuning large language models for answer set programming},
year = {2024},
doi = {10.24963/kr.2024/78},
booktitle = {Proceedings of the 21st International Conference on Principles of Knowledge Representation and Reasoning},
series = {KR '24}
}
```
