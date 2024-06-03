# LLASP
LLASP: Fine-tuning Large Language Models for Answer Set Programming

The `data` folder contains the compressed version of the train, validation and test sets used to perform the experiments.

If you want to regenerate the data, launch `python3 dataset_generation.py`, for train and validation sets, and `python3 test_dataset_generation.py` for the test set.

Launch `python3 model_training.py` to fine-tune the base model (Gemma 2b) and `python3 model_testing.py` to test its performance over the test set.

