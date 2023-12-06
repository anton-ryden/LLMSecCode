# Project Name

Brief description of your project.

## Introduction

Provide a brief overview of the project, its purpose, and any key features.

## File Explanation

- **gptq.py**: Main file responsible for loading the model and calling functions.

- **arguments.py**: Used for setting up the argument parser.

- **get_prompts.py**: Handles the retrieval of prompts from the dataset.

- **test_code.py**: Facilitates the testing of whether an answer is a correct fix.


## Installation

You can install all the required dependencies by running:
```
pip install -r requirements.txt
```

However some of the dependencies might need to be installed in another way to get CUDA or Rocm support.
## Usage
Here is an example of how to run the program 
```
python gptq.py --template_set llama --model_id TheBloke/CodeLlama-7B-Instruct-GPTQ --model_path ./models --json_path ./answers.json
```
If you are unsure of what to provide the model with and how it affects the program please refer to
```
python gptq.py -h
```