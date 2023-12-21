# Project Name

Brief description of your project.

## Introduction

Provide a brief overview of the project, its purpose, and any key features.

# Installation

To set up the project and its dependencies, follow the steps below:
##  Step 1: Clone the Repository

Clone the project repository to your local machine using the following command:

```bash
git clone https://github.com/anton-ryden/APR_framework.git
```
## Step 2: Navigate to the Project Directory
Navigate to the root directory of the cloned repository.
```bash
cd APR_framework
```

## Step 3: Install Dependencies

Run the setup script to install the required dependencies. However, make sure to read the note on PyTorch and AutoGPTQ before doing so.

```bash
pip install -r requirements.txt
```

## Step 4: Download datasets

Run the setup script to clone repositories for the datasets and make the appropiate changes. 

```bash
python setup.py
```

## Note on PyTorch and AutoGPTQ

This project uses AutoGPTQ, which depends on PyTorch. Depending on your system and whether you plan to use GPU acceleration, you may need to check and adjust the PyTorch version manually. Refer to the PyTorch documentation for information on installing PyTorch with the appropriate CUDA or ROCm version.
Step 4: Run the Project

After successfully completing the setup, you can run the project. Ensure that the virtual environment, if created, is activated. If not, activate it using:

```bash
pyenv activate your-virtual-environment
```
Replace your-virtual-environment with the name of your virtual environment.

Run the project or execute any specific scripts as needed.

```bash
python main.py
```

Congratulations! You have successfully set up and installed the project.

## Usage
Here is an example of how to run the program 
```bash
python main.py 
```
If you are unsure of what to provide the model with and how it affects the program please refer to

```bash
python main.py -h
```