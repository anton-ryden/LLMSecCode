import argparse
import os
from typing import List
from dataset_loader.dataset_loader import DatasetLoader
from dataset_loader.quixbugs_python_loader import QuixBugsPythonLoader
from dataset_loader.quixbugs_java_loader import QuixBugsJavaLoader
from dataset_loader.defect4j_loader import Defect4JLoader
from dataset_loader.human_eval_loader import HumanEvalLoader
from model_loader.model_loader import ModelLoader


class Configurator:
    def __init__(self):
        # Default configuration values
        self.model_configs = "TheBloke/CodeLlama-7B-Instruct-GPTQ:llama"
        self.model_dir = "./models"
        self.patches_per_bug = 4
        self.max_length = 4096
        self.temperature = 0.8
        self.top_p = 0.95
        self.datasets = ["quixbugs-python"]
        self.chat_template = ""
        self.results_dir = "default"

        # Parse command line arguments and check model configurations
        self.parse_args()
        self.check_model_configs()

    def parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="This is option of arguments.")
        parser.add_argument(
            "--model_configs",
            type=str,
            nargs="+",
            default=[f"{self.model_configs}"],
            help="Specify one or more model configurations in the format 'model_id:template_name', separated by spaces.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=self.model_dir,
            help="Specify where to look and save models to.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--patches_per_bug",
            type=int,
            default=self.patches_per_bug,
            help="The number of patches to generate per bug",
        )
        parser.add_argument(
            "--max_length",
            type=int,
            default=self.max_length,
            help="The maximum lengt of the resposne from the model.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=self.temperature,
            help="The temperature used in generation, higher value -> more diverse answers.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--top_p",
            type=int,
            default=self.top_p,
            help="Top p, also known as nucleus sampling, is another hyperparameter that controls the randomness of language model output. It sets a threshold probability and selects the top tokens whose cumulative probability exceeds the threshold.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--datasets",
            nargs="+",
            default=self.datasets,
            choices=["quixbugs-python", "quixbugs-java", "defect4j", "human_eval"],
            help="Choose one or more datasets from 'quixbugs-python', 'quixbugs-java', 'defect4j', and 'human_eval'. Default is 'quixbugs defect4j human_eval'.",
        )
        parser.add_argument(
            "--results_dir",
            type=str,
            default=self.results_dir,
            help="Specify name of folder to save results to.\n Default is %(default)s.",
        )

        args = parser.parse_args()

        # Set attributes based on parsed arguments
        for attr, value in vars(args).items():
            setattr(self, attr, value)

    def check_model_configs(self):
        """Check if specified template sets exist."""
        files = os.listdir("./prompt_templates")

        for model_config in self.model_configs:
            _, template_set = model_config.split(":")
            if template_set not in files:
                raise ValueError(f"Invalid template set: {template_set}")

    def get_dataset_loaders(self) -> List[DatasetLoader]:
        """Get dataset loaders based on specified datasets."""
        dataset_loaders = []

        for dataset in self.datasets:
            if dataset == "quixbugs-python":
                dataset_loaders.append(QuixBugsPythonLoader())
            elif dataset == "quixbugs-java":
                dataset_loaders.append(QuixBugsJavaLoader())
            elif dataset == "defect4j":
                dataset_loaders.append(Defect4JLoader())
            elif dataset == "human_eval":
                dataset_loaders.append(HumanEvalLoader())
            else:
                raise ValueError(f"Invalid dataset: {dataset}")

        if not dataset_loaders:
            raise ValueError("No datasets specified")

        return dataset_loaders

    def get_model_loaders(self) -> List[ModelLoader]:
        """Get model loaders based on specified model configurations."""
        model_loaders = []

        for model_config in self.model_configs:
            model_id, template_name = model_config.split(":")
            model_loaders.append(ModelLoader(self, model_id, template_name))

        return model_loaders
