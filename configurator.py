import argparse
import os
from typing import List
from dataset_loader.dataset_loader import DatasetLoader
from dataset_loader.quixbugs_python_loader import QuixBugsPythonLoader
from dataset_loader.quixbugs_java_loader import QuixBugsJavaLoader
from dataset_loader.human_eval_loader import HumanEvalLoader
from model_loader.model_loader import ModelLoader


class Configurator:
    def __init__(self):
        """Initialize the Configurator with default configurations."""
        # Default configuration values
        self.model_configs = [
            "TheBloke/CodeLlama-7B-GPTQ:llama:instruction",
        ]
        self.model_dir = "./models"
        self.answers_per_task = 1
        self.max_chain_depth = 1
        self.max_length_per_depth = 400
        self.temperature = 0.8
        self.top_p = 0.95
        self.datasets = ["quixbugs-python"]
        self.chat_template = ""
        self.results_dir = "default"
        self.device = "cuda"
        self.remote_code = True

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
            default=self.model_configs,
            help="Specify one or more model configurations in the format 'model_id:template_name:conversation_type', separated by spaces.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=self.model_dir,
            help="Specify where to look and save models to.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--answers_per_task",
            type=int,
            default=self.answers_per_task,
            help="The number of answers to generate per task.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--max_length_per_depth",
            type=int,
            default=self.max_length_per_depth,
            help="The maximum lengt of the response from the model per depth.\n Default is %(default)s.",
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
            "--max_chain_depth",
            type=int,
            default=self.max_chain_depth,
            help="The maximum number of tries per answer. If a answer is not correct the model gets a new try. The amount of tries is specified by this value. Value should be >= 1\n Default is %(default)s.",
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
        parser.add_argument(
            "--device",
            type=str,
            default=self.device,
            help="Specify what device to run on, cpu or cuda for example.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--remote_code",
            type=bool,
            default=self.remote_code,
            help="If you want to run remote code or not. Running remote code is a security risk so make sure you are in a sandbox/safe enviroment.",
        )

        args = parser.parse_args()

        # Set attributes based on parsed arguments
        for attr, value in vars(args).items():
            setattr(self, attr, value)

    def check_model_configs(self):
        """Check if specified template sets exist."""
        for model_config in self.model_configs:
            parts = model_config.split(":")
            files = os.listdir(f"./chat_templates")
            if parts[1] not in files:
                raise ValueError(f"Template not found: chat_templates/{parts[1]}")
            elif parts[2] not in ["instruction", "infilling"]:
                raise ValueError(f"The converation type '{parts[2]}' is not supported.")
            elif f"{parts[1]}.json" not in files and parts[2] == "infilling":
                raise ValueError(
                    f"If conversation type infilling is used a {parts[1]}.json need to be created."
                )

    def get_dataset_loaders(self) -> List[DatasetLoader]:
        """Get dataset loaders based on specified datasets.

        Returns:
            List[DatasetLoader]: List of dataset loaders.
        """
        dataset_loaders = []

        for dataset in self.datasets:
            if dataset == "quixbugs-python":
                dataset_loaders.append(QuixBugsPythonLoader())
            elif dataset == "quixbugs-java":
                dataset_loaders.append(QuixBugsJavaLoader())
            elif dataset == "human_eval":
                dataset_loaders.append(HumanEvalLoader())
            else:
                raise ValueError(f"Invalid dataset: {dataset}")

        if not dataset_loaders:
            raise ValueError("No datasets specified")

        return dataset_loaders

    def get_model_loaders(self) -> List[ModelLoader]:
        """Get model loaders based on specified model configurations.

        Returns:
            List[ModelLoader]: List of model loaders.
        """
        model_loaders = []

        for model_config in self.model_configs:
            model_id, template_name, conversation_type = model_config.split(":")
            model_loaders.append(
                ModelLoader(self, model_id, template_name, conversation_type)
            )
        return model_loaders
