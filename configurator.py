import argparse
from typing import List
from dataset_loader.dataset_loader import DatasetLoader
from dataset_loader.quixbugs_python_loader import QuixBugsPythonLoader
from dataset_loader.quixbugs_java_loader import QuixBugsJavaLoader
from dataset_loader.defect4j_loader import Defect4JLoader
from dataset_loader.human_eval_loader import HumanEvalLoader


class Configurator:
    def __init__(self):
        self.template_set = "codellama"
        self.model_id = "TheBloke/CodeLlama-7B-Instruct-GPTQ"
        self.model_dir = "./models"
        self.json_path = "./answers.json"
        self.patches_per_bug = 2
        self.max_new_tokens = 3000
        self.temperature = 1.0
        self.top_p = 0.95
        self.datasets = ["quixbugs-python"]
        self.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

        self.parse_args()
        self.set_chat_template()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="This is option of arguments.")
        parser.add_argument(
            "--template_set",
            choices=["llama", "wizardcode", "codellama"],
            default=self.template_set,
            help="Choose the set of templates to use (llama or codewizard).\n Default is %(default)s.",
        )
        parser.add_argument(
            "--model_id",
            type=str,
            default=self.model_id,
            help="Specify the Hugging Face model ID ex: TheBloke/CodeLlama-7B-Instruct-GPTQ.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            default=self.model_dir,
            help="Specify where to look and save models to.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--json_path",
            type=str,
            default=self.json_path,
            help="Specify path and name for the JSON file. Default is %(default)s.",
        )
        parser.add_argument(
            "--patches_per_bug",
            type=int,
            default=self.patches_per_bug,
            help="The number of patches to generate per bug",
        )
        parser.add_argument(
            "--max_new_tokens",
            type=int,
            default=self.max_new_tokens,
            help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.\n Default is %(default)s.",
        )
        parser.add_argument(
            "--temperature",
            type=int,
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
        args = parser.parse_args()

        # Set attributes based on parsed arguments
        for attr, value in vars(args).items():
            setattr(self, attr, value)

    def set_chat_template(self) -> str:
        template_llama = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        template_codewizard = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{{ messages[0]['content'].strip() }}{% elif message['role'] == 'user' %}{{ bos_token + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'].strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
        template_set = self.template_set

        if template_set == "llama" or template_set == "codellama":
            self.chat_template = template_llama
        elif template_set == "wizardcode":
            self.chat_template = template_codewizard
        else:
            raise ValueError(f"Invalid template set: {template_set}")

    def get_dataset_loader(self) -> List[DatasetLoader]:
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

        if len(dataset_loaders) == 0:
            raise ValueError("No datasets specified")

        return dataset_loaders
