from typing import List, Dict
import os
import copy
import logging
from dataset_loader.dataset_loader import DatasetLoader


class QuixBugsJavaLoader(DatasetLoader):
    def __init__(self) -> None:
        self.name = "QuixBugs Java"
        self.load_prompts()

    def load_prompts(self) -> List[List[Dict[str, str]]]:
        print("Loading " + self.name + " prompts...")
        prompts = []

        # Receive prompt and inst from DatasetLoader
        system_prompt = self.system_prompt

        # Get all Java files in QuickBugs
        java_dir = "./QuickBugs/java_programs"
        java_list = os.listdir("./QuickBugs/java_programs")

        for file_name in java_list:
            try:
                file_path_full = os.path.join(java_dir, file_name)
                if os.path.isfile(file_path_full):
                    with open(file_path_full, "r") as file:
                        file_data = file.read()

                        prompt = copy.deepcopy(system_prompt)
                        prompt.append(
                            {"role": "user", "content": self.format_inst(file_data)}
                        )
                        prompts.append({file_name: prompt})
                else:
                    logging.error(f"'{file_path_full}' is not a file.")
            except Exception as e:
                logging.error(f"Error reading file '{file_name}': {str(e)}")

        print(self.name + " prompts loaded.}\n")
        self.prompts = prompts

    def format_code_responses(self, response: List[str]) -> List[str]:
        pass
