from typing import List, Dict
import os
import copy
import logging
from dataset_loader.dataset_loader import DatasetLoader


PYTHON_DIR = "./QuixBugs/python_programs"


class QuixBugsPythonLoader(DatasetLoader):
    def __init__(self) -> None:
        super().__init__()
        self.name = "QuixBugs Python"
        self.load_prompts()

    def load_prompts(self) -> List[List[Dict[str, str]]]:
        print("Loading " + self.name + " prompts...")
        prompts = []

        # Receive prompt and inst from DatasetLoader
        system_prompt = self.system_prompt

        # Get all python files in QuixBugs
        python_list = os.listdir("./QuixBugs/python_programs")

        for file_name in python_list:
            try:
                file_path_full = os.path.join(PYTHON_DIR, file_name)
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

        print(self.name + " prompts loaded.\n")
        self.prompts = prompts

    def format_code_responses(self, responses: List[List[str]]) -> List[List[str]]:
        return super().format_python_responses(responses)
    
    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        return_values = []
        for id, patches in zip(ids, patch_list):
            for patch in patches:
                syntax_error = super().check_python_syntax(patch)
                return_values.append(syntax_error)
        
        return return_values