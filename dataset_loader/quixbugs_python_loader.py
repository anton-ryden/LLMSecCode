from typing import List, Dict
import os
import copy
import logging
import subprocess
import re
from dataset_loader.dataset_loader import DatasetLoader
from utils import print_progress_bar


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
        python_dir = "./QuixBugs/python_programs_bug"
        python_list = os.listdir(python_dir)

        for file_name in python_list:
            try:
                file_path_full = os.path.join(python_dir, file_name)
                if os.path.isfile(file_path_full):
                    with open(file_path_full, "r") as file:
                        file_data = file.read()

                        prompt = copy.deepcopy(system_prompt)
                        prompt.append(
                            {"role": "user", "content": self.format_inst(file_data, "python")}
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

    def run_tests(self, program_path: str) -> (int, int):
        try:
            command = f"pytest {program_path}"

            # Run the pytest command and capture the output
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            if result:
                # Extract the last line from the output
                last_line = result.stdout.strip().splitlines()[-1]
                # Use a regular expression to find the number of failed and passed tests
                match = re.search(r"(\d+)(?: failed)?(?:,\s*(\d+) passed)?", last_line)

                # Extract the number of failed and passed tests if a match is found
                failed_tests_count = (
                    int(match.group(1)) if match.group(1) is not None else 0
                )
                passed_tests_count = (
                    int(match.group(2)) if match.group(2) is not None else 0
                )
            else:
                failed_tests_count = 0
                passed_tests_count = 0

        except subprocess.TimeoutExpired:
            failed_tests_count = "null"
            passed_tests_count = "null"
        
        return failed_tests_count, passed_tests_count

    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        result_list = []
        bug_nr = 0

        print("Starting testing for: " + self.name)
        for id, bugs in zip(ids, patch_list):
            test_list = []
            for patch_nr, patch in enumerate(bugs, start=1):
                # File paths
                dynamic_directory = "./QuixBugs/python_programs"
                test_module_directory = "./QuixBugs/python_testcases"
                program_paths = os.path.join(test_module_directory, f"test_{id}")
                dynamic_file_path = os.path.join(dynamic_directory, id)

                # Create the directory if it doesn't exist and write patch to file
                os.makedirs(dynamic_directory, exist_ok=True)
                with open(dynamic_file_path, "w") as file:
                    file.write(patch)

                if patch != "":
                    syntax_error = super().check_python_syntax(patch)
                else:
                    syntax_error = {
                        "syntax_error": "null",
                        "error_message": "Empty file, could not extract any code",
                    }

                if syntax_error["syntax_error"] == False:
                    failed_count, passed_count = self.run_tests(program_paths)
                else:
                    failed_count, passed_count = "null", "null"

                test_run_info = {
                    "TestProgramName": id,
                    "Passed": passed_count,
                    "Failed": failed_count,
                }
                test_list.append({**test_run_info, **syntax_error})

                print_progress_bar(
                    len(bugs) * bug_nr + patch_nr,
                    len(ids) * len(bugs),
                )

            result_list.append(test_list)
            bug_nr += 1
        print("\n")
        return result_list
