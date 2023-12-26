from typing import List, Dict
import os
import copy
import logging
import re
import subprocess
from dataset_loader.dataset_loader import DatasetLoader
from utils import print_progress_bar


class QuixBugsJavaLoader(DatasetLoader):
    def __init__(self) -> None:
        super().__init__()
        self.name = "QuixBugs Java"
        self.load_prompts()

    def load_prompts(self) -> List[List[Dict[str, str]]]:
        print("Loading " + self.name + " prompts...")
        prompts = []

        # Receive prompt and inst from DatasetLoader
        system_prompt = self.system_prompt

        # Get all Java files in QuickBugs
        java_dir = "./java_programs_bug"
        java_list = os.listdir("./java_programs_bug")

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
       return super().format_java_responses(response)

    def run_gradle_test(self, test_name: str) -> (int, int):
            original_dir= os.getcwd()
            quixbugs_dir = "./QuixBugs/"        
            if "QuixBugs" not in os.getcwd():
                os.chdir(quixbugs_dir)
            else:
                logging.error(f'Current wokring directory {os.getcwd()}')
            
            output_file = "output.txt"
            command = f"gradle test --tests {test_name}"
                    
            try:
                result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                os.chdir(original_dir)
                with open(output_file, "w") as output_file_writer:
                    output_file_writer.write(result.stdout.decode())
                    output_file_writer.write(result.stderr.decode())

                with open(output_file, "r") as output_file_reader:
                    output_content = output_file_reader.read()  
                # Use regular expression to find lines containing test results
                pattern = re.compile(r'(\d+) tests completed, (\d+) failed')
                match = pattern.search(output_content)

                if match:
                    tests_completed = int(match.group(1))
                    tests_failed = int(match.group(2))
                    return tests_completed, tests_failed
                else:
                    logging.error("Test results not found in the output.")
                    return 0, 0
            except subprocess.CalledProcessError as e:
                # Return the stdout of the error if the command fails
                logging.error("Subprocess creation failed")
                return 0, 0


    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        result_list = []
        bug_nr = 0

        print("Starting testing for: " + self.name)
        for id, bugs in zip(ids, patch_list):
            test_list = []
            for patch_nr, patch in enumerate(bugs, start=1):
                
                dynamic_directory = "./QuixBugs/java_programs"
                dynamic_file_path = os.path.join(dynamic_directory, id)
                # Create the directory if it doesn't exist
                os.makedirs(dynamic_directory, exist_ok=True)
                with open(dynamic_file_path, 'w') as file:
                    file.write(patch)
                class_name = id.split('.')[0]
                test_name = f"{class_name}_TEST"
                if patch != "":
                    syntax_error = super().check_java_syntax(dynamic_file_path)
                else:
                    syntax_error = {
                        "syntax_error": "null",
                        "error_message": "Empty file, could not extract any code",
                    }

                if syntax_error["syntax_error"] == False:
                    total_count, failed_count = self.run_gradle_test(test_name)
                    passed_count = int(total_count)-int(failed_count)
                else:
                    passed_count, failed_count  = "null", "null"

                test_run_info = {
                    "TestProgramName": id,
                    "Passed": passed_count,
                    "Failed": failed_count
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
