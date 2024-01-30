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

        # Get all Java files in QuixBugs director
        java_directory = "./QuixBugs/java_programs_bug"
        java_file_list = os.listdir(java_directory)

        for file_name in java_file_list:
            try:
                file_path_full = os.path.join(java_directory, file_name)
                if os.path.isfile(file_path_full):
                    # Read the content of each Java file and create prompts
                    with open(file_path_full, "r") as file:
                        file_data = file.read()

                        prompt = copy.deepcopy(system_prompt)
                        prompt.append(
                            {
                                "role": "user",
                                "content": self.format_inst(file_data, "java"),
                            }
                        )
                        new_file_name = file_name.split(".")[0] + ".java"
                        prompts.append({new_file_name: prompt})
                else:
                    logging.error(f"'{file_path_full}' is not a file.")
            except Exception as e:
                logging.error(f"Error reading file '{file_name}': {str(e)}")

        print(self.name + " prompts loaded\n")
        self.prompts = prompts

    def format_code_responses(self, response: List[str]) -> List[str]:
        return super().format_java_responses(response)

    def run_gradle_test(self, class_name: str) -> (int, int):
        original_dir = os.getcwd()
        quixbugs_dir = "./QuixBugs/"
        failed_tests_count = 0
        passed_tests_count = 0

        try:
            if "QuixBugs" not in os.getcwd():
                os.chdir(quixbugs_dir)
            else:
                logging.error(f"Current working directory {os.getcwd()}")

            # Run Gradle test command and capture the output
            gradle_command = f"gradle test --tests {class_name}_TEST"

            timeout = 60
            if class_name == "KNAPSACK" or class_name == "LEVENSHTEIN":
                # KNAPSACK and LEVEHSHTEIN might need a long time to finish
                timeout = 300

            result = subprocess.run(
                gradle_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            os.chdir(original_dir)
            if result:
                output_content = result.stdout + result.stderr

                # Use regular expression to find lines containing test results
                pattern = re.compile(r"(\d+) tests completed, (\d+) failed")
                match = pattern.search(output_content)
                pattern2 = re.compile(r"BUILD SUCCESSFUL")
                match2 = pattern2.search(output_content)

                if match:
                    tests_completed = int(match.group(1))
                    failed_tests_count = int(match.group(2))
                    passed_tests_count = tests_completed - failed_tests_count

                elif match2:
                    # If the second pattern is found, open the corresponding JSON file
                    json_file_path = (
                        f"./QuixBugs/json_testcases/{class_name.lower()}.json"
                    )
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as json_file:
                            # Read the number of lines in the JSON file thats not empty
                            passed_tests_count = sum(
                                1 for line in json_file if line.strip()
                            )
                    else:
                        test_file_path = (
                            f"./QuixBugs/java_testcases/junit/{class_name}_TEST.java"
                        )
                        with open(test_file_path, "r") as file:
                            java_code = file.read()

                        # Get the number of @Test in test code
                        test_instances = re.findall(r"@Test", java_code)
                        passed_tests_count = len(test_instances)

        except subprocess.TimeoutExpired:
            # Handle timeout and kill the pytest subprocess
            failed_tests_count = "null"
            passed_tests_count = "null"
            subprocess.run(["pkill", "-f", gradle_command])
        except Exception as e:
            # Handle any other unexpected exceptions
            logging.error(e)
            failed_tests_count = "null"
            passed_tests_count = "null"
            subprocess.run(["pkill", "-f", gradle_command])

        os.chdir(original_dir)
        return passed_tests_count, failed_tests_count

    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        result_list = []
        bug_nr = 0
        dynamic_directory = "./QuixBugs/java_programs"

        try:
            print("Starting testing for: " + self.name)
            for test_id, bugs in zip(ids, patch_list):
                test_list = []
                for patch_nr, patch in enumerate(bugs, start=1):
                    dynamic_file_path = os.path.join(dynamic_directory, test_id)

                    with open(dynamic_file_path, "w") as file:
                        file.write(patch)

                    class_name = test_id.split(".")[0]
                    syntax_error = None
                    if patch != "":
                        syntax_error = super().check_java_syntax(dynamic_file_path)
                    else:
                        syntax_error = {
                            "syntax_error": "null",
                            "error_message": "Empty file, could not extract any code",
                        }

                    if syntax_error["syntax_error"] != True:
                        passed_count, failed_count = self.run_gradle_test(class_name)
                    else:
                        passed_count, failed_count = "null", "null"

                    test_run_info = {
                        "TestProgramName": test_id,
                        "Passed": passed_count,
                        "Failed": failed_count,
                    }
                    test_list.append({**test_run_info, **syntax_error})

                    before = ""
                    file_path = os.path.join(
                        "./QuixBugs/java_programs_bug", test_id.split(".")[0] + ".txt"
                    )
                    with open(file_path, "r") as file:
                        before = file.read()

                    with open(dynamic_file_path, "w") as file:
                        file.write(before)

                    print_progress_bar(
                        len(bugs) * bug_nr + patch_nr,
                        len(ids) * len(bugs),
                    )

                result_list.append(test_list)
                bug_nr += 1
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        print("\n")
        return result_list
