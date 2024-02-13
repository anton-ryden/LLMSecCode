from typing import Tuple
import os
import copy
import logging
import re
import subprocess

from dataset_loader.dataset_loader import DatasetLoader
from patch_tracker.patch import Patch
from patch_tracker.bug import Bug


class QuixBugsJavaLoader(DatasetLoader):
    """
    Class for loading and testing the dataset QuixBugs Java.
    """

    def __init__(self) -> None:
        """
        Initialize the QuixBugsPythonLoader.
        """
        super().__init__()
        self.name = "QuixBugs Java"

    def load_prompts(self, max_chain_depth: int, patches_per_bug: int) -> None:
        """
        Load prompts for QuixBugs Python dataset.

        :param max_chain_depth: Maximum chain depth.
        :param patches_per_bug: Number of patches per bug.
        """
        print("Loading " + self.name + " prompts...")
        bugs = []

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
                        bugs.append(
                            Bug(new_file_name, prompt, max_chain_depth, patches_per_bug)
                        )
                else:
                    logging.error(f"'{file_path_full}' is not a file.")
            except Exception as e:
                logging.error(f"Error reading file '{file_name}': {str(e)}")

        print(self.name + " prompts loaded\n")
        self.bugs = bugs

    def run_gradle_test(self, class_name: str) -> Tuple[int, int]:
        """
        Run Gradle tests for a specified Java class.

        :param class_name: The name of the Java class to run tests for.
        :return: A tuple containing the number of passed tests and the number of failed tests.
        """
        original_dir = os.getcwd()
        quixbugs_dir = "./QuixBugs/"
        failed_tests = 0
        passed_tests = 0

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
                    failed_tests = int(match.group(2))
                    passed_tests = tests_completed - failed_tests

                elif match2:
                    # If the second pattern is found, open the corresponding JSON file
                    json_file_path = (
                        f"./QuixBugs/json_testcases/{class_name.lower()}.json"
                    )
                    if os.path.exists(json_file_path):
                        with open(json_file_path, "r") as json_file:
                            # Read the number of lines in the JSON file thats not empty
                            passed_tests = sum(1 for line in json_file if line.strip())
                    else:
                        test_file_path = (
                            f"./QuixBugs/java_testcases/junit/{class_name}_TEST.java"
                        )
                        with open(test_file_path, "r") as file:
                            java_code = file.read()

                        # Get the number of @Test in test code
                        test_instances = re.findall(r"@Test", java_code)
                        passed_tests = len(test_instances)

        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-f", gradle_command])
        except Exception as e:
            # Handle any other unexpected exceptions
            logging.error(e)
            subprocess.run(["pkill", "-f", gradle_command])

        os.chdir(original_dir)
        return passed_tests, failed_tests

    def test_code(self, patch: Patch) -> None:
        """
        Test the provided patch.

        :param patch: Patch object.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        dynamic_directory = "./QuixBugs/java_programs"

        try:
            dynamic_file_path = os.path.join(dynamic_directory, patch.id)

            with open(dynamic_file_path, "w") as file:
                file.write(patch.code)

            class_name = patch.id.split(".")[0]
            if patch.code != "":
                patch.syntax_error, patch.error_message = super().check_java_syntax(
                    dynamic_file_path
                )
            else:
                patch.other_error = True
                patch.error_message = "Empty file, could not extract any code"

            if patch.syntax_error != True and patch.other_error != True:
                patch.passed, patch.failed = self.run_gradle_test(class_name)

            before = ""
            file_path = os.path.join(
                "./QuixBugs/java_programs_bug", patch.id.split(".")[0] + ".txt"
            )

            # Overwrite to original state, if this is not done is could introduce errors for other patches.
            with open(file_path, "r") as file:
                before = file.read()

            with open(dynamic_file_path, "w") as file:
                file.write(before)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
