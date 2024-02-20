import os
import copy
import logging
import subprocess
import re

from dataset_loader.dataset_loader import DatasetLoader
from answer_tracker.task import Task
from answer_tracker.answer import Answer


class QuixBugsPythonLoader(DatasetLoader):
    """
    Class for loading and testing the dataset QuixBugs Python.
    """

    def __init__(self) -> None:
        """
        Initialize the QuixBugsPythonLoader.
        """
        super().__init__()
        self.name = "QuixBugs Python"

    def load_prompts(self, max_chain_depth: int, answers_per_task: int) -> None:
        """
        Load prompts for QuixBugs Python dataset.

        :param max_chain_depth: Maximum chain depth.
        :param answers_per_task: Number of answers per task.
        """
        print("Loading " + self.name + " prompts...")
        tasks = []

        # Receive prompt and inst from DatasetLoader
        system_prompt = self.system_prompt

        # Get all python files in QuixBugs
        python_directory = "./QuixBugs/python_programs_bug"
        python_file_list = os.listdir(python_directory)

        for file_name in python_file_list:
            try:
                file_path = os.path.join(python_directory, file_name)
                if os.path.isfile(file_path):
                    # Read the content of each Python file and create prompts
                    with open(file_path, "r") as file:
                        file_data = file.read()

                        prompt = copy.deepcopy(system_prompt)
                        prompt.append(
                            {
                                "role": "user",
                                "content": self.format_inst(file_data, "python"),
                            }
                        )
                        tasks.append(
                            Task(file_name, prompt, max_chain_depth, answers_per_task)
                        )

                else:
                    logging.error(f"'{file_path}' is not a file.")
            except Exception as e:
                logging.error(f"Error reading file '{file_name}': {str(e)}")

        print(self.name + " prompts loaded.\n")
        self.tasks = tasks

    def run_tests(self, program_path: str, answer: Answer) -> None:
        """
        Run tests for the answer.

        :param program_path: Path to the program.
        :param answer: Answer object.
        """
        try:
            # Run the pytest command and capture the output
            pytest_command = f"pytest {program_path}"

            timeout = 60
            if answer.id == "knapsack.py" or answer.id == "levenshtein.py":
                # KNAPSACK and LEVEHSHTEIN might need a long time to finish
                timeout = 300

            result = subprocess.run(
                pytest_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result and result.stdout:
                # Extract the last line from the output
                last_line = result.stdout.strip().splitlines()[-1]

                # Use a regular expression to find the number of failed and passed tests
                passed_match = re.search(r"(\d+) passed", last_line)
                failed_match = re.search(r"(\d+) failed", last_line)

                answer.passed = int(passed_match.group(1)) if passed_match else 0
                answer.failed = int(failed_match.group(1)) if failed_match else 0
            else:
                answer.passed = 0
                answer.failed = 0
                answer.error_message = "Error running pytest subprocess or no output."

        except subprocess.TimeoutExpired as e:
            # Handle timeout and kill the pytest subprocess
            answer.failed = 0
            answer.passed = 0
            answer.other_error = True
            answer.error_message = "Timed out"
            subprocess.run(["pkill", "-f", pytest_command])

    def test_code(self, answer: Answer) -> None:
        """
        Test the provided answer.

        :param answer: Answer object.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # File paths
        dynamic_directory = "./QuixBugs/python_programs"
        test_module_directory = "./QuixBugs/python_testcases"
        program_path = os.path.join(test_module_directory, f"test_{answer.id}")
        dynamic_file_path = os.path.join(dynamic_directory, answer.id)

        # Create the directory if it doesn't exist and write answer to file
        os.makedirs(dynamic_directory, exist_ok=True)
        with open(dynamic_file_path, "w") as file:
            file.write(answer.code)

        # Check syntax errors and run tests on the program
        if answer.code != "":
            answer.syntax_error, answer.error_message = super().check_python_syntax(
                answer.code
            )
        else:
            answer.syntax_error = False
            answer.other_error = True
            answer.error_message = "Empty file, could not extract any code"

        if answer.syntax_error == False:
            self.run_tests(program_path, answer)
        else:
            answer.failed_count, answer.passed_count = 0, 0
