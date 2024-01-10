from typing import List, Dict
import os
import copy
import logging
import re
import subprocess
import shutil
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
                            {"role": "user", "content": self.format_inst(file_data, "java")}
                        )
                        prompts.append({file_name: prompt})
                else:
                    logging.error(f"'{file_path_full}' is not a file.")
            except Exception as e:
                logging.error(f"Error reading file '{file_name}': {str(e)}")

        print(self.name + " prompts loaded\n")
        self.prompts = prompts

    def format_code_responses(self, response: List[str]) -> List[str]:
       return super().format_java_responses(response)

    def run_gradle_test(self, class_name: str) -> (int, int):
            original_dir= os.getcwd()
            quixbugs_dir = "./QuixBugs/" 
            try:       
                if "QuixBugs" not in os.getcwd():
                    os.chdir(quixbugs_dir)
                else:
                    logging.error(f'Current wokring directory {os.getcwd()}')
                
                output_file = "output.txt"
                command = f"gradle test --tests {class_name}_TEST"
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
                pattern2 = re.compile(r'BUILD SUCCESSFUL')
                match2 = pattern2.search(output_content)

                if match:
                    tests_completed = int(match.group(1))
                    tests_failed = int(match.group(2))
                    tests_passed = tests_completed - tests_failed
                    return tests_passed, tests_failed
                elif match2:
                    # If the second pattern is found, open the corresponding JSON file
                    try:
                        json_file_path = f'./QuixBugs/json_testcases/{class_name.lower()}.json'
                        if os.path.exists(json_file_path):
                            with open(json_file_path, 'r') as json_file:
                            # Read the number of lines in the JSON file
                                tests_passed = sum(1 for line in json_file)
                            return tests_passed, 0
                        else:
                            test_file_path = f'./java_testcases/junit/{class_name}_TEST.java'
                            with open(test_file_path, 'r') as file:
                                java_code = file.read()
                            # Get the number of @Test in test code
                            test_instances = re.findall(r'@Test', java_code)
                            return len(test_instances),0
                            
                    except FileNotFoundError as e:
                        logging.error(f'File not found {e}')
                        return 0,0
                else:
                    logging.error('Test results not found in the output.')
                    return 0, 0
            except subprocess.CalledProcessError as e:
                # Return the stdout of the error if the command fails
                logging.error(f'Subprocess creation failed{e}')
                return 0, 0
            except Exception as e:
                 # Handle any other unexpected exceptions
                logging.error(f'An unexpected error occurred: {e}')
                return 0, 0


    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        result_list = []
        bug_nr = 0
        try:
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
                    flag=0
                    if patch != "":
                        flag=1
                        syntax_error = super().check_java_syntax(dynamic_file_path)
                    else:
                        syntax_error = {
                            "syntax_error": "null",
                            "error_message": "Empty file, could not extract any code",
                        }
                    
                    if syntax_error["syntax_error"] == False and flag==1:
                        # passed_count, failed_count = self.run_gradle_test(class_name)
                        # Define paths for source and destination
                        source_path = f"./java_testcases/junit/{class_name}_TEST.java"
                        destination_path = f"./QuixBugs/java_testcases/junit/{class_name}_TEST.java"
                        java_file_path = f"./QuixBugs/java_programs/{class_name}.java"
                        class_file_path = f"./QuixBugs/java_programs/{class_name}.class"
                        # Copy the test file
                        shutil.copy(source_path, destination_path)
                        # Execute run_gradle_test function
                        passed_count, failed_count = self.run_gradle_test(class_name)
                        os.remove(java_file_path)
                        os.remove(class_file_path)
                        os.remove(destination_path)                       
                    else:
                        java_file_path = f"./QuixBugs/java_programs/{class_name}.java"
                        passed_count, failed_count  = "null", "null"
                        os.remove(java_file_path)
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
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        print("\n")
        return result_list
