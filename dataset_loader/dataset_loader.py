from typing import List, Dict
from abc import ABC, abstractmethod
import os
import re
import subprocess


class DatasetLoader(ABC):
    def __init__(self) -> None:
        self.system_prompt = [
            {
                "role": "system",
                "content": "You are a coding assistant.",
            }
        ]
        self.prompts = []
        self.name = ""

    @abstractmethod
    def load_prompts(self) -> List[List[Dict[str, str]]]:
        pass

    @abstractmethod
    def format_code_responses(self, response: str) -> None:
        pass

    @abstractmethod
    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        pass

    @staticmethod
    def format_inst(bug: str, language: str) -> str:
        return f"""
Please repair the buggy function. You are only allowed to modify the given code. Please return all completed function in a codeblock. Here is the given code to do repair:
```{language}
{bug}
```"""

    @staticmethod
    def check_python_syntax(code: str) -> dict:
        error_message = ""
        syntax_error = False
        try:
            if code is not None:
                compile(code, filename="<string>", mode="exec")
                syntax_error = False
            else:
                syntax_error = True

        except SyntaxError as e:
            # Extract line and column information
            line_number, column_offset = e.lineno, e.offset
            lines = code.split("\n")
            error_line = lines[line_number - 1]

            error_message += (
                f"SyntaxError at line {line_number}, column {column_offset}:\n"
            )
            error_message += error_line + "\n"
            error_message += " " * (column_offset - 1) + "^\n"
            error_message += f"{type(e).__name__}: {e}\n"
            syntax_error = True

        return {"syntax_error": syntax_error, "error_message": error_message}

    @staticmethod
    def format_python_responses(responses: List[List[str]]) -> List[List[str]]:
        def extract_code_from_patch(patch: str) -> str:
                patch = patch.replace("\t", "    ")
                patch = patch.replace("\\n", "\n")
                patch = patch.strip()
                # Define the regex pattern
                pattern = re.compile(r'\b(?:import|from|def)\b|\w*(?:import|from|def)\w*')

                # Find all matches in the patched string
                matches = pattern.finditer(patch)

                # Extract the start index of each match and store them in a list
                function_indices = [match.start() for match in matches]
                function_indices.sort()

                for i, function_index in enumerate(function_indices):
                    temp = patch[function_index:]

                    lines = temp.split("\n")
                    line_end_index = 0
                    for i, line in enumerate(lines[1:]):
                        pattern2 = re.compile(r'\b(import|from|def)\b')
                        if pattern2.search(line):
                            continue
                        if len(line) > 0 and line[0] != " ":
                            for j in range(i+1):
                                line_end_index += len(lines[j])+1

                            return patch[function_index:function_index+line_end_index]   
                        
                    return patch[function_index:]                     

                return ""
        

        ret_responses = []
        for patches in responses:
            patches_code = []
            for patch in patches:
                code = extract_code_from_patch(patch)
                patches_code.append(code)

            ret_responses.append(patches_code)

        return ret_responses

    @staticmethod
    def format_java_responses(responses: List[List[str]]) -> List[List[str]]:
        ret_responses = []

        # Regular expressions to match various Java code patterns
        start_pattern = re.compile(
            r"\b(?:package|class|public|private|protected|void)\b"
        )
        end_pattern = re.compile(
            r"\b(?:}\s*//\s*end\sof\s+class\b|}\s*//\s*end\sof\s+method\b)\b"
        )

        for patches in responses:
            patches_code = []
            for patch in patches:
                # Find the first match of the start pattern in the string
                start_match = start_pattern.search(patch)

                if start_match:
                    # Extract code starting from the match
                    start_index = start_match.start()
                    code_start = patch[start_index:]

                    # Find the first match of the end pattern after the start
                    end_match = end_pattern.search(code_start)

                    if end_match:
                        # Extract code up to the end match
                        end_index = end_match.start()
                        java_code_cleaned = code_start[:end_index]
                        # Check for triple backticks and remove content after them
                        triple_backticks_index = java_code_cleaned.find("```")
                        if triple_backticks_index != -1:
                            java_code_cleaned = java_code_cleaned[
                                :triple_backticks_index
                            ]
                        patches_code.append(java_code_cleaned)
                    else:
                        java_code_cleaned = code_start
                        # Check for triple backticks and remove content after them
                        triple_backticks_index = java_code_cleaned.find("```")
                        if triple_backticks_index != -1:
                            java_code_cleaned = java_code_cleaned[
                                :triple_backticks_index
                            ]
                        patches_code.append(java_code_cleaned)
                        # If no end match found, keep the original code
                        # patches_code.append(code_start)
                else:
                    java_code_cleaned = patch
                    # Check for triple backticks and remove content after them
                    triple_backticks_index = java_code_cleaned.find("```")
                    if triple_backticks_index != -1:
                        java_code_cleaned = java_code_cleaned[:triple_backticks_index]
                    patches_code.append(java_code_cleaned)
                    # If no start match found, keep the original patch
                    # patches_code.append(patch)

            ret_responses.append(patches_code)

        return ret_responses

    @staticmethod
    def check_java_syntax(file_path: str) -> dict:
        error_message = ""
        syntax_error = False
        line_number = None
        file_name=["SHORTEST_PATH_LENGTH.java","SHORTEST_PATHS.java", "BREADTH_FIRST_SEARCH.java", "TOPOLOGICAL_ORDERING.java", "DETECT_CYCLE.java", "MINIMUM_SPANNING_TREE.java", "REVERSE_LINKED_LIST.java", "DEPTH_FIRST_SEARCH.java"]

        try:
            if file_path is not None:
                if os.path.basename(file_path) in file_name:
                    command = ['javac' ,file_path, "./QuixBugs/java_programs/Node.java", "./QuixBugs/java_programs/WeightedEdge.java"]
                    # Use subprocess to invoke the Java compiler directly on the file
                    result = subprocess.run(
                        command, check=True, stderr=subprocess.PIPE, text=True
                    )
                else:
                    result = subprocess.run(
                        ["javac", file_path], check=True, stderr=subprocess.PIPE, text=True
                    )
            else:
                syntax_error = True

        except subprocess.CalledProcessError as e:
            syntax_error = True
            error_message += f"JavaSyntaxError:\n"

            # Check if stderr is not None before decoding
            if e.stderr is not None:
                error_message += e.stderr

                # Extract line number from the error message
                match = re.search(r"error:.*:(\d+):", e.stderr)
                if match:
                    line_number = int(match.group(1))

            else:
                error_message += "No stderr output available."

        return {
            "syntax_error": syntax_error,
            "error_message": error_message,
            "line_number": line_number,
        }

    @staticmethod
    def format_patches(
        responses: List[List[str]],
        ids: List[str],
        prompts: List[List[dict]],
        patches: List[List[str]],
        tot_time: List[float],
        tokens_generated: List[float],
        test_result_list: List[Dict],
    ) -> List[dict]:
        bugs = []

        if(len(test_result_list) > len(ids)):
                bugs.append(test_result_list[-1])

        for response, id, prompt, patch, time, tokens, test_result_1 in zip(
            responses,
            ids,
            prompts,
            patches,
            tot_time,
            tokens_generated,
            test_result_list,
        ):
            bugs.append(
                {
                    id: {
                        "response": response,
                        "prompt": prompt,
                        "patches": format_patch(patch, test_result_1),
                        "time_s": time,
                        "tokens_generated": tokens,
                        "tokens/s": tokens / time,
                    }
                }
            )

        return bugs


def format_patch(patches: List[List[str]], test_result_list: List[Dict]) -> List[dict]:
    patch_list = []

    for patch, test_result in zip(patches, test_result_list):
        patch_list.append(
            {
                "patch": patch,
                "test_result": test_result,
            }
        )

    return patch_list
