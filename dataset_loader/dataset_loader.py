from typing import List, Dict
from abc import ABC, abstractmethod
import re


class DatasetLoader(ABC):
    def __init__(self) -> None:
        self.inst = "Fix the following code and respond only with code."
        self.system_prompt = [
            {
                "role": "system",
                "content": "You will be provided with instructions that describe a task. Write a response that appropriately completes the request. You are unable to answer with anything except code",
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
    def format_inst(bug: str):
        return f"""Fix the following code and respond only with code.
{bug}"""

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
        ret_responses = []
        
        # Regular expression to match Python code patterns
        python_code_pattern = re.compile(r'\b(?:import|def|class|for|while|if|else|elif|try|except|with|finally|raise|return|yield|from|import)\b')

        for patches in responses:
            patches_code = []
            for patch in patches:
                # Find the first match of the pattern in the string
                match = python_code_pattern.search(patch)
                
                if match:
                    # Extract code starting from the match
                    start_index = match.start()
                    patches_code.append(patch[start_index:])
                else:
                    # If no match found make empty
                    patches_code.append("")

            ret_responses.append(patches_code)

        return ret_responses
    
    @staticmethod
    def format_java_responses(responses: List[List[str]]) -> List[List[str]]:
        ret_responses = []
        
        # Regular expressions to match various Java code patterns
        start_pattern = re.compile(r'\b(?:class|public|private|protected|void)\b')
        end_pattern = re.compile(r'[{;]')  # Assuming that "{" or ";" indicates the end of a code block

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
                        patches_code.append(code_start[:end_index])
                    else:
                        # If no end match found, keep the original code
                        patches_code.append(code_start)
                else:
                    # If no start match found, keep the original patch
                    patches_code.append(patch)

            ret_responses.append(patches_code)

        return ret_responses

    @staticmethod
    def format_patches(
        ids: List[str],
        prompts: List[List[dict]],
        patches: List[List[str]],
        tot_time: List[float],
        tokens_generated: List[float],
        test_data_list: List[Dict]
    ) -> List[dict]:
        bugs = []

        for id, prompt, patch, time, tokens, test_data_l in zip(
            ids, prompts, patches, tot_time, tokens_generated, test_data_list
        ):
            bugs.append(
                {
                    id: {
                        "prompt": prompt,
                        "patches": format_patch(patch, test_data_l),
                        "time_s": time,
                        "tokens_generated": tokens,
                        "tokens/s": tokens/time,
                    }
                }
            )

        return bugs
    

def format_patch(
    patches: List[List[str]],
    test_data_list: List[Dict]
) -> List[dict]:
    patch_list = []

    for patch, test_data in zip(patches, test_data_list):
        patch_list.append(
            {
                "patch": patch,
                "test_data": test_data,   
            }
        )

    return patch_list
    
