from typing import List, Dict
from abc import ABC, abstractmethod


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
        for patches in responses:
            patches_code = []
            for patch in patches:
                # Formatting response for python bugs
                index_import = patch.find("import")
                index_def = patch.find("def")
                first_index = min(
                    index_import if index_import != -1 else float("inf"),
                    index_def if index_def != -1 else float("inf"),
                )

                if first_index != float("inf"):
                    patch = patch[int(first_index) :]

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

        for id, prompt, patch, time, tokens, test_data in zip(
            ids, prompts, patches, tot_time, tokens_generated, test_data_list
        ):
            bugs.append(
                {
                    id: {
                        "prompt": prompt,
                        "patches": patch,
                        "time_s": time,
                        "tokens generated": tokens,
                        "tokens/s": tokens/time,
                        "test data": test_data,
                    }
                }
            )

        return bugs
    
