import os
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Tuple

from patch_tracker.patch import Patch


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    def __init__(self) -> None:
        """
        Initialize the DatasetLoader.
        """
        self.system_prompt = [
            {
                "role": "system",
                "content": "You are a coding assistant.",
            }
        ]
        self.bugs = []
        self.name = ""

    @abstractmethod
    def load_prompts(self, max_chain_depth: int, patches_per_bug: int) -> None:
        """
        Abstract method to load prompts for dataset.

        :param max_chain_depth: Maximum chain depth.
        :param patches_per_bug: Number of patches per bug.
        """
        pass

    @abstractmethod
    def test_code(self, patch: Patch) -> None:
        """
        Abstract method to test code patches.

        :param patch: The patch to test.
        """
        pass

    @staticmethod
    def format_inst(bug: str, language: str) -> str:
        """
        Format instruction for a given bug.

        :param bug: The bug code.
        :param language: The programming language of the bug code.
        :return: Formatted instruction string.
        """
        return f"""Rewrite this function so that you remove any bug. Please return all completed functions in a codeblock:
```{language}
{bug}
```"""

    @staticmethod
    def check_python_syntax(code: str) -> Tuple[bool, str]:
        """
        Check Python syntax errors in code.

        :param code: The Python code to check.
        :return: Tuple indicating whether a syntax error occurred (bool) and the corresponding error message (str).
        """
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

        return syntax_error, error_message

    @staticmethod
    def extract_code(llm_resp_clean: str) -> str:
        """
        Extract code from cleaned response.

        :param llm_resp_clean: The cleaned response containing code.
        :return: Extracted code.
        """
        # Define the regex pattern
        code_pattern = re.compile(r"```([a-zA-Z]+)?\n?\n(.*?)```", re.DOTALL)

        # Find all matches in the patched string
        res_string = ""
        match = re.search(code_pattern, llm_resp_clean)
        if match:
            language, code_block = match.groups()
            fixed_code_block = code_block.lstrip()
            if language:
                fixed_code_block = re.sub(
                    rf"^{language}", "", fixed_code_block, flags=re.MULTILINE
                )
            res_string += fixed_code_block.strip()
        return res_string

    @staticmethod
    def check_java_syntax(file_path: str) -> Tuple[bool, str]:
        """
        Check Java syntax errors in code.

        :param filepath: The location of the java code.
        :return: Tuple indicating whether a syntax error occurred (bool) and the corresponding error message (str).
        """
        error_message = ""
        syntax_error = False
        line_number = None
        file_name = [
            "SHORTEST_PATH_LENGTH.java",
            "SHORTEST_PATHS.java",
            "BREADTH_FIRST_SEARCH.java",
            "TOPOLOGICAL_ORDERING.java",
            "DETECT_CYCLE.java",
            "MINIMUM_SPANNING_TREE.java",
            "REVERSE_LINKED_LIST.java",
            "DEPTH_FIRST_SEARCH.java",
        ]

        try:
            if file_path is not None:
                if os.path.basename(file_path) in file_name:
                    command = [
                        "javac",
                        file_path,
                        "./QuixBugs/java_programs/Node.java",
                        "./QuixBugs/java_programs/WeightedEdge.java",
                    ]
                    # Use subprocess to invoke the Java compiler directly on the file
                    result = subprocess.run(
                        command, check=True, stderr=subprocess.PIPE, text=True
                    )
                else:
                    result = subprocess.run(
                        ["javac", file_path],
                        check=True,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
            else:
                syntax_error = True

        except subprocess.CalledProcessError as e:
            syntax_error = True
            error_message += f"JavaSyntaxError:\n"

            # Check if stderr is not None before decoding
            if e.stderr is not None:
                error_message += e.stderr
            else:
                error_message += "No stderr output available."

        return syntax_error, error_message
