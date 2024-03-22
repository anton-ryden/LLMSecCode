import os
import subprocess
from abc import ABC, abstractmethod
from typing import Tuple

from data_structures.answer import Answer
from model_loader.model_loader import ModelLoader


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
        self.prompts = []
        self.name = ""
        self.area = ""

    @abstractmethod
    def load_prompts(self, max_chain_depth: int, answers_per_task: int) -> None:
        """
        Abstract method to load prompts for dataset.

        Args:
            max_chain_depth (int): Maximum chain depth.
            answers_per_task (int): Number of answers per task.
        """
        pass

    @abstractmethod
    def test_code(self, answers: list[Answer], model: ModelLoader) -> None:
        """
        Abstract method to test code answers.

        Args:
            answer (Answer): The answer to test.
        """
        pass

    @staticmethod
    def format_inst(code: str, language: str) -> str:
        """
        Format instruction for a given task.

        Args:
            code (str): The code.
            language (str): The programming language of the code.
        Returns:
            str: Formatted instruction string.
        """
        return f"""Rewrite this function so that you remove any bug. Please return all completed functions in a codeblock:
```{language}
{code}
```"""

    @staticmethod
    def check_python_syntax(code: str) -> Tuple[bool, str]:
        """
        Check Python syntax errors in code.

        Args:
            code (str): The Python code to check.
        Returns:
            Tuple[bool, str]: Tuple indicating whether a syntax error occurred (bool) and the corresponding error message (str).
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
            error_line = lines[line_number - 2]

            error_message += (
                f"SyntaxError at line {line_number}, column {column_offset}:\n"
            )
            error_message += error_line + "\n"
            error_message += " " * (column_offset - 1) + "^\n"
            error_message += f"{type(e).__name__}: {e}\n"
            syntax_error = True

        return syntax_error, error_message

    @staticmethod
    def check_java_syntax(file_path: str) -> Tuple[bool, str]:
        """
        Check Java syntax errors in code.

        Args:
            file_path (str): The location of the java code.
        Returns:
            Tuple[bool, str]: Tuple indicating whether a syntax error occurred (bool) and the corresponding error message (str).
        """
        error_message = ""
        syntax_error = False
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
                # error_message += e.stderr
                error_message += ""
            else:
                error_message += "No stderr output available."
        return syntax_error, error_message
