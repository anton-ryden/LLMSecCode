import re
import copy
import json

from data_structures.prompt import Prompt


class Answer:
    """
    Represents an answer generated by the model.
    """

    def __init__(
        self, id: str, prompt_instance: Prompt, conversation_type: str, chain_depth: int
    ) -> None:
        """
        Initializes an Answer object.

        Args:
            id (str): Unique identifier for the answer.
            prompt_instance (Prompt): Prompt for generating the answer.
            conversation_type (str): Type of conversation (conversation, completion, infilling).
            chain_depth (int): Depth of the answer in the conversation chain.
        """
        self.id = id
        self.prompt_instance = prompt_instance
        self.conversation_type = conversation_type
        self.chain_depth = chain_depth
        self.syntax_error = False
        self.other_error = False
        self.error_message = ""
        self.llm_resp = ""
        self.llm_resp_clean = ""
        self.code = ""
        self.infill_piece = ""
        self.time_to_gen = 0
        self.tokens_generated = 0
        self.passed = 0
        self.failed = 0
        self.memory = 0

    def get_next_chain(self) -> "Answer":
        """
        Generates the next answer in the conversation chain.

        Returns:
            Answer: A new Answer object representing the next answer in the chain.
        """
        new_prompt = copy.deepcopy(self.prompt_instance.prompt)
        new_prompt.append({"role": "assistant", "content": self.code})
        if self.syntax_error:
            new_prompt.append(
                {
                    "role": "user",
                    "content": f"The code you provided contain a syntax error. The syntax error: {self.error_message}\nChange the code and return a updated version in a codeblock",
                }
            )
        elif self.other_error:
            new_prompt.append(
                {
                    "role": "user",
                    "content": f"The code you provided caused an error. The error: {self.error_message}\nChange the code and return a updated version in a codeblock",
                }
            )
        else:
            new_prompt.append(
                {
                    "role": "user",
                    "content": f"The code did not pass all test cases. Improve the code and return a updated version in a codeblock",
                }
            )

        prompt_instance = copy.copy(self.prompt_instance)
        prompt_instance.prompt = new_prompt
        return Answer(
            self.id, prompt_instance, self.conversation_type, self.chain_depth + 1
        )

    def detailed_json(self):
        """
        Converts the answer to a detailed JSON format.

        Returns:
            dict: Detailed JSON representation of the answer.
        """
        return {
            "Id": self.id,
            "Prompt": self.prompt_instance.prompt,
            "Chain depth": self.chain_depth,
            "Syntax error": self.syntax_error,
            "Other error": self.other_error,
            "Error message": self.error_message,
            "LLM resp": self.llm_resp,
            "LLM resp clean": self.llm_resp_clean,
            "Code": self.code,
            "Time(sec)": self.time_to_gen,
            "Tokens generated": self.tokens_generated,
            "Passed": self.passed,
            "Failed": self.failed,
        }

    def extract_code(self, template_name: str):
        """
        Extracts code from the cleaned response and sets the Answer's code attribute.

        Args:
            template_name (str): Name of the template used.
        """
        if self.conversation_type == "instruction":
            self.code = self.extract_conversation()
            self.code = self.code.strip()
        elif self.conversation_type == "completion":
            self.code = self.extract_completion()
        elif self.conversation_type == "infilling":
            self.code = self.extract_infilling(
                self.extract_conversation(),
                template_name,
            )
            infilling_piece = self.code.replace(self.prompt_instance.prefix, "")
            infilling_piece = infilling_piece.replace(self.prompt_instance.suffix, "")
            self.infill_piece = infilling_piece
        else:
            raise NotImplemented("This mode is not supported.")

    def extract_conversation(self):
        """
        Extracts code from a conversation type response.

        Returns:
            str: Extracted code.
        """
        # Define the regex pattern
        code_pattern = re.compile(r"\n?```([a-zA-Z]+)?\n?\n(.*?)```", re.DOTALL)

        # Find all matches in the string
        res_string = ""
        match = re.search(code_pattern, self.llm_resp_clean)
        if match:
            language, code_block = match.groups()
            fixed_code_block = code_block.lstrip()
            if language:
                fixed_code_block = re.sub(
                    rf"^{language}", "", fixed_code_block, flags=re.MULTILINE
                )
            res_string += fixed_code_block.strip()
        else:
            res_string = re.sub(r"```", "", self.llm_resp_clean, flags=re.MULTILINE)

        return res_string

    def extract_completion(self):
        """
        Extracts code from a completion type response.

        Returns:
            str: Extracted code.
        """
        start = self.prompt_instance.prompt[-1]["content"] + "\n"
        return start + self.llm_resp_clean

    def extract_infilling(self, answer: str, template_name: str):
        """
        Extracts code from an infilling type response.

        Args:
            answer (str): Answer extracted from the conversation.
            template_name (str): Name of the template used.

        Returns:
            str: Extracted code.
        """
        try:
            with open(f"./chat_templates/{template_name}.json", "r") as file:
                tokens = json.load(file)
        except FileNotFoundError:
            raise Exception(
                "File not found. Make sure the file exists and contains valid JSON data."
            )

        prefix = self.prompt_instance.prefix
        suffix = self.prompt_instance.suffix

        return tokens["answer_template"].format(
            prefix=prefix, suffix=suffix, answer=answer
        )
