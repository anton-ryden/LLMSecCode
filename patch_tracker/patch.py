import copy


class Patch:
    def __init__(self, id: str, prompt: list[dict], chain_depth: int) -> None:
        """
        Initialize a Patch object.

        :param id: Unique identifier for the patch.
        :param prompt: Prompt for generating the patch.
        :param chain_depth: Depth of the patch in the chain.
        """
        self.id = id
        self.prompt = prompt
        self.chain_depth = chain_depth
        self.syntax_error = False
        self.other_error = False
        self.error_message = ""
        self.llm_resp = ""
        self.llm_resp_clean = ""
        self.code = ""
        self.time_to_gen = 0
        self.tokens_generated = 0
        self.passed = 0
        self.failed = 0

    def get_next_chain(self) -> "Patch":
        """
        Get the next patch in the chain.

        :return: A new Patch object representing the next patch in the chain.
        """
        new_prompt = copy.deepcopy(self.prompt)
        new_prompt.append({"role": "assistant", "content": self.code})
        new_prompt.append(
            {"role": "user", "content": "It is not quite correct, try again."}
        )
        return Patch(self.id, new_prompt, self.chain_depth + 1)

    def detailed_json(self):
        """
        Convert the patch to a detailed JSON format.

        :return: Detailed JSON representation of the patch.
        """
        return {
            "Id": self.id,
            "Prompt": self.prompt,
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
