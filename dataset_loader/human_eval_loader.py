from typing import List, Dict
from dataset_loader.dataset_loader import DatasetLoader


class HumanEvalLoader(DatasetLoader):
    def __init__(self) -> None:
        super().__init__()
        self.load_prompts()
        self.name = "Human Eval"

    def load_prompts(self) -> List[List[Dict[str, str]]]:
        print("Loading " + self.name + " prompts...")
        prompts = []

        print(self.name + " not implemented.\n")

        print(self.name + " prompts loaded.\n")
        self.prompts = prompts

    def format_code_responses(self, response: List[str]) -> List[str]:
        pass
