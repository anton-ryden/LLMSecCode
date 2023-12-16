from typing import List, Dict
from dataset_loader.dataset_loader import DatasetLoader


class Defect4JLoader(DatasetLoader):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Defect4J"
        self.load_prompts()

    def load_prompts(self) -> List[List[Dict[str, str]]]:
        print("Loading " + self.name + " prompts...")
        prompts = []

        print(self.name + " not implemented.\n")

        print(self.name + " prompts loaded.\n")
        self.prompts = prompts

    def format_code_responses(self, response: List[str]) -> List[str]:
        pass

    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        pass
