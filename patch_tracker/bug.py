import copy

from patch_tracker.patch import Patch


class Bug:
    """
    Class for storing bug.
    """

    def __init__(
        self, id: str, prompt: str, max_chain_depth: int, patches_per_bug: int
    ) -> None:
        """
        Initialize a Bug object.

        :param id: Unique identifier for the bug.
        :param prompt: Prompt for generating patches.
        :param max_chain_depth: Maximum chain depth.
        :param patches_per_bug: Number of patches per bug.
        """
        self.id = id
        self.prompt = prompt
        self.max_chain_depth = max_chain_depth
        self.syntax_errors = {depth: 0 for depth in range(max_chain_depth)}
        self.other_errors = {depth: 0 for depth in range(max_chain_depth)}
        self.time_to_gen = {depth: 0 for depth in range(max_chain_depth)}
        self.tokens_generated = {depth: 0 for depth in range(max_chain_depth)}
        self.passed = {depth: 0 for depth in range(max_chain_depth)}
        self.failed = {depth: 0 for depth in range(max_chain_depth)}
        self.patches = [[] for _ in range(max_chain_depth)]
        self.num_patches = {depth: 0 for depth in range(max_chain_depth)}
        self.correct = {depth: 0 for depth in range(max_chain_depth)}

        p = Patch(id, prompt, 0)
        for _ in range(patches_per_bug):
            self.add_patch(copy.copy(p))

    def add_patch(self, patch: Patch):
        """
        Add a patch to the bug.

        :param patch: Patch object to be added.
        """
        depth = patch.chain_depth
        self.patches[depth].append(patch)

    def update_stats(self):
        """
        Update statistics for the bug.
        """
        for i in range(self.max_chain_depth):
            for patch in self.patches[i]:
                self.time_to_gen[i] += patch.time_to_gen
                self.tokens_generated[i] += patch.tokens_generated
                self.passed[i] += patch.passed
                self.failed[i] += patch.failed
                self.syntax_errors[i] += 1 if patch.syntax_error else 0
                self.other_errors[i] += 1 if patch.other_error else 0
                self.num_patches[i] += 1
                self.correct[i] += 1 if patch.failed == 0 and patch.passed > 0 else 0

    def detailed_json(self):
        """
        Convert the bug to a detailed JSON format.

        :return: Detailed JSON representation of the bug.
        """
        patches = []
        for depth in range(self.max_chain_depth):
            for patch in self.patches[depth]:
                if patch != []:
                    patches.append(patch.detailed_json())

        return {"Id": self.id, "Prompt": self.prompt, "Patches": patches}

    def summary_json(self):
        """
        Convert the bug to a summary JSON format.

        :return: Summary JSON representation of the bug.
        """
        statistics = {
            depth: {
                "Syntax errors": self.syntax_errors[depth],
                "Other errors": self.other_errors[depth],
                "Time(sec)": self.time_to_gen[depth],
                "Tokens generated": self.tokens_generated[depth],
                "Passed": self.passed[depth],
                "Failed": self.failed[depth],
                "Correct": self.correct[depth],
            }
            for depth in range(self.max_chain_depth)
            if any(
                [
                    self.syntax_errors[depth],
                    self.other_errors[depth],
                    self.time_to_gen[depth],
                    self.tokens_generated[depth],
                    self.passed[depth],
                    self.failed[depth],
                    self.correct[depth],
                ]
            )
        }
        return {
            "Id": self.id,
            "Prompt": self.prompt,
            "Statistics": statistics,
        }
