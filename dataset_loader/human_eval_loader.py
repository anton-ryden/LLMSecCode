import os
import copy

from human_eval.execution import check_correctness
from human_eval.data import read_problems, HUMAN_EVAL
from collections import defaultdict
from dataset_loader.dataset_loader import DatasetLoader
from patch_tracker.bug import Bug
from patch_tracker.patch import Patch


class HumanEvalLoader(DatasetLoader):
    """
    Class for loading and testing the dataset HumanEval.
    """

    def __init__(self) -> None:
        """
        Initialize the HumanEvalLoader.
        """
        super().__init__()
        self.name = "HumanEval"

    def load_prompts(self, max_chain_depth: int, patches_per_bug: int) -> None:
        """
        Load prompts for HumanEval dataset.

        :param max_chain_depth: Maximum chain depth.
        :param patches_per_bug: Number of patches per bug.
        """
        print("Loading " + self.name + " prompts...")
        bugs = []

        # Receive prompt and inst from DatasetLoader
        system_prompt = self.system_prompt

        # Fetch all problems from HumanEval
        problems = read_problems(HUMAN_EVAL)
        data = [problems]

        for task_id, entry in data[0].items():
            prompt = copy.deepcopy(system_prompt)
            prompt.append(
                {
                    "role": "user",
                    "content": f"Write a Python function `{entry['entry_point']}` to solve the following problem:\n{entry['prompt']}",
                }
            )
            bugs.append(Bug(task_id, prompt, max_chain_depth, patches_per_bug))

        print(self.name + " prompts loaded.\n")
        self.bugs = bugs

    def test_code(self, patch: Patch) -> None:
        """
        Test the provided patch.

        :param patch: Patch object.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        run_eval_list = defaultdict(list)
        run_eval_list = run_eval(patch.id, patch.code, run_eval_list)

        if run_eval_list[patch.id][0][1]["passed"]:
            patch.passed = 1
        else:
            patch.failed = 1

        # If it contains anything other than this it is a syntax error
        result = run_eval_list[patch.id][0][1]["result"]
        if result == "failed: " or result == "passed":
            patch.syntax_error = False

        else:
            if result == "timed out":
                patch.other_error = True
            else:
                patch.syntax_error = True
            patch.error_message = result


def run_eval(bug_id, patch, results):
    """
    Evaluate a patch for a bug using human evaluation.

    :param bug_id: ID of the bug being evaluated.
    :param patch: Patch or completion to be evaluated.
    :param results: Dictionary storing evaluation results.

    :return: Updated evaluation results.

    Evaluates the given patch against problems from HUMAN_EVAL, checks correctness with a timeout,
    and updates the results dictionary accordingly.
    """
    problem_file = HUMAN_EVAL
    timeout = 3.0
    problems = read_problems(problem_file)

    # Check generated samples against test suites.
    futures = []

    task_id = bug_id
    completion = patch
    future = check_correctness(problems[task_id], completion, timeout)
    futures.append(future)

    correctness_result = future
    results[correctness_result["task_id"]].append(
        (correctness_result["completion_id"], correctness_result)
    )

    return results
