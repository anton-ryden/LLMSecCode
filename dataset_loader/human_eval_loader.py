import os
import copy

from human_eval.execution import check_correctness
from human_eval.data import read_problems, HUMAN_EVAL
from collections import defaultdict
from dataset_loader.dataset_loader import DatasetLoader
from data_structures.task import Task
from data_structures.answer import Answer
from data_structures.prompt_store import PromptsStore


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
        self.area = "CodeGen"

    def load_prompts(self) -> None:
        """
        Load prompts for HumanEval dataset.

        :param max_chain_depth: Maximum chain depth.
        :param answers_per_task: Number of answers per task.
        """
        print(f"Loading {self.name} prompts...")
        prompts = PromptsStore(self.area)

        # Fetch all problems from HumanEval
        problems = read_problems(HUMAN_EVAL)
        data = [problems]

        for task_id, entry in data[0].items():
            prompts.add_conversation(task_id, entry["prompt"], "python")
            prompts.add_infilling()

        print(f"{self.name} prompts loaded.\n")
        self.prompts = prompts

    def test_code(self, answer: Answer) -> None:
        """
        Test the provided answer.

        :param answer: Answer object.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        run_eval_list = defaultdict(list)
        run_eval_list = run_eval(answer.id, answer.code, run_eval_list)

        if run_eval_list[answer.id][0][1]["passed"]:
            answer.passed = 1
        else:
            answer.failed = 1

        # If it contains anything other than this it is a syntax error
        result = run_eval_list[answer.id][0][1]["result"]
        if result == "failed: " or result == "passed":
            answer.syntax_error = False

        else:
            if result == "timed out":
                answer.other_error = True
            else:
                answer.syntax_error = True
            answer.error_message = result


def run_eval(task_id, answer, results):
    """
    Evaluate a answer for a task using human evaluation.

    :param task_id: ID of the task being evaluated.
    :param answer: Answer or completion to be evaluated.
    :param results: Dictionary storing evaluation results.

    :return: Updated evaluation results.

    Evaluates the given answer against problems from HUMAN_EVAL, checks correctness with a timeout,
    and updates the results dictionary accordingly.
    """
    problem_file = HUMAN_EVAL
    timeout = 3.0
    problems = read_problems(problem_file)

    # Check generated samples against test suites.
    futures = []

    task_id = task_id
    completion = answer
    future = check_correctness(problems[task_id], completion, timeout)
    futures.append(future)

    correctness_result = future
    results[correctness_result["task_id"]].append(
        (correctness_result["completion_id"], correctness_result)
    )

    return results
