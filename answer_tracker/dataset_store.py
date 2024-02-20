from answer_tracker.task import Task
import numpy as np

from utils import get_pass_k


class DatasetStore:
    """
    Class for storing dataset data.
    """

    def __init__(self, name: str, max_chain_depth: int, tasks: list[Task]) -> None:
        """
        Initialize a DatasetStore object.

        :param name: Name of the dataset.
        :param max_chain_depth: Maximum chain depth.
        :param tasks: List of task objects.
        """
        self.name = name
        self.max_chain_depth = max_chain_depth
        self.tasks = tasks
        self.syntax_errors = {depth: 0 for depth in range(max_chain_depth)}
        self.other_errors = {depth: 0 for depth in range(max_chain_depth)}
        self.time_s = {depth: 0 for depth in range(max_chain_depth)}
        self.tokens_generated = {depth: 0 for depth in range(max_chain_depth)}
        self.passed = {depth: 0 for depth in range(max_chain_depth)}
        self.failed = {depth: 0 for depth in range(max_chain_depth)}
        self.num_answers = {depth: 0 for depth in range(max_chain_depth)}
        self.correct = {depth: 0 for depth in range(max_chain_depth)}
        self.pass_at_k = np.array([])

    def update_stats(self):
        """
        Update statistics for each task in the dataset.
        """
        total_answers, correct_answers = [], []
        for task in self.tasks:
            task.update_stats()

            for depth in range(task.max_chain_depth):
                self.syntax_errors[depth] += task.syntax_errors[depth]
                self.other_errors[depth] += task.other_errors[depth]
                self.time_s[depth] += task.time_to_gen[depth]
                self.tokens_generated[depth] += task.tokens_generated[depth]
                self.passed[depth] += task.passed[depth]
                self.failed[depth] += task.failed[depth]
                self.num_answers[depth] += task.num_answers[depth]
                self.correct[depth] += task.correct[depth]
                if task.answers[depth] and depth == 0:
                    total_answers.append(task.num_answers[depth])
                    correct_answers.append(task.correct[depth])

        k = max(total_answers)
        self.pass_at_k = get_pass_k(total_answers, correct_answers, k)

    def to_detailed_json(self):
        """
        Convert the dataset store to a summary JSON format.

        :param conf: Configuration object.
        """
        return {
            "Name": self.name,
            "Tasks": [task.detailed_json() for task in self.tasks],
        }

    def to_summary_json(self, conf):
        """
        Convert the dataset store to a brief summary JSON format.

        :param conf: Configuration object.
        """
        statistics = {
            depth: {
                "Syntax errors": self.syntax_errors[depth],
                "Other errors": self.other_errors[depth],
                "Time(sec)": self.time_s[depth],
                "Tokens generated": self.tokens_generated[depth],
                "Passed": self.passed[depth],
                "Failed": self.failed[depth],
                "Correct": self.correct[depth],
                "Amount of answers": self.num_answers[depth],
                "Success Rate": (
                    round((self.correct[depth] / self.num_answers[depth]) * 100, 1)
                    if depth > 0
                    else None
                ),
                "Pass@1": round(self.pass_at_k[0] * 100, 1) if depth == 0 else None,
                f"Pass@{conf.answers_per_task}": (
                    round(self.pass_at_k[1] * 100, 1) if depth == 0 else None
                ),
            }
            for depth in range(self.max_chain_depth)
            if any(
                [
                    self.syntax_errors[depth],
                    self.other_errors[depth],
                    self.time_s[depth],
                    self.tokens_generated[depth],
                    self.passed[depth],
                    self.failed[depth],
                    self.correct[depth],
                    self.num_answers[depth],
                    self.pass_at_k[0],
                    self.pass_at_k[1],
                ]
            )
        }

        statistics = {
            depth: {k: v for k, v in stats.items() if v is not None}
            for depth, stats in statistics.items()
        }

        return {
            "Name": self.name,
            "Statistics": statistics,
            "Configurations": {
                "Answers per task": conf.answers_per_task,
                "Max length": conf.max_length,
                "Temperature": conf.temperature,
                "Top p": conf.top_p,
            },
        }

    def to_brief_summary_json(self, conf):
        total_syntax_errors = sum(self.syntax_errors.values())
        total_other_errors = sum(self.other_errors.values())
        total_time_s = sum(self.time_s.values())
        total_tokens_generated = sum(self.tokens_generated.values())
        total_passed = sum(self.passed.values())
        total_failed = sum(self.failed.values())
        total_correct = sum(self.correct.values())
        amount_of_answers = sum(self.num_answers.values())
        pass_at_1 = round(self.pass_at_k[0] * 100, 1)

        return {
            "Syntax errors": total_syntax_errors,
            "Other errors": total_other_errors,
            "Time(sec)": total_time_s,
            "Tokens generated": total_tokens_generated,
            "Tokens/Sec": total_tokens_generated / total_time_s,
            "Passed": total_passed,
            "Failed": total_failed,
            "Correct": total_correct,
            "Amount of answers": amount_of_answers,
            "Avg Pass@1": pass_at_1,
            "Configurations": {
                "Answers per task": conf.answers_per_task,
                "Max length": conf.max_length,
                "Temperature": conf.temperature,
                "Top p": conf.top_p,
            },
        }
