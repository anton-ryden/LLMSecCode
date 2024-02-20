import copy

from answer_tracker.answer import Answer


class Task:
    """
    Class for storing tasks.
    """

    def __init__(
        self, id: str, prompt: str, max_chain_depth: int, answers_per_task: int
    ) -> None:
        """
        Initialize a Task object.

        :param id: Unique identifier for the task.
        :param prompt: Prompt for generating answers.
        :param max_chain_depth: Maximum chain depth.
        :param answers_per_task: Number of answers per task.
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
        self.answers = [[] for _ in range(max_chain_depth)]
        self.num_answers = {depth: 0 for depth in range(max_chain_depth)}
        self.correct = {depth: 0 for depth in range(max_chain_depth)}

        p = Answer(id, prompt, 0)
        for _ in range(answers_per_task):
            self.add_answer(copy.copy(p))

    def add_answer(self, answer: Answer):
        """
        Add a answer to the task.

        :param answer: Answer object to be added.
        """
        depth = answer.chain_depth
        self.answers[depth].append(answer)

    def update_stats(self):
        """
        Update statistics for the task.
        """
        for i in range(self.max_chain_depth):
            for answer in self.answers[i]:
                self.time_to_gen[i] += answer.time_to_gen
                self.tokens_generated[i] += answer.tokens_generated
                self.passed[i] += answer.passed
                self.failed[i] += answer.failed
                self.syntax_errors[i] += 1 if answer.syntax_error else 0
                self.other_errors[i] += 1 if answer.other_error else 0
                self.num_answers[i] += 1
                self.correct[i] += 1 if answer.failed == 0 and answer.passed > 0 else 0

    def detailed_json(self):
        """
        Convert the task to a detailed JSON format.

        :return: Detailed JSON representation of the task.
        """
        answers = []
        for depth in range(self.max_chain_depth):
            for answer in self.answers[depth]:
                if answer != []:
                    answers.append(answer.detailed_json())

        return {"Id": self.id, "Prompt": self.prompt, "Answers": answers}

    def summary_json(self):
        """
        Convert the task to a summary JSON format.

        :return: Summary JSON representation of the task.
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
