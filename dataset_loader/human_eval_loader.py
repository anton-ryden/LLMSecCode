import os
import numpy as np
import copy

from human_eval.execution import check_correctness
from human_eval.evaluation import estimate_pass_at_k
from human_eval.data import read_problems, HUMAN_EVAL
from collections import defaultdict
from typing import List, Dict
from dataset_loader.dataset_loader import DatasetLoader
from utils import print_progress_bar

class HumanEvalLoader(DatasetLoader):
    def __init__(self) -> None:
        super().__init__()
        self.load_prompts()
        self.name = "HumanEval"

    def load_prompts(self) -> List[List[Dict[str, str]]]:
        print("Loading " + self.name + " prompts...")
        prompts = []

        # Receive prompt and inst from DatasetLoader
        system_prompt = self.system_prompt

        # Fetch all problems from HumanEval
        problems = read_problems(HUMAN_EVAL)
        data = [problems]

        for task_id, entry in data[0].items():

            prompt = copy.deepcopy(system_prompt)
            prompt.append(
                {"role": "user", "content": f"Complete the following Python code without any tests or explanation\n{entry['prompt']}\n{entry['test']}"}      
            )
            prompts.append({task_id: prompt})
        

        print(self.name + " prompts loaded.\n")
        self.prompts = prompts

    def format_code_responses(self, responses: List[str]) -> List[str]:
        return super().format_python_responses(responses)
    
    def test_code(self, ids: List[str], patch_list: List[List[str]]) -> List[Dict]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        run_eval_list = defaultdict(list)
        result_list = []
        bug_nr = 0
        temp = 0
        os.makedirs("results/", exist_ok=True)

        print("Starting testing for: " + self.name)
        for id, bugs in zip(ids, patch_list):
            test_list = []
            for patch_nr, patch in enumerate(bugs, start=1):

                
                run_eval_list = run_eval(id, patch_nr, patch, run_eval_list)
                passed_count = 0
                failed_count = 0

                if run_eval_list[id][patch_nr-1][1]['passed']:
                    passed_count += 1
                else:
                    failed_count += 1
                test_run_info = {
                    "task_id": id,
                    "Passed": passed_count,
                    "Failed": failed_count
                }

                if run_eval_list[id][patch_nr-1][1]['result']:
                    syntax_error = {
                        "syntax_error": run_eval_list[id][patch_nr-1][1]['result'],
                    }
                else:
                    syntax_error = {
                        "syntax_error": "null",
                        "error_message": "Empty file, could not extract any code",
                    }

                test_list.append({**test_run_info, **syntax_error})

                print_progress_bar(
                    len(bugs) * bug_nr + patch_nr,
                    len(ids) * len(bugs),
                )
                if patch_nr+1 > temp: temp = patch_nr

            result_list.append(test_list)
            bug_nr += 1


        k : List[int] = [1, temp]
        pass_at_k = cal_pass_at_k(run_eval_list, k)
        pass1 = pass_at_k[0].tolist()
        passX = pass_at_k[1].tolist()

        avg_pass1 = sum(pass1) / len(pass1)
        avg_passX = sum(passX) / len(passX)
        
        pass_result = {
            "Avg Pass@1": avg_pass1,
            "Pass@1": pass1,
            f"Avg Pass@{temp}" : avg_passX,
            f"Pass@{temp}": passX
        }
        result_list.append(pass_result)
        
        print("\n")
        print("pass_at_k: ", pass_at_k)
        return result_list
    
def cal_pass_at_k(results, k):
    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = []
    for k in ks:
        pass_at_k.append(estimate_pass_at_k(total, correct, k))

    return pass_at_k

def run_eval(bug_id, patch_id, patch, results):
    problem_file = HUMAN_EVAL
    timeout = 3.0
    problems = read_problems(problem_file)
    completion_id = patch_id

    # Check generated samples against test suites.
    futures = []

    task_id = bug_id
    completion = patch
    future = check_correctness(problems[task_id], completion, timeout, completion_id)
    futures.append(future)

    correctness_result = future
    results[correctness_result["task_id"]].append((correctness_result["completion_id"], correctness_result))

    return results